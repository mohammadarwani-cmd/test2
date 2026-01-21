import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import time
import json
import os
import hashlib

# 安全导入 scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 设置页面宽屏模式
st.set_page_config(page_title="动量轮动策略回测 Pro", layout="wide")

# ==========================================
# 0. 配置持久化管理与辅助函数
# ==========================================
CONFIG_FILE = 'strategy_config.json'

# 默认标的池 (代码)
DEFAULT_CODES = ["518880", "588000", "513100", "510180"]

DEFAULT_PARAMS = {
    'lookback': 25,
    'smooth': 3,
    'threshold': 0.005,
    'min_holding': 3,
    'allow_cash': True,
    'mom_method': 'Risk-Adjusted (稳健)', 
    'selected_codes': DEFAULT_CODES
}

@st.cache_data(ttl=3600)
def get_all_etf_info():
    """
    获取全市场ETF列表，用于搜索和中文名称映射
    返回: map_dict (code -> name), search_list (["code | name", ...])
    """
    try:
        # 获取所有ETF实时行情，包含代码和名称
        df = ak.fund_etf_spot_em()
        if df.empty:
            return {}, []
        
        # 构建字典映射
        code_map = dict(zip(df['代码'], df['名称']))
        
        # 构建搜索列表
        search_options = [f"{row['代码']} | {row['名称']}" for _, row in df.iterrows()]
        
        return code_map, search_options
    except Exception as e:
        # 如果接口失败，返回基础默认值
        st.warning(f"无法获取全市场ETF名称列表，将使用默认代码显示。错误: {e}")
        return {}, []

def load_config():
    """从本地文件加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved_config)
                return config
        except:
            return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        st.error(f"配置保存失败: {e}")

# ==========================================
# 1. 数据获取 (Data Fetching)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(codes, start_date="20200101", end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    data_dict = {}
    valid_codes = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(codes):
        status_text.text(f"正在获取 {code} 数据...")
        try:
            # 使用 akshare 获取后复权数据
            df = ak.fund_etf_hist_em(
                symbol=code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date, 
                adjust="hfq"
            )
            
            if df.empty or '收盘' not in df.columns:
                continue
                
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df = df[['收盘']].rename(columns={'收盘': code})
            data_dict[code] = df
            valid_codes.append(code)
            
        except Exception as e:
            print(f"Error fetching {code}: {e}")
            
        progress_bar.progress((i + 1) / len(codes))
        
    status_text.empty()
    progress_bar.empty()
    
    if not data_dict:
        return pd.DataFrame(), []
        
    # 合并数据
    df_merged = pd.concat(data_dict.values(), axis=1).sort_index()
    df_merged.ffill(inplace=True) # 填充缺失值
    df_merged.dropna(inplace=True) # 删除最开始的空值
    
    return df_merged, valid_codes

# ==========================================
# 2. 核心策略逻辑 (Core Logic)
# ==========================================
def calculate_momentum(df, lookback, smooth, method):
    """计算动量分数"""
    returns = df.pct_change()
    
    if method == 'Risk-Adjusted (稳健)':
        # 收益 / 波动率
        mom = df.pct_change(lookback)
        vol = returns.rolling(lookback).std() * np.sqrt(252)
        score = mom / (vol + 1e-6) # 避免除零
        
    elif method == 'Slope (斜率)':
        if not HAS_SCIPY:
            st.warning("Scipy 未安装，降级为累计收益率模式")
            score = df.pct_change(lookback)
        else:
            # 线性回归斜率 (较慢)
            def get_slope(y):
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
            score = df.rolling(lookback).apply(get_slope, raw=True)
            
    else: # Simple Return
        score = df.pct_change(lookback)
        
    # 平滑处理
    if smooth > 1:
        score = score.rolling(smooth).mean()
        
    return score

def backtest_strategy(df_price, df_score, threshold, min_holding, allow_cash):
    """
    执行回测
    :param df_price: 价格数据
    :param df_score: 动量分数数据
    :param threshold: 换仓阈值 (例如 0.05 代表 5%)
    :param min_holding: 最小持仓天数
    :param allow_cash: 是否允许空仓
    """
    
    # 初始化记录
    dates = df_price.index
    cash = 1.0
    holdings = [] # 记录每天持有的标的
    equity = []   # 记录每天的净值
    operations = [] # 记录操作日志
    
    current_holding = 'CASH'
    days_held = 0
    
    # 从足够数据的第一天开始
    start_idx = 0
    while start_idx < len(dates) and df_score.iloc[start_idx].isna().all():
        start_idx += 1
        
    # 填充前期空缺
    for _ in range(start_idx):
        equity.append(1.0)
        holdings.append('CASH')
    
    # 正式回测循环
    for i in range(start_idx, len(dates)):
        today = dates[i]
        
        # 获取当天各标的动量分数
        scores = df_score.iloc[i]
        
        # 排除无效数据的标的
        valid_scores = scores.dropna()
        
        best_target = 'CASH'
        best_score = -999
        
        if not valid_scores.empty:
            # 找到分数最高的
            best_target = valid_scores.idxmax()
            best_score = valid_scores.max()
        
        # 决策逻辑
        action = None
        
        # 如果动量均为负且允许空仓 -> 切换到现金
        if allow_cash and best_score < 0:
            target_holding = 'CASH'
        else:
            target_holding = best_target
            
        # 判断是否换仓
        if current_holding == 'CASH':
            # 从现金买入，只要有正向动量
            if target_holding != 'CASH':
                current_holding = target_holding
                days_held = 1
                action = f"买入 {current_holding}"
        else:
            # 已持仓，判断是否切换
            days_held += 1
            
            if days_held >= min_holding:
                # 1. 如果当前持仓动量转负且允许空仓 -> 卖出
                if allow_cash and scores.get(current_holding, -999) < 0:
                    action = f"止损/空仓: 卖出 {current_holding}"
                    current_holding = 'CASH'
                    days_held = 0
                
                # 2. 如果有更好的标的，且超过阈值 -> 换仓
                elif target_holding != current_holding and target_holding != 'CASH':
                    curr_score = scores.get(current_holding, -999)
                    # 只有当 新标的分数 - 当前标的分数 > 阈值 时才换
                    if (valid_scores[target_holding] - curr_score) > threshold:
                        action = f"换仓: {current_holding} -> {target_holding} (Diff: {(valid_scores[target_holding] - curr_score):.2%})"
                        current_holding = target_holding
                        days_held = 1
        
        if action:
            operations.append({
                '日期': today,
                '操作': action,
                '标的': current_holding
            })
            
        holdings.append(current_holding)
        
        # 计算当天净值变化 (使用当天的涨跌幅)
        # 注意：这里简化处理，假设以收盘价成交，实际收益从持仓第二天开始计算
        if i > 0:
            prev_holding = holdings[-2]
            if prev_holding == 'CASH':
                step_ret = 0
            else:
                # 今天的净值变化取决于昨天持有的标的在今天的涨跌
                if prev_holding in df_price.columns:
                    step_ret = df_price[prev_holding].iloc[i] / df_price[prev_holding].iloc[i-1] - 1
                else:
                    step_ret = 0
            
            equity.append(equity[-1] * (1 + step_ret))
        else:
            equity.append(1.0)

    # 创建结果 DataFrame
    df_res = pd.DataFrame({
        '日期': dates[:len(equity)],
        '净值': equity,
        '当前持仓': holdings
    }).set_index('日期')
    
    return df_res, pd.DataFrame(operations)

# ==========================================
# 3. 界面与主程序 (UI & Main)
# ==========================================
def main():
    # --- 加载配置 ---
    config = load_config()
    
    # --- 获取全市场 ETF 映射 (用于搜索) ---
    etf_name_map, etf_search_list = get_all_etf_info()

    # --- 侧边栏：参数设置 ---
    with st.sidebar:
        st.header("⚙️ 策略参数设置")
        
        # 标的搜索/选择逻辑
        st.subheader("1. 标的池选择")
        
        # 预处理默认选中项：将纯代码转换为 "代码 | 名称" 格式，以便在多选框中默认选中
        default_options = []
        if etf_search_list:
            for code in config.get('selected_codes', DEFAULT_CODES):
                # 尝试找到对应的完整描述
                match = [s for s in etf_search_list if s.startswith(code)]
                if match:
                    default_options.append(match[0])
                else:
                    # 如果找不到名字（比如非ETF基金），就直接显示代码
                    default_options.append(code)
        else:
            default_options = config.get('selected_codes', DEFAULT_CODES)

        # 渲染多选框
        selected_search_items = st.multiselect(
            "搜索并添加标的 (支持中文名称)",
            options=etf_search_list if etf_search_list else default_options,
            default=default_options,
            help="输入代码或中文名称进行搜索"
        )
        
        # 将选中的 "代码 | 名称" 解析回纯代码列表
        selected_codes = []
        for item in selected_search_items:
            # item 格式可能是 "518880 | 黄金ETF" 或 "518880"
            code = item.split('|')[0].strip()
            selected_codes.append(code)

        st.subheader("2. 回测参数")
        start_date = st.date_input("开始日期", value=pd.to_datetime("2020-01-01"))
        lookback = st.slider("动量窗口 (Lookback)", 5, 60, config['lookback'])
        smooth = st.slider("平滑窗口 (Smooth)", 1, 10, config['smooth'])
        threshold = st.number_input("换仓阈值 (Threshold)", 0.0, 0.1, config['threshold'], step=0.001, format="%.3f")
        min_holding = st.number_input("最小持仓天数", 1, 20, config['min_holding'])
        
        st.subheader("3. 策略逻辑")
        mom_method = st.selectbox("动量算法", ["Risk-Adjusted (稳健)", "Simple Return (激进)", "Slope (斜率)"], index=0)
        allow_cash = st.checkbox("允许空仓 (负动量时持有现金)", value=config['allow_cash'])

        if st.button("开始回测", type="primary"):
            # 保存配置
            new_config = {
                'lookback': lookback,
                'smooth': smooth,
                'threshold': threshold,
                'min_holding': min_holding,
                'allow_cash': allow_cash,
                'mom_method': mom_method,
                'selected_codes': selected_codes
            }
            save_config(new_config)
            st.session_state['run_backtest'] = True
        
    # --- 主区域 ---
    st.title("📈 动量轮动策略回测 Pro")
    
    if len(selected_codes) < 2:
        st.info("请在左侧至少选择两个标的进行对比。")
        return

    # 数据获取
    with st.spinner("正在获取历史数据..."):
        df_price, valid_codes = get_data(selected_codes, start_date=start_date.strftime("%Y%m%d"))
    
    if df_price.empty:
        st.error("无法获取有效数据，请检查代码或网络。")
        return

    # 计算动量
    df_score = calculate_momentum(df_price, lookback, smooth, mom_method)
    
    # 执行回测
    df_res, df_ops = backtest_strategy(df_price, df_score, threshold, min_holding, allow_cash)
    
    # 计算全市场表现 (等权平均) 作为基准
    df_res['全市场表现'] = df_price.mean(axis=1) / df_price.mean(axis=1).iloc[0]
    
    # ==========================================
    # NEW FEATURE: 下一个交易日操作建议
    # ==========================================
    st.markdown("---")
    st.subheader("📢 下一交易日操作建议")
    
    # 获取最后一天的数据进行判断
    last_date = df_price.index[-1]
    last_scores = df_score.iloc[-1].dropna()
    
    if not last_scores.empty:
        # 获取当前持仓（回测最后一天的持仓）
        current_holding_code = df_res['当前持仓'].iloc[-1]
        
        # 获取当前最强标的
        best_target_code = last_scores.idxmax()
        best_score_val = last_scores.max()
        
        # 获取中文名辅助函数
        def get_name(c):
            return f"{c} ({etf_name_map.get(c, '未知')})" if c != 'CASH' else "现金 (CASH)"

        current_holding_name = get_name(current_holding_code)
        best_target_name = get_name(best_target_code)
        
        # 计算建议逻辑
        suggestion_type = "hold" # hold, switch, sell
        suggestion_msg = ""
        
        # 获取当前持有标的的分数 (如果是CASH则设为0，或者不参与比较)
        current_score_val = last_scores.get(current_holding_code, -999) if current_holding_code != 'CASH' else 0
        
        diff = best_score_val - current_score_val
        
        if current_holding_code == 'CASH':
            if best_score_val > 0:
                suggestion_type = "buy"
                suggestion_msg = f"建议买入: **{best_target_name}** (动量值: {best_score_val:.4f} > 0)"
            else:
                suggestion_type = "wait"
                suggestion_msg = "建议观望: 所有标的动量均较弱，继续持有现金。"
        else:
            # 已经有持仓
            if best_target_code != current_holding_code:
                if diff > threshold:
                    suggestion_type = "switch"
                    suggestion_msg = f"建议调仓: 从 **{current_holding_name}** 切换至 **{best_target_name}** (动量差: {diff:.2%} > 阈值 {threshold})"
                else:
                    suggestion_type = "hold_weak"
                    suggestion_msg = f"建议保持: 虽然 **{best_target_name}** 更强，但优势 ({diff:.2%}) 未超过阈值 ({threshold})，继续持有当前标的。"
            else:
                # 最强就是当前持有
                if allow_cash and current_score_val < 0:
                    suggestion_type = "sell"
                    suggestion_msg = f"建议卖出: **{current_holding_name}** 动量转负 ({current_score_val:.4f})，建议清仓止损。"
                else:
                    suggestion_type = "hold_strong"
                    suggestion_msg = f"建议持有: **{current_holding_name}** 依然是当前最强标的 (动量值: {current_score_val:.4f})。"

        # 渲染建议 UI
        cols = st.columns([1, 3])
        with cols[0]:
            st.metric("数据日期", last_date.strftime("%Y-%m-%d"))
        with cols[1]:
            if suggestion_type in ["switch", "sell", "buy"]:
                st.error(f"🚨 {suggestion_msg}")
            elif suggestion_type in ["hold_weak", "wait"]:
                st.warning(f"✋ {suggestion_msg}")
            else:
                st.success(f"✅ {suggestion_msg}")
            
            # 额外显示前三名详情
            top3 = last_scores.sort_values(ascending=False).head(3)
            detail_str = " | ".join([f"{get_name(c)}: {s:.4f}" for c, s in top3.items()])
            st.caption(f"当前动量前三: {detail_str}")

    st.markdown("---")

    # ==========================================
    # 可视化 (Charts)
    # ==========================================
    
    # 1. 净值曲线
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("策略净值表现")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['净值'], name='策略净值', line=dict(color='#2E86C1', width=2)))
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['全市场表现'], name='等权基准', line=dict(color='#BDC3C7', dash='dot')))
        
        # 标记买卖点
        buy_indices = df_res[df_res['当前持仓'] != df_res['当前持仓'].shift(1)].index
        fig.add_trace(go.Scatter(
            x=buy_indices, 
            y=df_res.loc[buy_indices, '净值'],
            mode='markers',
            marker=dict(color='orange', size=8, symbol='triangle-up'),
            name='调仓点'
        ))
        
        fig.update_layout(height=450, margin=dict(l=0,r=0,t=30,b=0), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # 关键指标卡片
        total_ret = df_res['净值'].iloc[-1] - 1
        annual_ret = (1 + total_ret) ** (252 / len(df_res)) - 1
        daily_ret = df_res['净值'].pct_change().dropna()
        volatility = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - 0.02) / volatility if volatility > 0 else 0
        max_drawdown = (df_res['净值'] / df_res['净值'].cummax() - 1).min()
        
        st.subheader("绩效指标")
        st.metric("总收益率", f"{total_ret:.2%}")
        st.metric("年化收益率", f"{annual_ret:.2%}")
        st.metric("最大回撤", f"{max_drawdown:.2%}", delta_color="inverse")
        st.metric("夏普比率", f"{sharpe:.2f}")

    # 2. 动态回撤图
    st.subheader("动态回撤")
    dd = df_res['净值'] / df_res['净值'].cummax() - 1
    fig_dd = px.area(x=dd.index, y=dd, color_discrete_sequence=['#E74C3C'])
    fig_dd.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_dd, use_container_width=True)

    # 3. 详细数据表格
    st.subheader("持仓详情与信号")
    
    # 组合详细数据：价格 + 动量 + 信号
    df_details = df_price.copy()
    # 增加后缀避免重名
    df_details.columns = [f"{c}_价格" for c in df_details.columns]
    
    # 加入当前持仓列
    df_details['当前持仓'] = df_res['当前持仓']
    
    # 计算每段持仓的收益
    df_details['段内收益'] = 0.0
    df_details['持仓天数'] = 0
    
    current_h = 'CASH'
    start_p = 1.0
    days = 0
    
    seg_returns = []
    holding_days = []
    
    for i in range(len(df_details)):
        date = df_details.index[i]
        h = df_details['当前持仓'].iloc[i]
        
        if h != current_h:
            # 换仓了，重置
            current_h = h
            days = 1
            if h != 'CASH' and f"{h}_价格" in df_details.columns:
                start_p = df_details[f"{h}_价格"].iloc[i]
            else:
                start_p = 1.0
            seg_returns.append(0.0)
        else:
            days += 1
            if h != 'CASH' and f"{h}_价格" in df_details.columns:
                curr_p = df_details[f"{h}_价格"].iloc[i]
                ret = (curr_p / start_p) - 1
                seg_returns.append(ret)
            else:
                seg_returns.append(0.0)
        
        holding_days.append(days)
        
    df_details['段内收益'] = seg_returns
    df_details['持仓天数'] = holding_days
    
    # 映射操作记录到表格
    df_details['操作'] = ""
    for op in df_ops.to_dict('records'):
        d = op['日期']
        if d in df_details.index:
            df_details.at[d, '操作'] = op['操作']

    df_details['总资产'] = df_res['净值']
    df_details['全市场表现'] = df_res['全市场表现']
    
    # 倒序显示
    df_show = df_details.sort_index(ascending=False).head(500)
    
    # 格式化
    if not df_show.empty:
        df_show['段内收益'] = df_show['段内收益'] * 100 # 转百分比数值方便显示
        
        col_config = {
            "持仓天数": st.column_config.NumberColumn("持仓天数", help="当前连续持仓天数"),
            "段内收益": st.column_config.NumberColumn("段内收益", help="本段持仓期间的累计收益率", format="%.2f%%"),
            "操作": st.column_config.TextColumn("调仓操作", width="medium"),
            "总资产": st.column_config.NumberColumn("总资产", format="%.4f"),
            "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
        }
        
        # 动态添加价格列配置
        final_cols = ["当前持仓", "段内收益", "持仓天数", "总资产", "操作"]
        
        # 显示表格
        st.dataframe(
            df_show[final_cols],
            column_config=col_config,
            use_container_width=True,
            height=400
        )

if __name__ == "__main__":
    main()
