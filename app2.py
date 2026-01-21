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

# ==========================================
# 0. 配置与常数 (Config & Constants)
# ==========================================
CONFIG_FILE = 'strategy_config.json'

# --- FEATURE ADDITION: ETF Chinese Name Mapping ---
ETF_NAME_MAP = {
    # Gold / Commodities
    "518880": "黄金ETF", "518800": "黄金ETF-SX", "159980": "大宗商品ETF", "159981": "能源化工ETF", "159985": "豆粕ETF",
    # Tech / Growth
    "588000": "科创50ETF", "588080": "科创板50ETF", "159915": "创业板ETF", "159949": "创业板50",
    "512660": "军工ETF", "512480": "半导体ETF", "512760": "芯片ETF", "515030": "新能源车ETF", "515790": "光伏ETF",
    "515050": "5G ETF", "159995": "芯片ETF-SZ", "512690": "酒ETF", "512980": "传媒ETF",
    # Broad Market
    "510300": "沪深300ETF", "510050": "上证50ETF", "510500": "中证500ETF", "159901": "深100ETF",
    "510180": "上证180ETF", "159902": "中小板ETF", "159922": "中证500ETF-SZ", "510880": "红利ETF",
    # Cross Border / US
    "513100": "纳指ETF", "513050": "中概互联ETF", "513500": "标普500ETF", "159941": "纳指ETF-SZ",
    "513330": "恒生互联ETF", "513060": "恒生医疗ETF", "159920": "恒生ETF", "513180": "恒生科技ETF",
    # Bonds / Money
    "511260": "十年国债ETF", "511010": "国债ETF", "511880": "银华日利", "511990": "华宝添益"
}

def get_etf_display_name(code):
    """Helper to get Code - Name string"""
    name = ETF_NAME_MAP.get(code, "未知名称")
    return f"{code} - {name}"

# 默认标的池
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

def load_config():
    """从本地文件加载配置"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved_config)
                return config
        except Exception as e:
            st.warning(f"无法读取配置文件，使用默认设置: {e}")
            return DEFAULT_PARAMS
    return DEFAULT_PARAMS

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        st.error(f"无法保存配置文件: {e}")

# ==========================================
# 1. 数据获取 (Data Fetching)
# ==========================================
@st.cache_data(ttl=3600*4)  # 缓存4小时
def get_single_data(code):
    """获取单个ETF的历史数据"""
    try:
        # 使用 akshare 获取 ETF 历史数据
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date="20150101", adjust="hfq")
        if df.empty:
            return None
        
        # 重命名列以符合习惯
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume"
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        
        # 仅保留收盘价用于计算动量，可扩展
        return df[['close']]
    except Exception as e:
        print(f"Error fetching {code}: {e}")
        return None

@st.cache_data(ttl=3600*4)
def get_all_data(codes):
    """获取所有选中ETF的数据并合并"""
    data_dict = {}
    valid_codes = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(codes):
        status_text.text(f"正在获取 {get_etf_display_name(code)} 数据...")
        df = get_single_data(code)
        if df is not None:
            data_dict[code] = df['close']
            valid_codes.append(code)
        progress_bar.progress((i + 1) / len(codes))
        time.sleep(0.1)  # 避免请求过快
        
    status_text.empty()
    progress_bar.empty()
    
    if not data_dict:
        return pd.DataFrame(), []
    
    # 合并为一个 DataFrame
    df_combined = pd.DataFrame(data_dict)
    # 前向填充处理停牌，再去除全空行
    df_combined = df_combined.fillna(method='ffill').dropna()
    
    return df_combined, valid_codes

# ==========================================
# 2. 动量计算核心 (Momentum Logic)
# ==========================================
def calculate_momentum_score(df, method, lookback, smooth):
    """计算动量分数"""
    returns = df.pct_change()
    
    if method == 'Rate of Change (普通)':
        # 简单收益率: (Price_t / Price_{t-n}) - 1
        # 使用平滑后的价格计算
        prices_smoothed = df.rolling(window=smooth).mean()
        mom = prices_smoothed.pct_change(lookback)
        return mom
        
    elif method == 'Risk-Adjusted (稳健)':
        # 风险调整动量: 收益率 / 波动率
        # R = (Pt / Pt-n) - 1
        # Vol = std(daily_returns) * sqrt(n)
        
        ret_lookback = df.pct_change(lookback)
        vol_lookback = returns.rolling(window=lookback).std() * np.sqrt(lookback)
        
        # 避免除以0
        adj_mom = ret_lookback / (vol_lookback + 1e-6)
        return adj_mom
        
    elif method == 'Slope (线性回归)' and HAS_SCIPY:
        # 使用线性回归斜率作为动量
        # 这比较慢，需要对每一列滚动计算
        def calc_slope(y):
            x = np.arange(len(y))
            slope, _, r_value, _, _ = stats.linregress(x, y)
            # R^2 * Slope 也是一种变体，这里简单用 Slope
            return slope * (r_value ** 2) # 加入R方惩罚，偏好平稳上涨
            
        # Log prices for slope calculation usually better
        log_prices = np.log(df)
        mom = log_prices.rolling(window=lookback).apply(calc_slope, raw=True)
        return mom
        
    else:
        # Default or fallback
        prices_smoothed = df.rolling(window=smooth).mean()
        mom = prices_smoothed.pct_change(lookback)
        return mom

def run_backtest(df_prices, params):
    """运行回测逻辑"""
    lookback = params['lookback']
    smooth = params['smooth']
    threshold = params['threshold']
    min_holding = params['min_holding']
    allow_cash = params['allow_cash']
    mom_method = params['mom_method']
    
    # 1. 计算动量
    df_mom = calculate_momentum_score(df_prices, mom_method, lookback, smooth)
    
    # 2. 生成信号
    # 初始化
    holdings = pd.DataFrame(index=df_prices.index, columns=['Position', 'Days_Held', 'Value'])
    holdings['Position'] = 'Cash'
    holdings['Days_Held'] = 0
    
    current_pos = 'Cash'
    days_held = 0
    
    # 转换为 Numpy 数组加速循环
    dates = df_prices.index
    mom_values = df_mom.values
    price_values = df_prices.values
    cols = df_prices.columns.tolist()
    
    pos_history = []
    days_history = []
    
    # 从足够数据产生动量的那一天开始
    start_idx = lookback + smooth
    
    # 初始空仓
    pos_history.extend(['Cash'] * start_idx)
    days_history.extend([0] * start_idx)
    
    for i in range(start_idx, len(dates)):
        today_date = dates[i]
        today_moms = mom_values[i] # 当日收盘计算出的动量
        
        # 决策逻辑 (基于今日收盘动量决定明日持仓)
        # 找到动量最大的
        valid_moms = [(cols[k], today_moms[k]) for k in range(len(cols)) if not np.isnan(today_moms[k])]
        
        if not valid_moms:
            target_pos = 'Cash'
        else:
            best_asset, best_mom = max(valid_moms, key=lambda x: x[1])
            
            # 门槛判断
            if best_mom > threshold:
                target_pos = best_asset
            else:
                target_pos = 'Cash' if allow_cash else best_asset
        
        # 最小持仓期限制
        if current_pos != 'Cash' and days_held < min_holding:
            # 强制保持，除非该资产数据丢失
            target_pos = current_pos 
            
        # 更新状态
        if target_pos != current_pos:
            days_held = 1
            current_pos = target_pos
        else:
            days_held += 1
            
        pos_history.append(current_pos)
        days_history.append(days_held)
        
    # 整理结果
    df_res = pd.DataFrame(index=dates, data={
        '当前持仓': pos_history,
        '持仓天数': days_history
    })
    
    # 计算资金曲线
    # 策略收益：如果 T 日持仓 X，则 T+1 日享受 X 的涨跌幅
    df_res['策略涨跌幅'] = 0.0
    
    # 当前持仓列 shift(1) 代表 "昨天确定的持仓，今天享受收益"
    # 但我们的逻辑是: pos_history[i] 是基于 i 日数据计算出并在 i+1 日执行的？
    # 通常回测：Signal computed at T close, Executed at T+1 Open/Close.
    # 简化处理：按 T+1 日收盘价计算收益
    
    # 实际持仓序列 (T日的持仓)
    actual_pos = df_res['当前持仓'].shift(1).fillna('Cash')
    
    daily_rets = df_prices.pct_change().fillna(0)
    
    strategy_rets = []
    for date, pos in zip(daily_rets.index, actual_pos):
        if pos == 'Cash':
            strategy_rets.append(0.0)
        elif pos in daily_rets.columns:
            strategy_rets.append(daily_rets.loc[date, pos])
        else:
            strategy_rets.append(0.0)
            
    df_res['策略涨跌幅'] = strategy_rets
    df_res['总资产'] = (1 + df_res['策略涨跌幅']).cumprod()
    
    # 记录每个资产的表现用于对比
    df_res = pd.concat([df_res, df_prices], axis=1)
    
    return df_res, df_mom

def get_transaction_table(df_res):
    """生成交易明细表"""
    # 找出持仓变化的行
    df_res['昨日持仓'] = df_res['当前持仓'].shift(1)
    # 变化点：今日 != 昨日 (且不是第一天)
    trades = df_res[df_res['当前持仓'] != df_res['昨日持仓']].dropna()
    
    trade_records = []
    last_nav = 1.0
    
    # 我们需要计算每一段的收益
    # 遍历整个 DataFrame 寻找段落
    current_asset = 'Cash'
    start_date = df_res.index[0]
    start_nav = 1.0
    
    # 简单的分段逻辑
    temp_df = df_res.copy()
    temp_df['group'] = (temp_df['当前持仓'] != temp_df['当前持仓'].shift()).cumsum()
    
    grouped = temp_df.groupby('group')
    
    details = []
    for g, frame in grouped:
        asset = frame['当前持仓'].iloc[0]
        start_d = frame.index[0]
        end_d = frame.index[-1]
        days = len(frame)
        
        # 该段结束时的累计净值 / 该段开始前一天的累计净值
        # 简化：用段内涨幅 = (1+r).prod() - 1
        period_ret = (1 + frame['策略涨跌幅']).prod() - 1
        
        # 上一段的结束净值
        end_nav = frame['总资产'].iloc[-1]
        
        action = "继续持有"
        if asset != current_asset:
            if asset == 'Cash':
                action = f"卖出 {get_etf_display_name(current_asset)} -> 空仓"
            elif current_asset == 'Cash':
                action = f"买入 {get_etf_display_name(asset)}"
            else:
                action = f"换仓 {get_etf_display_name(current_asset)} -> {get_etf_display_name(asset)}"
        
        current_asset = asset
        
        # 如果是最后一段（也就是当前状态），修正日期显示
        date_str = start_d.strftime('%Y-%m-%d')
        if start_d == end_d:
            range_str = date_str
        else:
            range_str = f"{date_str} ~ {end_d.strftime('%Y-%m-%d')}"

        # 提取这段时间内各资产的累计表现(仅参考)
        # 这里比较复杂，暂时只存策略表现
        
        details.append({
            "日期": start_d, # 用于排序
            "操作": action,
            "当前持仓": get_etf_display_name(asset) if asset != 'Cash' else '空仓 (Cash)',
            "持仓天数": days,
            "段内收益": period_ret,
            "总资产": end_nav
        })
        
    return pd.DataFrame(details).sort_values('日期', ascending=False)

# ==========================================
# 3. Streamlit 页面布局
# ==========================================
st.set_page_config(page_title="ETF 动量轮动策略 Pro", layout="wide", page_icon="📈")

# 侧边栏配置
st.sidebar.header("⚙️ 策略参数设置")

# 加载配置
current_config = load_config()

# --- 参数控件 ---
lookback = st.sidebar.number_input("动量回看窗口 (天)", min_value=5, max_value=250, value=current_config['lookback'])
smooth = st.sidebar.number_input("平滑窗口 (天)", min_value=1, max_value=60, value=current_config['smooth'])
threshold = st.sidebar.number_input("动量阈值 (Threshold)", min_value=-0.1, max_value=0.1, value=current_config['threshold'], step=0.001, format="%.3f")
min_holding = st.sidebar.number_input("最小持仓天数", min_value=1, max_value=20, value=current_config['min_holding'])
allow_cash = st.sidebar.checkbox("允许空仓 (低于阈值时)", value=current_config['allow_cash'])

mom_methods_list = ['Risk-Adjusted (稳健)', 'Rate of Change (普通)']
if HAS_SCIPY:
    mom_methods_list.append('Slope (线性回归)')

mom_method = st.sidebar.selectbox("动量计算方法", mom_methods_list, index=0 if current_config['mom_method'] not in mom_methods_list else mom_methods_list.index(current_config['mom_method']))

st.sidebar.markdown("---")
st.sidebar.header("🎯 标的池选择")

# --- FEATURE ADDITION: Searchable Select Box ---
# 准备选项列表: "Code - Name"
# 1. 获取所有已知名称的列表
known_codes = list(ETF_NAME_MAP.keys())
# 2. 确保之前保存的 code 也在列表中 (即使没有名字)
all_relevant_codes = list(set(known_codes + current_config['selected_codes']))
all_relevant_codes.sort()

# 创建显示用的 Label 列表
options_map = {code: get_etf_display_name(code) for code in all_relevant_codes}
options_labels = [options_map[code] for code in all_relevant_codes]

# 找出当前选中的 codes 对应的 labels
default_labels = [options_map[c] for c in current_config['selected_codes'] if c in options_map]

# 使用 Multiselect，Streamlit 原生支持在选项中搜索
selected_labels = st.sidebar.multiselect(
    "选择或搜索 ETF (输入代码或名称)",
    options=options_labels,
    default=default_labels,
    help="在下拉框中输入数字代码或中文名称即可搜索"
)

# 将选中的 labels 转换回 codes
selected_codes = [label.split(" - ")[0] for label in selected_labels]

# 保存配置按钮
if st.sidebar.button("💾 保存当前配置"):
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
    st.sidebar.success("配置已保存!")

# 主界面
st.title("📈 量化动量轮动策略 (ETF Momentum)")
st.caption("基于动量因子在 ETF 标的池中进行轮动，支持风险调整动量与最小持仓限制。")

if st.button("🚀 开始回测 / 更新数据", type="primary"):
    if not selected_codes:
        st.error("请至少选择一个标的!")
    else:
        with st.spinner("正在获取数据并计算..."):
            df_prices, valid_codes = get_all_data(selected_codes)
            
            if df_prices.empty:
                st.error("未能获取到任何数据，请检查网络或代码是否正确。")
            else:
                # 运行回测
                params = {
                    'lookback': lookback,
                    'smooth': smooth,
                    'threshold': threshold,
                    'min_holding': min_holding,
                    'allow_cash': allow_cash,
                    'mom_method': mom_method
                }
                df_result, df_mom = run_backtest(df_prices, params)
                
                # ==========================================
                # FEATURE ADDITION: 下一个交易日操作建议
                # ==========================================
                # 逻辑：
                # 1. 获取最新一天的日期
                last_date = df_result.index[-1]
                # 2. 获取该日的回测结果状态（基于当天收盘价计算出的持仓）
                #    注意：df_result['当前持仓'] 第i行代表的是基于i日数据计算出的、建议在i+1日持有的仓位
                last_row = df_result.iloc[-1]
                current_signal = last_row['当前持仓'] # 这是模型建议你在"明天"持有的
                
                # 为了对比，我们需要知道"今天"（也就是倒数第二天建议持有的）是什么
                if len(df_result) > 1:
                    prev_signal = df_result.iloc[-2]['当前持仓']
                else:
                    prev_signal = 'Cash'
                
                # 动量排名前三（用于参考）
                last_mom_row = df_mom.iloc[-1]
                top_assets = last_mom_row.sort_values(ascending=False).head(3)
                
                st.markdown("### 🔔 下一个交易日操作建议")
                
                # 构建建议卡片
                col_alert, col_info = st.columns([2, 3])
                
                with col_alert:
                    # 判断操作类型
                    if current_signal == prev_signal:
                        if current_signal == 'Cash':
                            st.warning(f"🛑 **继续空仓**\n\n当前无标的满足动量阈值，建议持有现金。")
                        else:
                            st.info(f"✊ **继续持有**\n\n建议继续持有 **{get_etf_display_name(current_signal)}**")
                    else:
                        if prev_signal == 'Cash':
                            st.success(f"🟢 **买入信号**\n\n建议买入 **{get_etf_display_name(current_signal)}**")
                        elif current_signal == 'Cash':
                            st.error(f"🔴 **卖出信号**\n\n建议卖出 **{get_etf_display_name(prev_signal)}**，转为空仓。")
                        else:
                            st.warning(f"🔄 **换仓信号**\n\n卖出 **{get_etf_display_name(prev_signal)}**\n\n买入 **{get_etf_display_name(current_signal)}**")
                
                with col_info:
                    st.markdown(f"**数据截止日期**: {last_date.strftime('%Y-%m-%d')}")
                    st.markdown("**当前动量排名 Top 3:**")
                    for asset, score in top_assets.items():
                        asset_name = ETF_NAME_MAP.get(asset, asset)
                        st.text(f"{asset_name}: {score:.4f}")
                    
                    if current_signal == 'Cash' and not top_assets.empty:
                         st.caption(f"注：排名第一的 {get_etf_display_name(top_assets.index[0])} 动量值为 {top_assets[0]:.4f}，低于设定阈值 {threshold}，故不持有。")

                st.divider()

                # ==========================================
                # 原有图表展示逻辑
                # ==========================================
                
                # 1. 资金曲线图
                st.subheader("📊 策略表现 (净值曲线)")
                
                # 计算基准收益 (等权)
                df_prices_norm = df_prices / df_prices.iloc[0]
                df_result['平均基准'] = df_prices_norm.mean(axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_result.index, y=df_result['总资产'], mode='lines', name='策略净值', line=dict(width=2, color='red')))
                fig.add_trace(go.Scatter(x=df_result.index, y=df_result['平均基准'], mode='lines', name='标的平均表现', line=dict(width=1, color='grey', dash='dash')))
                
                # 标记买卖点
                # 找出持仓切换的点
                changes = df_result[df_result['当前持仓'] != df_result['当前持仓'].shift(1)]
                # 排除初始点
                if not changes.empty:
                    # 只要不是 Cash 的点，都标记一下
                    changes_buy = changes[changes['当前持仓'] != 'Cash']
                    fig.add_trace(go.Scatter(
                        x=changes_buy.index, 
                        y=changes_buy['总资产'],
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='blue'),
                        name='调仓/买入',
                        text=[f"买入: {get_etf_display_name(c)}" for c in changes_buy['当前持仓']],
                        hoverinfo='text+x+y'
                    ))
                
                fig.update_layout(xaxis_title="日期", yaxis_title="净值", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. 详细数据表格
                st.subheader("📝 交易与持仓详情")
                
                df_details = get_transaction_table(df_result)
                if not df_details.empty:
                    # 格式化显示
                    # 将百分比转为更好看的格式
                    df_details['段内收益'] = df_details['段内收益'].map(lambda x: f"{x*100:.2f}%")
                    df_details['总资产'] = df_details['总资产'].map(lambda x: f"{x:.4f}")
                    df_details['日期'] = df_details['日期'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(
                        df_details,
                        column_config={
                            "日期": "开始日期",
                            "操作": "调仓动作",
                            "当前持仓": "持有标的",
                            "持仓天数": st.column_config.NumberColumn("持仓天数"),
                            "段内收益": "本段收益",
                            "总资产": "期末净值"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("暂无交易记录")

else:
    st.info("👈 请在左侧调整参数并点击 '开始回测' 按钮")
