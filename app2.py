import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="核心资产轮动策略看板",
    page_icon="📈",
    layout="wide"
)

# 标的池配置 (固定不变)
ASSETS = {
    '510180': {'name': '上证180 (价值)', 'color': '#1f77b4'},
    '159915': {'name': '创业板指 (成长)', 'color': '#2ca02c'},
    '513100': {'name': '纳指100 (海外)', 'color': '#9467bd'},
    '518880': {'name': '黄金ETF (避险)', 'color': '#ff7f0e'}
}

# ==========================================
# 2. 数据获取与缓存
# ==========================================
@st.cache_data(ttl=3600*12)
def load_data():
    """下载全量数据"""
    price_dict = {}
    # 下载足够早的数据以确保2014年初始动量可计算
    start_str = '20130101'
    end_str = datetime.now().strftime('%Y%m%d')
    
    # 进度提示
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    idx = 0
    for code, info in ASSETS.items():
        name = info['name']
        status_text.text(f"正在下载: {name}...")
        try:
            # 使用前复权 (qfq) 保证收益率真实性
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[name] = df['收盘'].astype(float)
        except Exception as e:
            st.error(f"{name} 下载失败: {e}")
        
        idx += 1
        progress_bar.progress(idx / len(ASSETS))
    
    status_text.text("数据清洗中...")
    # 对齐数据，前向填充处理停牌
    data = pd.concat(price_dict, axis=1).sort_index().ffill().dropna()
    
    progress_bar.empty()
    status_text.empty()
    
    return data

def calculate_slope(series):
    """辅助函数：计算线性回归斜率 (简化版，用于rolling apply)"""
    # x是时间序列 0, 1, 2... n
    y = np.log(series) # 使用对数价格，计算出的斜率近似于指数增长率
    n = len(y)
    x = np.arange(n)
    # 线性回归斜率公式: (n*Sum(xy) - Sum(x)*Sum(y)) / (n*Sum(x^2) - (Sum(x))^2)
    # 为了速度，直接使用 numpy 的 polyfit
    try:
        slope, _ = np.polyfit(x, y, 1)
        return slope
    except:
        return 0.0

def calculate_indicators(data, lookback, smooth_window, method):
    """
    根据参数动态计算指标
    :param method: "普通动量 (ROC)", "夏普动量 (Sharpe)", "回归动量 (Slope)"
    """
    # 1. 每日收益率
    daily_returns = data.pct_change().fillna(0)
    
    raw_mom = pd.DataFrame()

    # --- 核心动量算法分支 ---
    if method == "普通动量 (ROC)":
        # 经典算法: Pt / Pt-n - 1
        raw_mom = data.pct_change(lookback)
        
    elif method == "夏普动量 (Sharpe)":
        # 科学算法1: 风险调整后收益
        # 计算窗口期内的平均日收益率 / 收益率标准差
        # 乘以 sqrt(252) 年化，虽然比较时可以约掉，但保留年化习惯更好
        window_mean = daily_returns.rolling(lookback).mean()
        window_std = daily_returns.rolling(lookback).std()
        # 避免除以0
        raw_mom = (window_mean / (window_std + 1e-9)) * np.sqrt(252)
        
    elif method == "回归动量 (Slope)":
        # 科学算法2: 线性回归斜率 (抗噪音能力最强)
        # 计算 log(price) 对 time 的回归斜率
        # rolling apply 速度稍慢，但对于几千行数据是可以接受的
        raw_mom = data.rolling(lookback).apply(calculate_slope, raw=True)

    # 3. 动量平滑 (如果 smooth_window=1 则相当于不平滑)
    if smooth_window > 1:
        signal_mom = raw_mom.rolling(smooth_window).mean()
    else:
        signal_mom = raw_mom
        
    # 4. 信号偏移: T日的持仓只能基于T-1日的收盘数据
    signal_mom_shifted = signal_mom.shift(1)
    
    return daily_returns, signal_mom_shifted

# ==========================================
# 3. 回测引擎
# ==========================================
def run_backtest(start_date, end_date, initial_capital, daily_returns, signal_mom, threshold):
    # 截取时间段
    mask = (daily_returns.index >= pd.to_datetime(start_date)) & (daily_returns.index <= pd.to_datetime(end_date))
    period_ret = daily_returns.loc[mask]
    period_mom = signal_mom.loc[mask]
    
    if period_ret.empty:
        return None, 0

    dates = period_ret.index
    capital = initial_capital
    curve = []
    holdings = []
    mom_scores = [] 
    
    current_holding = None
    trade_count = 0
    
    for date in dates:
        row = period_mom.loc[date]
        
        # 选出最高分
        best_asset = row.idxmax()
        best_score = row.max()
        
        target = current_holding
        
        # 决策逻辑
        if pd.isna(best_asset) or pd.isna(best_score):
            pass 
        else:
            if current_holding is None:
                target = best_asset
            elif current_holding not in row.index:
                target = best_asset
            else:
                curr_score = row[current_holding]
                if best_asset != current_holding:
                    # 阈值判定
                    if best_score > curr_score + threshold:
                        target = best_asset
                    else:
                        target = current_holding
        
        if target != current_holding and target is not None:
            trade_count += 1
            
        current_holding = target
        
        if current_holding:
            r = period_ret.loc[date, current_holding]
            capital = capital * (1 + r)
            holdings.append(current_holding)
            mom_scores.append(row[current_holding])
        else:
            holdings.append('准备期')
            mom_scores.append(0)
            
        curve.append(capital)
        
    res_df = pd.DataFrame({
        '总资产': curve,
        '持仓': holdings,
        '持仓动量分': mom_scores
    }, index=dates)
    
    mom_display = period_mom.copy()
    mom_display.columns = [f"{c}_分" for c in mom_display.columns]
    res_df = pd.concat([res_df, mom_display], axis=1)
    
    return res_df, trade_count

# ==========================================
# 4. 主界面逻辑
# ==========================================
def main():
    with st.sidebar:
        st.header("⚙️ 策略控制台")
        
        # 1. 动量模型选择 (本次更新核心)
        mom_method = st.selectbox(
            "动量计算模型 (Algorithm)",
            ["普通动量 (ROC)", "夏普动量 (Sharpe)", "回归动量 (Slope)"],
            index=0,
            help="""
            - 普通动量: 简单计算 (P_t / P_t-n) - 1。对噪音敏感。
            - 夏普动量: 收益率 / 波动率。优先选择涨得稳的标的 (风险调整)。
            - 回归动量: 计算价格走势的线性斜率。利用了期间所有数据，抗干扰最强。
            """
        )

        st.divider()
        
        # 2. 模式与参数
        mode = st.radio(
            "回测模式",
            ("PPT严格复刻", "自定义稳健"),
            index=0
        )
        
        if mode == "PPT严格复刻":
            lookback = 25
            smooth = 1
            threshold = 0.0
            st.caption("🔒 参数已锁定: 25日周期 / 无平滑 / 无阈值")
        else:
            lookback = st.number_input("动量周期 (日)", value=25)
            smooth = st.number_input("平滑窗口 (日)", value=3)
            threshold = st.number_input("换仓阈值", value=0.005, step=0.001, format="%.3f")
        
        st.divider()
        init_cash = st.number_input("初始本金", value=500000, step=10000)
        
        # 日期选择
        data = load_data()
        min_date = data.index[0].date()
        max_date = data.index[-1].date()
        default_start = datetime(2014, 1, 1).date()
        
        col1, col2 = st.columns(2)
        start_date = col1.date_input("开始", value=default_start, min_value=min_date, max_value=max_date)
        end_date = col2.date_input("结束", value=max_date, min_value=min_date, max_value=max_date)

    # --- 主区域 ---
    st.title("📊 核心资产轮动策略看板 (Pro)")
    
    # 动态显示当前算法原理
    with st.expander(f"📖 当前算法详解: {mom_method}", expanded=True):
        if mom_method == "普通动量 (ROC)":
            st.markdown(r"$$ \text{Score} = \frac{P_t}{P_{t-25}} - 1 $$")
            st.info("最原始的算法。优点是反应快，缺点是如果25天前正好是个低点，今天的动量会虚高（基数效应）。")
        elif mom_method == "夏普动量 (Sharpe)":
            st.markdown(r"$$ \text{Score} = \frac{\text{Mean}(R)}{\text{Std}(R)} \times \sqrt{252} $$")
            st.info("最科学的算法。它惩罚波动率。如果纳指和黄金都涨了10%，但黄金走势更平稳，系统会认为黄金的动量更强。适合追求稳健收益。")
        elif mom_method == "回归动量 (Slope)":
            st.markdown(r"$$ \ln(P_t) = \alpha + \beta \cdot t + \epsilon \quad (\text{Score} = \beta) $$")
            st.info("最稳健的算法。它对过去25天的价格取对数后拟合一条直线，直线的斜率代表平均增长速度。它使用了期间所有数据点，极难被单日暴涨暴跌干扰。")

    # 计算指标
    daily_returns, signal_mom = calculate_indicators(data, lookback, smooth, mom_method)
    
    # 运行回测
    df_res, trade_count = run_backtest(start_date, end_date, init_cash, daily_returns, signal_mom, threshold)
    
    if df_res is None:
        st.error("无数据")
        st.stop()
        
    # --- 结果展示 ---
    final_val = df_res['总资产'].iloc[-1]
    total_ret = (final_val / init_cash) - 1
    days = (df_res.index[-1] - df_res.index[0]).days
    annual_ret = (final_val / init_cash) ** (365.25/days) - 1 if days > 0 else 0
    avg_days = days / trade_count if trade_count > 0 else days

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("区间收益率", f"{total_ret*100:.2f}%", f"期末: {final_val:,.0f}")
    c2.metric("年化收益率", f"{annual_ret*100:.2f}%")
    c3.metric("调仓次数", f"{trade_count} 次", f"平均 {avg_days:.1f} 天/换")
    
    # 图表
    st.subheader("📈 资金曲线")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.85, 0.15])
    
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['总资产'], mode='lines', name='策略净值', line=dict(color='#d62728', width=2)), row=1, col=1)
    
    for code, info in ASSETS.items():
        name = info['name']
        bench = (1 + daily_returns.loc[df_res.index, name]).cumprod()
        bench = bench / bench.iloc[0] * init_cash
        fig.add_trace(go.Scatter(x=df_res.index, y=bench, name=name, line=dict(width=1, dash='dot'), opacity=0.3), row=1, col=1)

    # 色带
    df_res['group'] = (df_res['持仓'] != df_res['持仓'].shift()).cumsum()
    groups = df_res.reset_index().groupby('group').agg({'日期': ['first', 'last'], '持仓': 'first'})
    groups.columns = ['start', 'end', 'asset']
    
    for _, row in groups.iterrows():
        asset = row['asset']
        color = 'gray'
        for _, info in ASSETS.items():
            if info['name'] == asset: color = info['color']
        
        fig.add_trace(go.Scatter(x=[row['start'], row['end']], y=[1, 1], mode='lines', line=dict(color=color, width=15), name=asset, showlegend=False, hovertemplate=f"持仓: {asset}<extra></extra>"), row=2, col=1)

    fig.update_layout(height=500, hovermode="x unified", yaxis=dict(title='总资产'), yaxis2=dict(showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)
    
    # 详细数据
    with st.expander("📋 每日详细数据 (含动量分)"):
        st.dataframe(df_res.sort_index(ascending=False).style.format({'总资产': '{:,.2f}'}), use_container_width=True)

if __name__ == "__main__":
    main()