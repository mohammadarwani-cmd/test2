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
# 0. 配置持久化管理 (Config Persistence)
# ==========================================
CONFIG_FILE = 'strategy_config.json'

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
            return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    """保存配置到本地文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass

# ==========================================
# 1. 投行级页面配置 & CSS样式 (UI优化版)
# ==========================================
st.set_page_config(
    page_title="AlphaTarget | 核心资产轮动策略终端",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局背景与字体优化 */
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    /* 侧边栏优化 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }

    /* 指标卡片 (Metric Card) */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #eaeaea;
        border-radius: 12px;
        padding: 20px 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #d0d0d0;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #2c3e50;
        font-size: 1.6rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #95a5a6;
        margin-top: 6px;
    }

    /* 信号横幅 (Signal Banner) */
    .signal-banner {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* 建议横幅 - 继续持有 */
    .advice-banner-hold {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* 建议横幅 - 调仓 */
    .advice-banner-switch {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* 建议横幅 - 空仓 */
    .advice-banner-cash {
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
        background: linear-gradient(135deg, #7f8c8d 0%, #95a5a6 100%);
        box-shadow: 0 4px 15px rgba(127, 140, 141, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* 表格样式优化 */
    .dataframe {
        font-size: 13px !important;
        border: 1px solid #eee;
    }
    
    /* 总资产大标题 */
    .total-asset-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0.2rem;
        font-family: 'Arial', sans-serif;
    }
    .total-asset-sub {
        font-size: 1.1rem;
        color: #7f8c8d;
        font-weight: 500;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* 优化器结果卡片高亮 */
    .opt-highlight {
        background-color: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

TRANSACTION_COST = 0.0001  # 万分之一

PRESET_ETFS = {
    "518880": "黄金ETF (避险)", "588000": "科创50 (硬科技)", "513100": "纳指100 (海外)",
    "510180": "上证180 (蓝筹)", "159915": "创业板指 (成长)", "510300": "沪深300 (大盘)",
    "510500": "中证500 (中盘)", "512890": "红利低波 (防御)", "513500": "标普500 (美股)",
    "512480": "半导体ETF (行业)", "512880": "证券ETF (Beta)"
}

# 辅助函数：根据名称生成柔和的颜色
def get_color_from_name(name):
    if name == 'Cash':
        return 'rgba(200, 200, 200, 0.2)' 
    hash_obj = hashlib.md5(name.encode())
    hex_dig = hash_obj.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    r = (r + 255) // 2
    g = (g + 255) // 2
    b = (b + 255) // 2
    return f'rgba({r}, {g}, {b}, 0.25)' 

def metric_html(label, value, sub="", color="#2c3e50"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """

# ==========================================
# 2. 数据层 (Data Layer) - 纯净历史数据版
# ==========================================

@st.cache_data(ttl=3600*12) 
def get_all_etf_list():
    try:
        df = ak.fund_etf_spot_em()
        df['display'] = df['代码'] + " | " + df['名称']
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_market_data(codes_list, end_date_str):
    """
    纯净历史数据下载，不进行实时融合
    返回：收盘价数据、开盘价数据、名称映射
    """
    start_str = '20150101' 
    close_dict = {}
    open_dict = {}
    name_map = {}
    
    etf_list = get_all_etf_list()
    
    for code in codes_list:
        name = code
        if code in PRESET_ETFS:
            name = PRESET_ETFS[code].split(" ")[0]
        elif not etf_list.empty:
            match = etf_list[etf_list['代码'] == code]
            if not match.empty:
                name = match.iloc[0]['名称']
        name_map[code] = name
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                close_dict[name] = df['收盘'].astype(float)
                open_dict[name] = df['开盘'].astype(float)
        except Exception:
            continue

    if not close_dict:
        return None, None, None

    close_data = pd.concat(close_dict, axis=1).sort_index().ffill()
    open_data = pd.concat(open_dict, axis=1).sort_index().ffill()
    close_data.dropna(how='all', inplace=True)
    open_data.dropna(how='all', inplace=True)
    if len(close_data) < 20: return None, None, None
    return close_data, open_data, name_map

# ==========================================
# 3. 策略内核 (Strategy Core)
# ==========================================

def calculate_momentum(data, lookback, smooth, method='Classic (普通)'):
    if method == 'Classic (普通)':
        mom = data.pct_change(lookback)
    elif method == 'Risk-Adjusted (稳健)':
        ret = data.pct_change(lookback)
        vol = data.pct_change().rolling(lookback).std()
        mom = ret / (vol + 1e-9)
    elif method == 'MA Distance (趋势)':
        ma = data.rolling(lookback).mean()
        mom = (data / ma) - 1
    else:
        mom = data.pct_change(lookback)

    if smooth > 1:
        mom = mom.rolling(smooth).mean()
        
    return mom

def fast_backtest_vectorized(daily_ret, mom_df, threshold, min_holding=1, cost_rate=0.0001, allow_cash=True):
    signal_mom = mom_df.shift(1)
    n_days, n_assets = daily_ret.shape
    p_ret = daily_ret.values
    p_mom = signal_mom.values
    
    strategy_ret = np.zeros(n_days)
    curr_idx = -2 
    trade_count = 0
    days_held = 0 
    
    for i in range(n_days):
        if curr_idx != -2:
            days_held += 1
            
        row_mom = p_mom[i]
        if np.isnan(row_mom).all(): continue
            
        clean_mom = np.nan_to_num(row_mom, nan=-np.inf)
        best_idx = np.argmax(clean_mom)
        best_val = clean_mom[best_idx]
        target_idx = curr_idx
        
        if allow_cash and best_val < 0:
            target_idx = -1
        else:
            if curr_idx == -2:
                if best_val > -np.inf: target_idx = best_idx
            elif curr_idx == -1:
                if best_val > 0 or (not allow_cash): target_idx = best_idx
            else:
                is_stop_loss = (target_idx == -1) 
                if not is_stop_loss:
                    if days_held >= min_holding:
                        curr_val = clean_mom[curr_idx]
                        if best_idx != curr_idx:
                            if best_val > curr_val + threshold:
                                target_idx = best_idx
                    else:
                        target_idx = curr_idx
        
        if target_idx != curr_idx:
            if curr_idx != -2:
                strategy_ret[i] -= cost_rate
                trade_count += 1
                days_held = 0
            curr_idx = target_idx
            
        if curr_idx >= 0:
            strategy_ret[i] += p_ret[i, curr_idx]
            
    equity_curve = (1 + strategy_ret).cumprod()
    total_ret = equity_curve[-1] - 1
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    return total_ret, max_dd, equity_curve, trade_count

# ==========================================
# 3.5 下一个交易日建议功能 (新增)
# ==========================================

def generate_next_day_advice(close_data, open_data, last_hold, mom_method, lookback, smooth, threshold, allow_cash, name_map):
    """
    生成下一个交易日的投资建议
    
    参数:
        close_data: 收盘价数据
        open_data: 开盘价数据
        last_hold: 当前持仓标的
        mom_method: 动量计算方法
        lookback: 动量周期
        smooth: 平滑窗口
        threshold: 换仓阈值
        allow_cash: 是否允许空仓
        name_map: 代码到名称的映射
    
    返回:
        advice_type: 'hold', 'switch', 'cash'
        advice_text: 建议文本
        target_asset: 目标标的
        current_asset: 当前持仓标的
        mom_ranking: 动量排名
    """
    # 计算最新动量
    mom = calculate_momentum(close_data, lookback, smooth, mom_method)
    
    # 获取最新动量值
    latest_mom = mom.iloc[-1].dropna()
    latest_mom_sorted = latest_mom.sort_values(ascending=False)
    
    # 获取最佳标的
    best_asset = latest_mom_sorted.index[0]
    best_score = latest_mom_sorted.iloc[0]
    
    # 确定当前持仓的动量值
    current_score = latest_mom.get(last_hold, -np.inf) if last_hold and last_hold != 'Cash' else -np.inf
    
    # 判断是否调仓（不考虑最小持仓天数）
    target = last_hold
    
    if allow_cash and best_score < 0:
        # 所有标的动量为负，建议空仓
        target = 'Cash'
    else:
        if last_hold is None or last_hold == 'Cash':
            # 当前空仓，建议买入最佳标的
            target = best_asset
        else:
            # 当前有持仓，判断是否调仓
            if best_asset != last_hold:
                if best_score > current_score + threshold:
                    target = best_asset
    
    # 生成建议文本
    if target == last_hold:
        if target == 'Cash':
            advice_type = 'cash'
            advice_text = "下一个交易日：继续空仓避险"
        else:
            advice_type = 'hold'
            target_name = name_map.get(target, target)
            advice_text = f"下一个交易日：继续持有 {target_name}"
    else:
        advice_type = 'switch'
        current_name = name_map.get(last_hold, last_hold) if last_hold and last_hold != 'Cash' else "空仓"
        target_name = name_map.get(target, target) if target != 'Cash' else "空仓避险"
        advice_text = f"下一个交易日：调仓至 {target_name}"
    
    return {
        'advice_type': advice_type,
        'advice_text': advice_text,
        'target_asset': target,
        'current_asset': last_hold,
        'mom_ranking': latest_mom_sorted,
        'best_score': best_score,
        'current_score': current_score
    }

# ==========================================
# 4. 分析师工具箱
# ==========================================

def calculate_pro_metrics(equity_curve, benchmark_curve, trade_count):
    if len(equity_curve) < 2: return {}
    s_eq = pd.Series(equity_curve)
    daily_ret = s_eq.pct_change().fillna(0)
    days = len(equity_curve)
    
    total_ret = equity_curve[-1] - 1
    ann_ret = (1 + total_ret) ** (252 / days) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    rf = 0.03
    sharpe = (ann_ret - rf) / (ann_vol + 1e-9)
    
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) + 1e-9)
    
    beta, alpha = 0.0, 0.0
    if HAS_SCIPY and len(benchmark_curve) == len(equity_curve):
        s_bm = pd.Series(benchmark_curve)
        bm_ret = s_bm.pct_change().fillna(0)
        try:
            slope, intercept, _, _, _ = stats.linregress(bm_ret.values[1:], daily_ret.values[1:])
            beta = slope
            alpha = intercept * 252
        except: pass
            
    return {
        "Total Return": total_ret, "CAGR": ann_ret, "Volatility": ann_vol,
        "Max Drawdown": max_dd, "Sharpe Ratio": sharpe, "Calmar Ratio": calmar,
        "Alpha": alpha, "Beta": beta, "Trades": trade_count
    }

def optimize_parameters(data, allow_cash, min_holding):
    methods = ['Classic (普通)', 'Risk-Adjusted (稳健)', 'MA Distance (趋势)']
    lookbacks = range(20, 31, 1) 
    smooths = range(1, 8, 1)     
    thresholds = np.arange(0.0, 0.013, 0.001)
    
    daily_ret = data.pct_change().fillna(0)
    n_days = len(daily_ret) 
    results = []
    
    total_iters = len(methods) * len(lookbacks) * len(smooths) * len(thresholds)
    my_bar = st.progress(0, text="正在进行四维全景扫描 (Method/Loop/Smooth/Th)...")
    
    idx = 0
    for method in methods:
        for lb in lookbacks:
            for sm in smooths:
                mom = calculate_momentum(data, lb, sm, method)
                for th in thresholds:
                    ret, dd, equity, count = fast_backtest_vectorized(
                        daily_ret, mom, th, 
                        min_holding=min_holding,
                        cost_rate=TRANSACTION_COST, 
                        allow_cash=allow_cash
                    )
                    
                    ann_ret = (1 + ret) ** (252 / n_days) - 1
                    if n_days > 1:
                        eq_s = pd.Series(equity)
                        d_r = eq_s.pct_change().fillna(0)
                        ann_vol = d_r.std() * np.sqrt(252)
                        sharpe = (ann_ret - 0.03) / (ann_vol + 1e-9)
                    else:
                        sharpe = 0.0
                    
                    ann_trades = count * (252 / n_days)
                    score = ret / (abs(dd) + 0.05)
                    
                    results.append([method, lb, sm, th, ret, ann_ret, count, ann_trades, dd, sharpe, score])
                    
                    idx += 1
                    if idx % 200 == 0:
                        my_bar.progress(min(idx / total_iters, 1.0))
                    
    my_bar.empty()
    df_res = pd.DataFrame(results, columns=['方法', '周期', '平滑', '阈值', '累计收益', '年化收益', '调仓次数', '年化调仓', '最大回撤', '夏普比率', '得分'])
    return df_res

# ==========================================
# 5. 主程序 UI
# ==========================================

def main():
    if 'params' not in st.session_state:
        saved_config = load_config()
        st.session_state.params = saved_config

    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = None

    with st.sidebar:
        st.title("🎛️ 策略控制台")
        
        # --- 1. 资产与数据 ---
        st.subheader("1. 资产池配置")
        all_etfs = get_all_etf_list()
        options = all_etfs['display'].tolist() if not all_etfs.empty else DEFAULT_CODES
        current_selection_codes = st.session_state.params.get('selected_codes', DEFAULT_CODES)
        
        default_display = []
        if not all_etfs.empty:
            for code in current_selection_codes:
                match = all_etfs[all_etfs['代码'] == code]
                if not match.empty:
                    default_display.append(match.iloc[0]['display'])
                else:
                    for opt in options:
                        if opt.startswith(code):
                            default_display.append(opt)
                            break
        else:
            default_display = current_selection_codes
            
        valid_defaults = [x for x in default_display if x in options]
        selected_display = st.multiselect("核心标的池", options, default=valid_defaults)
        selected_codes = [x.split(" | ")[0] for x in selected_display]
        
        st.divider()
        st.subheader("2. 资金管理")
        
        date_mode = st.radio("回测区间", ["全历史", "自定义"], index=0)
        
        # [修改] 默认开始时间改为 2021-01-01
        start_date_input = datetime(2021, 1, 1)
        end_date_input = datetime.now()
        
        if date_mode == "自定义":
            c1, c2 = st.columns(2)
            start_date_input = c1.date_input("Start", datetime(2021, 1, 1))
            end_date_input = c2.date_input("End", datetime.now())

        invest_mode = st.radio("投资模式", ["一次性投入 (Lump Sum)", "定期定额 (SIP)"], index=0)
        
        initial_capital = 100000.0
        sip_amount = 0.0
        sip_freq = "None"
        
        if invest_mode == "一次性投入 (Lump Sum)":
            initial_capital = st.number_input("初始本金", value=100000.0, step=10000.0)
        else:
            c1, c2 = st.columns(2)
            initial_capital = c1.number_input("初始底仓", value=10000.0, step=1000.0)
            sip_amount = c2.number_input("定投金额", value=2000.0, step=500.0)
            sip_freq = st.selectbox("定投频率", ["每月 (Monthly)", "每周 (Weekly)"], index=0)

        st.divider()
        
        # --- 3. 策略参数 ---
        with st.form(key='settings_form'):
            st.subheader("3. 策略内核参数")
            
            mom_options = ['Classic (普通)', 'Risk-Adjusted (稳健)', 'MA Distance (趋势)']
            default_mom = st.session_state.params.get('mom_method', 'Risk-Adjusted (稳健)')
            if default_mom not in mom_options: default_mom = 'Classic (普通)'
            
            p_method = st.selectbox("动量计算逻辑", mom_options, index=mom_options.index(default_mom))
            
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                p_lookback = st.number_input("动量周期", min_value=2, max_value=120, value=st.session_state.params.get('lookback', 25), step=1)
            with c_p2:
                p_smooth = st.number_input("平滑窗口", min_value=1, max_value=60, value=st.session_state.params.get('smooth', 3), step=1)
                
            p_threshold = st.number_input("换仓阈值", 0.0, 0.05, st.session_state.params.get('threshold', 0.005), step=0.001, format="%.3f")
            
            st.markdown("---")
            st.markdown("**🛑 风控参数**")
            p_min_holding = st.number_input("最小持仓天数", min_value=1, max_value=60, value=st.session_state.params.get('min_holding', 3), step=1)
            p_allow_cash = st.checkbox("启用绝对动量避险 (Cash Protection)", value=st.session_state.params.get('allow_cash', True))
            
            submit_btn = st.form_submit_button("🚀 确认并运行 (Run Analysis)")

        if submit_btn:
            current_params = {
                'lookback': p_lookback, 'smooth': p_smooth, 'threshold': p_threshold,
                'min_holding': p_min_holding, 'allow_cash': p_allow_cash, 'selected_codes': selected_codes,
                'mom_method': p_method 
            }
            if current_params != st.session_state.params:
                st.session_state.params = current_params
                save_config(current_params)
        
        if st.button("🔄 重置默认配置"):
            st.session_state.params = DEFAULT_PARAMS.copy()
            save_config(DEFAULT_PARAMS)
            st.rerun()

    # 日期逻辑
    start_date = datetime.combine(start_date_input, datetime.min.time()) if isinstance(start_date_input, datetime) == False else start_date_input
    end_date = datetime.combine(end_date_input, datetime.min.time()) if isinstance(end_date_input, datetime) == False else end_date_input
    if not isinstance(start_date, datetime): start_date = datetime.combine(start_date, datetime.min.time())
    if not isinstance(end_date, datetime): end_date = datetime.combine(end_date, datetime.min.time())

    st.markdown("## 🚀 核心资产轮动策略终端 (Pro Ver.)")
    
    if not selected_codes:
        st.warning("请选择标的。")
        st.stop()
        
    with st.spinner("正在加载历史行情数据 (Historical Data Only)..."):
        # [修改] 同时获取收盘价和开盘价数据
        raw_data, open_data, name_map = download_market_data(selected_codes, end_date.strftime('%Y%m%d'))
        
    if raw_data is None:
        st.error("数据不足或下载失败。")
        st.stop()

    daily_ret_all = raw_data.pct_change().fillna(0)
    mom_method_curr = st.session_state.params.get('mom_method', 'Classic (普通)')
    mom_all = calculate_momentum(raw_data, p_lookback, p_smooth, mom_method_curr)
    
    mask = (raw_data.index >= start_date) & (raw_data.index <= end_date)
    sliced_data = raw_data.loc[mask]
    sliced_mom = mom_all.loc[mask] 
    sliced_ret = daily_ret_all.loc[mask]
    
    if sliced_data.empty:
        st.error("区间内无数据")
        st.stop()

    signal_mom = sliced_mom.shift(1)
    dates = sliced_ret.index
    
    # === 回测逻辑（保持不变）===
    cash = initial_capital
    share_val = 0.0
    curr_hold = None
    days_held = 0
    current_hold_start_val = 0.0 
    
    holdings_history = []
    total_assets_curve = []
    total_invested_curve = []
    total_invested = initial_capital
    trade_count_real = 0
    daily_details = [] 
    last_sip_date = dates[0]
    
    for i, date in enumerate(dates):
        r_today = sliced_ret.loc[date]
        
        # A. 定投
        if invest_mode == "定期定额 (SIP)" and i > 0:
            is_sip_day = False
            if sip_freq.startswith("每月"):
                if date.month != last_sip_date.month: is_sip_day = True
            elif sip_freq.startswith("每周"):
                if date.weekday() == 0 and last_sip_date.weekday() != 0: is_sip_day = True
            
            if is_sip_day:
                cash += sip_amount
                total_invested += sip_amount
                last_sip_date = date

        # B. 信号
        if curr_hold is not None: days_held += 1
        row = signal_mom.loc[date]
        target = curr_hold
        
        if not row.isna().all():
            clean_row = row.fillna(-np.inf)
            best_asset = clean_row.idxmax()
            best_score = clean_row.max()
            
            if p_allow_cash and best_score < 0:
                target = 'Cash'
            else:
                if curr_hold is None or curr_hold == 'Cash':
                    target = best_asset
                else:
                    if days_held >= p_min_holding:
                        curr_score = clean_row.get(curr_hold, -np.inf)
                        if best_asset != curr_hold:
                            if best_score > curr_score + p_threshold: target = best_asset
                    else:
                        target = curr_hold

        day_return = 0.0
        if curr_hold and curr_hold != 'Cash' and curr_hold in r_today:
            day_return = r_today[curr_hold]
        
        share_val = share_val * (1 + day_return)
        
        temp_segment_ret = 0.0
        if curr_hold and curr_hold != 'Cash' and current_hold_start_val > 0:
            temp_segment_ret = (share_val / current_hold_start_val) - 1
            
        log_hold = curr_hold
        log_days = days_held
        log_ret = temp_segment_ret
        note = ""

        # 交易执行
        if target != curr_hold:
            if curr_hold is not None:
                total_equity = share_val + cash
                cost = total_equity * TRANSACTION_COST
                if cash >= cost: cash -= cost
                else: share_val -= cost
                trade_count_real += 1
                days_held = 0
                
                old_name = name_map.get(curr_hold, curr_hold) if curr_hold else "Cash"
                new_name = name_map.get(target, target) if target else "Cash"
                note = f"调仓: {old_name} -> {new_name}"
                
            if target == 'Cash':
                cash += share_val
                share_val = 0.0
            else:
                total = share_val + cash
                share_val = total
                cash = 0.0
                current_hold_start_val = total
                
            curr_hold = target
            
        holdings_history.append(target if target else "Cash")
        current_total = share_val + cash
        total_assets_curve.append(current_total)
        total_invested_curve.append(total_invested)
        
        hold_name_display = name_map.get(log_hold, log_hold) if log_hold and log_hold != 'Cash' else 'Cash'
        
        daily_record = {
            "日期": date.strftime('%Y-%m-%d'),
            "当前持仓": hold_name_display,
            "持仓天数": log_days if log_hold != 'Cash' else 0,
            "段内收益": log_ret if log_hold != 'Cash' else 0.0,
            "操作": note,
            "总资产": current_total,
        }
        
        for code, val in r_today.items():
            col_name = name_map.get(code, code)
            daily_record[col_name] = val 
            
        daily_details.append(daily_record)

    df_res = pd.DataFrame({
        '总资产': total_assets_curve,
        '投入本金': total_invested_curve,
        '持仓': holdings_history
    }, index=dates)
    
    _, _, nav_series, _ = fast_backtest_vectorized(
        sliced_ret, sliced_mom, p_threshold, 
        min_holding=p_min_holding, cost_rate=TRANSACTION_COST, allow_cash=p_allow_cash
    )
    df_res['策略净值'] = nav_series
    bm_curve = (1 + sliced_ret.mean(axis=1)).cumprod()
    
    # === 生成下一个交易日建议（新增功能）===
    last_hold = holdings_history[-1]
    next_day_advice = generate_next_day_advice(
        raw_data, open_data, last_hold, 
        mom_method_curr, p_lookback, p_smooth, p_threshold, p_allow_cash, name_map
    )
    
    # 信号栏 - 显示下一个交易日建议
    col_sig1, col_sig2 = st.columns([2, 1])
    with col_sig1:
        # 根据建议类型选择样式
        advice_type = next_day_advice['advice_type']
        advice_text = next_day_advice['advice_text']
        banner_class = {
            'hold': 'advice-banner-hold',
            'switch': 'advice-banner-switch',
            'cash': 'advice-banner-cash'
        }.get(advice_type, 'signal-banner')
        
        # 构建附加信息
        target_asset = next_day_advice['target_asset']
        current_asset = next_day_advice['current_asset']
        best_score = next_day_advice['best_score']
        current_score = next_day_advice['current_score']
        
        extra_info = f"最佳标的动量: {best_score:.2%}"
        if current_asset and current_asset != 'Cash':
            extra_info += f" | 当前持仓动量: {current_score:.2%}"
        
        data_last_date = raw_data.index[-1].strftime('%Y-%m-%d')
        
        st.markdown(f"""
        <div class="{banner_class}">
            <h3 style="margin:0">📌 {advice_text}</h3>
            <div style="margin-top:5px; font-size: 0.9rem">
                逻辑: {mom_method_curr} | 阈值: {p_threshold:.3f} | {extra_info} | 数据截止: {data_last_date}
            </div>
        </div>""", unsafe_allow_html=True)
        
    with col_sig2:
        st.markdown("**🏆 动量排名**")
        mom_ranking = next_day_advice['mom_ranking']
        for i, (asset, score) in enumerate(mom_ranking.head(3).items()):
            display_name = name_map.get(asset, asset)
            st.markdown(f"{i+1}. **{display_name}**: `{score:.2%}`")

    # === 优化引擎 (4D) ===
    with st.expander("🛠️ 策略参数优化引擎 (4D Smart Optimizer)", expanded=False):
        opt_source = st.radio(
            "优化数据源 (Data Source for Optimization)", 
            ["当前选定区间 (Selected Range)", "全历史数据 (Full History: 2015+)"],
            index=0,
            horizontal=True
        )
        
        if st.button("运行全参数扫描 (Method/L/S/T)"):
            data_to_opt = sliced_data if opt_source.startswith("当前") else raw_data
            # [修改] 使用新的不带 method 参数的 optimize_parameters (内部自带循环)
            with st.spinner(f"正在基于 [{opt_source}] 进行四维全景扫描 (约 3000+ 次回测)..."):
                opt_df = optimize_parameters(data_to_opt, p_allow_cash, p_min_holding)
                st.session_state.opt_results = opt_df 
        
        if st.session_state.opt_results is not None:
            opt_df = st.session_state.opt_results
            
            best_ret_idx = opt_df['累计收益'].idxmax()
            best_r = opt_df.loc[best_ret_idx]
            
            best_sharpe_idx = opt_df['夏普比率'].idxmax()
            best_s = opt_df.loc[best_sharpe_idx]
            
            df_low = opt_df[opt_df['年化调仓'] <= 20]
            best_low = None
            if not df_low.empty:
                best_low = df_low.loc[df_low['夏普比率'].idxmax()] 
            
            def apply_params(row_data):
                new_params = st.session_state.params.copy()
                new_params['lookback'] = int(row_data['周期'])
                new_params['smooth'] = int(row_data['平滑'])
                new_params['threshold'] = float(row_data['阈值'])
                new_params['mom_method'] = row_data['方法']
                st.session_state.params = new_params
                save_config(new_params)
                st.toast("✅ 参数已更新并保存！正在重新回测...", icon="💾")
                time.sleep(1)
                st.rerun()

            c1, c2, c3 = st.columns(3)
            # 简写 helper
            def short_method(m): return m.split(" ")[0]

            is_same = (best_r['方法'] == best_s['方法'] and int(best_r['周期']) == int(best_s['周期']) and int(best_r['平滑']) == int(best_s['平滑']) and best_r['阈值'] == best_s['阈值'])
            note_str = " (参数重合)" if is_same else ""

            with c1:
                st.markdown(f'<div class="opt-highlight">🔥 <b>收益优先</b>{note_str}</div>', unsafe_allow_html=True)
                p_str = f"{short_method(best_r['方法'])}/L{int(best_r['周期'])}/S{int(best_r['平滑'])}/T{best_r['阈值']:.3f}"
                st.write(f"**年化:** `{best_r['年化收益']:.1%}`")
                st.write(f"**夏普:** `{best_r['夏普比率']:.2f}`")
                st.write(f"**调仓:** `{best_r['年化调仓']:.1f}次/年`")
                st.caption(f"配置: {p_str}")
                if st.button("💾 应用 (收益)", key="btn_apply_ret"):
                    apply_params(best_r)

            with c2:
                st.markdown(f'<div class="opt-highlight">💎 <b>夏普优先</b>{note_str}</div>', unsafe_allow_html=True)
                p_str_s = f"{short_method(best_s['方法'])}/L{int(best_s['周期'])}/S{int(best_s['平滑'])}/T{best_s['阈值']:.3f}"
                st.write(f"**年化:** `{best_s['年化收益']:.1%}`")
                st.write(f"**夏普:** `{best_s['夏普比率']:.2f}`")
                st.write(f"**调仓:** `{best_s['年化调仓']:.1f}次/年`")
                st.caption(f"配置: {p_str_s}")
                if not is_same: 
                    if st.button("💾 应用 (夏普)", key="btn_apply_sharpe"):
                        apply_params(best_s)
                else:
                    st.caption("与左侧参数一致")
                    
            with c3:
                st.markdown('<div class="opt-highlight">🐢 <b>最佳低频 (<20次/年)</b></div>', unsafe_allow_html=True)
                if best_low is not None:
                    p_str_l = f"{short_method(best_low['方法'])}/L{int(best_low['周期'])}/S{int(best_low['平滑'])}/T{best_low['阈值']:.3f}"
                    st.write(f"**年化:** `{best_low['年化收益']:.1%}`")
                    st.write(f"**夏普:** `{best_low['夏普比率']:.2f}`")
                    st.write(f"**调仓:** `{best_low['年化调仓']:.1f}次/年`")
                    st.caption(f"配置: {p_str_l}")
                    if st.button("💾 应用 (低频)", key="btn_apply_low"):
                        apply_params(best_low)
                else:
                    st.warning("无满足条件的组合")

            st.caption("🌌 参数空间映射 (X:周期, Y:阈值, Color:年化调仓) [已展示全部方法]")
            fig_3d = px.scatter_3d(
                opt_df, 
                x='周期', y='阈值', z='平滑',
                color='年化调仓', 
                color_continuous_scale='Turbo',
                symbol='方法', 
                hover_data=['年化收益', '最大回撤', '夏普比率', '方法'],
                opacity=0.8
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=300)
            st.plotly_chart(fig_3d, use_container_width=True)

    # 报表
    account_ret = df_res['总资产'].iloc[-1] / df_res['投入本金'].iloc[-1] - 1
    account_profit = df_res['总资产'].iloc[-1] - df_res['投入本金'].iloc[-1]
    metrics = calculate_pro_metrics(df_res['策略净值'].values, bm_curve.values, trade_count_real)
    
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div class="total-asset-header">¥{df_res['总资产'].iloc[-1]:,.0f}</div>
        <div class="total-asset-sub">投入本金: ¥{df_res['投入本金'].iloc[-1]:,.0f} | <span style="color: {'#d62728' if account_profit > 0 else 'green'}">总盈亏: {account_profit:+,.0f} ({account_ret:+.2%})</span></div>
    </div>""", unsafe_allow_html=True)
    
    six_months_ago = df_res.index[-1] - timedelta(days=180)
    idx_6m = df_res.index.searchsorted(six_months_ago)
    if idx_6m < len(df_res):
        ret_6m = df_res['策略净值'].iloc[-1] / df_res['策略净值'].iloc[idx_6m] - 1
        bm_ret_6m = bm_curve.iloc[-1] / bm_curve.iloc[idx_6m] - 1
    else: ret_6m = 0.0; bm_ret_6m = 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1: st.markdown(metric_html("累计收益", f"{metrics.get('Total Return',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m2: st.markdown(metric_html("年化收益", f"{metrics.get('CAGR',0):.1%}", "", "#c0392b"), unsafe_allow_html=True)
    with m3: st.markdown(metric_html("近半年收益", f"{ret_6m:.1%}", f"超额: {ret_6m - bm_ret_6m:+.1%}", "#2980b9"), unsafe_allow_html=True)
    with m4: st.markdown(metric_html("最大回撤", f"{metrics.get('Max Drawdown',0):.1%}", "", "#27ae60"), unsafe_allow_html=True)
    with m5: st.markdown(metric_html("夏普比率", f"{metrics.get('Sharpe Ratio',0):.2f}", "", "#2c3e50"), unsafe_allow_html=True)
    with m6: st.markdown(metric_html("交易次数", f"{trade_count_real}", "", "#2c3e50"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 综合图表", "📅 年度/月度回报", "📝 交易日记"])
    with tab1:
        # [New] Asset Overlay Selection
        st.caption("📉 标的走势叠加 (Asset Overlays)")
        all_assets = sliced_data.columns.tolist()
        overlay_assets = st.multiselect(
            "选择要对比的底层资产 (Select Assets to Compare)", 
            options=all_assets,
            default=[], 
            help="选择标的后，其净值曲线将叠加显示在主图中，方便对比策略与单一资产的表现。"
        )

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=df_res.index, y=df_res['策略净值'], name="策略净值", line=dict(color='#c0392b', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_res.index, y=bm_curve, name="基准", line=dict(color='#95a5a6', dash='dash')), row=1, col=1)
        
        # Add Asset Traces
        colors = px.colors.qualitative.Plotly
        for i, asset in enumerate(overlay_assets):
            s = sliced_data[asset]
            # Normalize to 1.0 at start (or first valid) then scale to strategy start? 
            # Standard comparison: normalize to 1.0 at day 0. Strategy also starts (implied) from 1.0 base.
            if not s.empty:
                first_valid = s.first_valid_index()
                if first_valid:
                    # Normalize: s / s[0] * strategy[0] (to align starting points visually)
                    # Strategy net value[0] is (1+ret[0]). Let's align to 1.0 roughly.
                    base_val = df_res['策略净值'].iloc[0] if not df_res['策略净值'].empty else 1.0
                    s_norm = (s / s.loc[first_valid]) * base_val
                    
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s_norm, 
                        name=f"{asset} (Normalized)", 
                        mode='lines',
                        line=dict(width=1, dash='dot'),
                        opacity=0.7
                    ), row=1, col=1)

        drawdown_series = (df_res['策略净值'] - df_res['策略净值'].cummax()) / df_res['策略净值'].cummax()
        fig.add_trace(go.Scatter(x=df_res.index, y=drawdown_series, name="回撤", fill='tozeroy', line=dict(color='#c0392b', width=1)), row=2, col=1)
        
        df_res['持仓名称'] = df_res['持仓'].map(lambda x: name_map.get(x, x))
        df_res['持仓变化'] = df_res['持仓'] != df_res['持仓'].shift(1)
        change_indices = df_res[df_res['持仓变化']].index.tolist()
        if df_res.index[0] not in change_indices: change_indices.insert(0, df_res.index[0])
        change_indices.append(df_res.index[-1] + timedelta(days=1))
        
        shapes = []
        for i in range(len(change_indices) - 1):
            start_t = change_indices[i]
            end_t = change_indices[i+1]
            try:
                if start_t > df_res.index[-1]: continue
                current_code = df_res.loc[start_t, '持仓']
                current_name = df_res.loc[start_t, '持仓名称']
                color = get_color_from_name(current_code)
                shapes.append(dict(type="rect", xref="x", yref="paper", x0=start_t, x1=end_t, y0=0, y1=1, fillcolor=color, opacity=0.3, layer="below", line_width=0))
                mid_point = start_t + (end_t - start_t) / 2
                if (end_t - start_t).days > 15: 
                    fig.add_annotation(x=mid_point, y=0.05, xref="x", yref="paper", text=current_name.split(' ')[0], showarrow=False, font=dict(size=10, color="gray"), opacity=0.7)
            except Exception: pass
        fig.update_layout(shapes=shapes, height=600, title_text="策略综合分析", hovermode="x unified", xaxis=dict(rangeslider=dict(visible=False), type="date"))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        res_y = []
        years = df_res.index.year.unique()
        for y in years:
            d_sub = df_res[df_res.index.year == y]
            if d_sub.empty: continue
            y_ret = d_sub['策略净值'].iloc[-1] / d_sub['策略净值'].iloc[0] - 1
            b_ret = bm_curve.loc[d_sub.index[-1]] / bm_curve.loc[d_sub.index[0]] - 1
            res_y.append({"年份": y, "策略收益": y_ret, "基准收益": b_ret, "超额(Alpha)": y_ret - b_ret})
        st.caption("📅 年度盈亏")
        st.dataframe(pd.DataFrame(res_y).set_index("年份").style.format("{:+.2%}").background_gradient(subset=["超额(Alpha)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2), use_container_width=True)
        
        st.caption("🗓️ 月度盈亏矩阵")
        df_nav = df_res['策略净值'].resample('ME').last()
        monthly_rets = df_nav.pct_change().fillna(0)
        monthly_data = []
        for date, val in monthly_rets.items():
            monthly_data.append({'Year': date.year, 'Month': date.month, 'Return': val})
        df_month = pd.DataFrame(monthly_data)
        pivot_month = df_month.pivot(index='Year', columns='Month', values='Return')
        for m in range(1, 13):
            if m not in pivot_month.columns: pivot_month[m] = np.nan
        pivot_month = pivot_month.sort_index(ascending=False).sort_index(axis=1)
        fig_m = px.imshow(pivot_month, labels=dict(x="月份", y="年份", color="收益率"), x=[f"{i}月" for i in range(1, 13)], color_continuous_scale="RdYlGn", color_continuous_midpoint=0.0, text_auto=".1%")
        fig_m.update_layout(height=400)
        st.plotly_chart(fig_m, use_container_width=True)

    with tab3:
        st.markdown("##### 📝 详细交易日记 (Heatmap Mode)")
        df_details = pd.DataFrame(daily_details)
        if not df_details.empty:
            df_details['段内收益'] = df_details['段内收益'] * 100
            
            asset_cols = sorted([col for col in df_details.columns if col not in ["日期", "当前持仓", "持仓天数", "段内收益", "操作", "总资产", "全市场表现"]])
            
            for ac in asset_cols:
                df_details[ac] = df_details[ac] * 100
            
            col_config = {
                "持仓天数": st.column_config.NumberColumn("持仓天数", help="当前连续持仓天数"),
                "段内收益": st.column_config.NumberColumn("段内收益", help="本段持仓期间的累计收益率", format="%.2f%%"),
                "操作": st.column_config.TextColumn("调仓操作", width="medium"),
                "总资产": st.column_config.NumberColumn("总资产", format="%.2f"),
                "日期": st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
            }
            
            for ac in asset_cols:
                col_config[ac] = st.column_config.NumberColumn(ac, format="%.2f%%")

            final_cols = ["日期"] + asset_cols + ["当前持仓", "持仓天数", "段内收益", "总资产", "操作"]
            df_show = df_details[final_cols]

            st.dataframe(
                df_show.sort_values(by="日期", ascending=False).style
                .format({ac: "{:+.2f}" for ac in asset_cols}) 
                .background_gradient(subset=asset_cols, cmap="RdYlGn_r", vmin=-3.0, vmax=3.0), 
                use_container_width=True,
                column_config=col_config
            )

if __name__ == "__main__":
    main()
