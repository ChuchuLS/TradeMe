"""
Stock Indicator Dashboard
散户/主力/走势 iFind indicator system
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request, json

st.set_page_config(
    page_title="Stock Indicator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Indicator math ─────────────────────────────────────────────────────────────
def HHV(s, n): return s.rolling(n, min_periods=1).max()
def LLV(s, n): return s.rolling(n, min_periods=1).min()
def SMA_w(s, n, m): return s.ewm(alpha=m/n, adjust=False).mean()
def EMA(s, n): return s.ewm(span=n, adjust=False).mean()
def MA(s, n): return s.rolling(n, min_periods=1).mean()
def REF(s, n): return s.shift(n)
def IF(cond, a, b): return pd.Series(np.where(cond, a, b), index=cond.index)
def CEILING(s): return np.ceil(s)

def FILTER(cond, n):
    result = pd.Series(False, index=cond.index)
    last = -n - 1
    for i in range(len(cond)):
        if cond.iloc[i] and (i - last) > n:
            result.iloc[i] = True
            last = i
    return result

def CROSS(a, b):
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return (a > b) & (REF(a, 1) <= REF(b, 1))

def compute_indicator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    C, H, L = df["CLOSE"], df["HIGH"], df["LOW"]

    hhv60, llv60 = HHV(H, 60), LLV(L, 60)
    df["散户"] = 100 * (hhv60 - C) / (hhv60 - llv60).replace(0, np.nan)

    hhv30, llv30 = HHV(H, 30), LLV(L, 30)
    RSV  = (C - llv30) / (hhv30 - llv30).replace(0, np.nan) * 100
    RSV1 = (C - MA(C, 4)) / MA(C, 4).replace(0, np.nan) * 100

    VAR2 = REF(L, 1)
    diff = L - VAR2
    VAR3 = SMA_w(diff.abs(), 3, 1) / SMA_w(diff.clip(lower=0), 3, 1).replace(0, np.nan) * 100
    VAR4 = EMA(IF(C * 1.3 != 0, VAR3 * 10, VAR3 / 10), 3)
    VAR5 = LLV(L, 30)
    VAR6 = HHV(VAR4, 30)
    VAR7 = IF(MA(C, 58) != 0, 1, 0)
    VAR8 = EMA(IF(L <= VAR5, (VAR4 + VAR6 * 2) / 2, 0), 3) / 618 * VAR7
    df["吸筹"] = VAR8.clip(upper=100)

    K = SMA_w(RSV, 5, 1)
    D = SMA_w(K, 3, 1)
    J = 3 * K - 2 * D
    df["主力"] = EMA(J, 6)
    df["J"]   = J

    hhv21, llv21 = HHV(H, 21), LLV(L, 21)
    d21  = (hhv21 - llv21).replace(0, np.nan)
    AA3  = (hhv21 - C) / d21 * 100 - 10
    AA4  = (C - llv21) / d21 * 100
    AA5  = SMA_w(AA4, 13, 8)
    df["走势"] = CEILING(SMA_w(AA5, 13, 8))
    AA6  = SMA_w(AA3, 21, 8)

    df["CDMA"] = EMA(EMA(RSV1, 7), 7) * 8

    hhv9, llv9 = HHV(H, 9), LLV(L, 9)
    df["快线"] = (C - llv9) / (hhv9 - llv9).replace(0, np.nan) * 100

    buy_raw  = (J > REF(J, 1)) & (df["走势"] >= REF(df["走势"], 1)) & (df["走势"] < 25)
    sell_raw = (J < REF(J, 1)) & (df["走势"] <= REF(df["走势"], 1)) & (J > 85)
    df["买入信号"] = FILTER(buy_raw, 5)
    df["卖出信号"] = FILTER(sell_raw, 5)
    df["关注低买"] = (pd.Series(90, index=df.index) > df["散户"]) & CROSS(pd.Series(90, index=df.index), df["散户"])
    df["观注顶"]  = (df["散户"] < 8) & (df["主力"] > 90) & (df["走势"] > 90) & (df["CDMA"] <= REF(df["CDMA"], 1))

    return df

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_yahoo(ticker: str, period: str = "2y"):
    period_map = {"1Y": "1y", "2Y": "2y", "5Y": "5y", "Max": "10y"}
    yf_period  = period_map.get(period, "2y")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker.upper()}"
           f"?interval=1d&range={yf_period}&corsDomain=finance.yahoo.com")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json",
        "Referer":    "https://finance.yahoo.com",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        result = data["chart"]["result"][0]
        meta   = result["meta"]
        ts     = result["timestamp"]
        q      = result["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Date":   pd.to_datetime(ts, unit="s").normalize(),
            "OPEN":   q["open"],  "HIGH":   q["high"],
            "LOW":    q["low"],   "CLOSE":  q["close"],
            "VOLUME": q["volume"],
        }).dropna(subset=["CLOSE"]).reset_index(drop=True)
        df = df.set_index("Date")
        name = meta.get("shortName") or meta.get("longName") or ticker
        return df, name, None
    except Exception as e:
        return None, None, str(e)

# ── Chart ─────────────────────────────────────────────────────────────────────
def build_chart(result: pd.DataFrame, ticker: str, name: str) -> go.Figure:
    r     = result.reset_index()
    dates = r["Date"].dt.strftime("%Y-%m-%d").tolist()

    buy_y  = [r["CLOSE"].iloc[i] if r["买入信号"].iloc[i]  else None for i in range(len(r))]
    sell_y = [r["CLOSE"].iloc[i] if r["卖出信号"].iloc[i] else None for i in range(len(r))]
    low_y  = [r["CLOSE"].iloc[i] if r["关注低买"].iloc[i]  else None for i in range(len(r))]
    top_y  = [r["CLOSE"].iloc[i] if r["观注顶"].iloc[i]   else None for i in range(len(r))]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.25, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{ticker.upper()}  {name}  —  price & signals",
            "散户 retail  /  主力 smart money  /  走势 trend  /  快线 fast-K",
            "吸筹  accumulation",
            "CDMA  momentum",
        ]
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=dates, y=r["CLOSE"].round(2),
        mode="lines", line=dict(color="#aaaaaa", width=1.5),
        name="price",
    ), row=1, col=1)

    # Signal markers
    for y_data, color, symbol, lbl, sz in [
        (buy_y,  "#4ade80", "triangle-up",   "buy signal",   12),
        (sell_y, "#f87171", "triangle-down",  "sell signal",  12),
        (low_y,  "#60a5fa", "star",           "watch low buy", 10),
        (top_y,  "#c084fc", "diamond",        "watch top",     10),
    ]:
        valid = [(d, v) for d, v in zip(dates, y_data) if v is not None]
        if valid:
            dx, dy = zip(*valid)
            fig.add_trace(go.Scatter(
                x=list(dx), y=[round(v, 2) for v in dy],
                mode="markers",
                marker=dict(symbol=symbol, size=sz, color=color,
                            line=dict(width=1, color="#222")),
                name=lbl,
            ), row=1, col=1)

    # Oscillators
    for col_name, color, dash, width in [
        ("散户",  "#ffffff", "dot",   1.2),
        ("主力",  "#f0c040", "solid", 1.5),
        ("走势",  "#e05050", "solid", 2.0),
        ("快线",  "#50c8ff", "solid", 1.0),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=r[col_name].round(1),
            mode="lines", line=dict(color=color, dash=dash, width=width),
            name=col_name,
        ), row=2, col=1)

    for lvl, col in [(80, "rgba(248,113,113,0.3)"), (20, "rgba(74,222,128,0.3)")]:
        fig.add_hline(y=lvl, line_dash="dash", line_color=col,
                      line_width=1, row=2, col=1)

    # Accumulation
    fig.add_trace(go.Bar(
        x=dates, y=r["吸筹"].round(2),
        marker_color="rgba(255,200,0,0.7)", name="吸筹",
    ), row=3, col=1)

    # CDMA
    cdma_colors = ["rgba(74,222,128,0.75)" if v >= 0 else "rgba(248,113,113,0.75)"
                   for v in r["CDMA"]]
    fig.add_trace(go.Bar(
        x=dates, y=r["CDMA"].round(2),
        marker_color=cdma_colors, name="CDMA",
    ), row=4, col=1)
    fig.add_hline(y=0, line_dash="dash",
                  line_color="rgba(150,150,150,0.4)", line_width=1, row=4, col=1)

    rb = [dict(bounds=["sat", "mon"], dvalue=86400000)]
    fig.update_layout(
        height=900,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=60, b=30),
        barmode="overlay",
    )
    for row in [1, 2, 3, 4]:
        fig.update_xaxes(type="date", showgrid=False, rangebreaks=rb, row=row, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                     tickfont=dict(size=10))
    fig.update_yaxes(range=[0, 105], row=2, col=1)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
ticker  = st.sidebar.text_input(
    "Ticker symbol",
    value="NVDA",
    placeholder="AAPL / TSLA / 0700.HK / 600519.SS",
    help="US: AAPL  |  HK: 0700.HK  |  A-share: 600519.SS  |  ETF: SPY",
).strip().upper()
period   = st.sidebar.selectbox("History", ["1Y", "2Y", "5Y", "Max"], index=1)
st.sidebar.divider()
st.sidebar.caption(
    "Data via Yahoo Finance (free, 15-min delay).\n\n"
    "Indicator translated from iFind/TongHuaShun formula system."
)
run = st.sidebar.button("Run indicator", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📊 Stock Indicator")
st.caption("散户 / 主力 / 走势 — iFind custom indicator system")

if not run and "result" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click **Run indicator**.")
    st.stop()

if run:
    with st.spinner(f"Fetching {ticker}…"):
        df_raw, name, err = fetch_yahoo(ticker, period)
    if err or df_raw is None:
        st.error(f"Could not fetch **{ticker}**: {err}")
        st.caption("Check the ticker format. US stocks need no suffix. "
                   "HK stocks: 0700.HK  |  Shanghai: 600519.SS  |  Shenzhen: 000858.SZ")
        st.stop()
    with st.spinner("Computing indicator…"):
        result = compute_indicator(df_raw)
    st.session_state["result"] = result
    st.session_state["ticker"] = ticker
    st.session_state["name"]   = name

result = st.session_state.get("result")
ticker = st.session_state.get("ticker", ticker)
name   = st.session_state.get("name",   ticker)

if result is not None:
    # Metrics
    latest = result.iloc[-1]
    prev   = result.iloc[-2]
    cols   = st.columns(6)
    cols[0].metric("Price",        f"${latest['CLOSE']:.2f}",
                   f"{latest['CLOSE']-prev['CLOSE']:+.2f}")
    cols[1].metric("散户 retail",  f"{latest['散户']:.1f}",
                   f"{latest['散户']-prev['散户']:+.1f}", delta_color="inverse")
    cols[2].metric("主力 smart $", f"{latest['主力']:.1f}",
                   f"{latest['主力']-prev['主力']:+.1f}")
    cols[3].metric("走势 trend",   f"{latest['走势']:.0f}",
                   f"{latest['走势']-prev['走势']:+.0f}")
    cols[4].metric("CDMA",         f"{latest['CDMA']:.2f}",
                   f"{latest['CDMA']-prev['CDMA']:+.2f}")
    cols[5].metric("快线 fast-K",  f"{latest['快线']:.1f}",
                   f"{latest['快线']-prev['快线']:+.1f}")

    # Signal banners
    last30 = result.tail(30)
    if last30["买入信号"].sum():
        st.success(f"Buy signal triggered {int(last30['买入信号'].sum())}x in the last 30 days")
    if last30["卖出信号"].sum():
        st.warning(f"Sell signal triggered {int(last30['卖出信号'].sum())}x in the last 30 days")
    if last30["观注顶"].sum():
        st.error("Watch-top alert active in the last 30 days — exercise caution")

    # Chart
    fig = build_chart(result, ticker, name)
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    with st.expander("View raw indicator values"):
        display_cols = ["CLOSE","散户","主力","走势","吸筹","CDMA","快线",
                        "买入信号","卖出信号","关注低买","观注顶"]
        st.dataframe(
            result[display_cols].tail(60).sort_index(ascending=False).round(2),
            use_container_width=True,
        )
