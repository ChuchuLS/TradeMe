"""
Global ETF Heatmap
- Global ETF universe: US broad/sectors, Europe, Asia, EM, Bonds, Commodities, Alternatives
- Tab 1: Price % change heatmap (1D / 1W / 1M / 3M / 1Y)
- Tab 2: Technical score heatmap (-5 to +5)
"""
import streamlit as st
import pandas as pd
import numpy as np
import urllib.request, json
import plotly.graph_objects as go

st.set_page_config(page_title="ETF Heatmap", page_icon="🌍", layout="wide")

# ── ETF Universe ───────────────────────────────────────────────────────────────
ETF_UNIVERSE = {
    "🇺🇸 US Broad": {
        "SPY":  "S&P 500",
        "QQQ":  "Nasdaq 100",
        "DIA":  "Dow Jones",
        "IWM":  "Russell 2000",
        "IWB":  "Russell 1000",
        "VTI":  "Total Market",
    },
    "🇺🇸 US Sectors": {
        "XLK":  "Technology",
        "XLF":  "Financials",
        "XLV":  "Health Care",
        "XLE":  "Energy",
        "XLI":  "Industrials",
        "XLY":  "Cons. Discret.",
        "XLP":  "Cons. Staples",
        "XLU":  "Utilities",
        "XLB":  "Materials",
        "XLRE": "Real Estate",
        "XLC":  "Comm. Services",
    },
    "🌍 Europe": {
        "EZU":  "Eurozone",
        "EWG":  "Germany",
        "EWU":  "UK",
        "EWF":  "France",
        "EWI":  "Italy",
        "EWP":  "Spain",
        "EWN":  "Netherlands",
        "EWQ":  "Switzerland alt",
        "EDEN": "Denmark",
        "EWK":  "Belgium",
    },
    "🌏 Asia Pacific": {
        "EWJ":  "Japan",
        "EWY":  "South Korea",
        "EWT":  "Taiwan",
        "EWA":  "Australia",
        "EWH":  "Hong Kong",
        "EWS":  "Singapore",
        "INDA": "India",
        "FXI":  "China Large Cap",
        "MCHI": "China MSCI",
        "EWZ":  "Brazil",
    },
    "🌐 Emerging Markets": {
        "EEM":  "EM Broad",
        "VWO":  "EM Vanguard",
        "IEMG": "EM Core",
        "FM":   "Frontier Mkts",
        "EWZ":  "Brazil",
        "EWW":  "Mexico",
        "ERUS": "Russia alt",
        "TUR":  "Turkey",
        "EPHE": "Philippines",
        "THD":  "Thailand",
    },
    "💵 Bonds": {
        "TLT":  "US 20Y+ Treasury",
        "IEF":  "US 7-10Y Treasury",
        "SHY":  "US 1-3Y Treasury",
        "HYG":  "US High Yield",
        "LQD":  "US Corp IG",
        "EMB":  "EM Bonds USD",
        "BNDX": "Intl Bonds",
        "TIP":  "TIPS",
        "BWX":  "Intl Govt Bonds",
        "MUB":  "Muni Bonds",
    },
    "🥇 Commodities": {
        "GLD":  "Gold",
        "SLV":  "Silver",
        "IAU":  "Gold (iShares)",
        "PDBC": "Diversified Cmdty",
        "DJP":  "Bloomberg Cmdty",
        "USO":  "Oil WTI",
        "BNO":  "Oil Brent",
        "UNG":  "Nat Gas",
        "CORN": "Corn",
        "WEAT": "Wheat",
    },
    "🔀 Alternatives": {
        "VNQ":  "US REITs",
        "VNQI": "Intl REITs",
        "AMLP": "MLP/Energy Infra",
        "GDX":  "Gold Miners",
        "GDXJ": "Jr Gold Miners",
        "ARKK": "Innovation",
        "JETS": "Airlines",
        "XBI":  "Biotech",
        "KWEB": "China Internet",
        "SOXX": "Semiconductors",
    },
}

# ── Indicator helpers ─────────────────────────────────────────────────────────
def EMA(s,n): return s.ewm(span=n,adjust=False).mean()
def MA(s,n):  return s.rolling(n,min_periods=1).mean()
def REF(s,n): return s.shift(n)
def HHV(s,n): return s.rolling(n,min_periods=1).max()
def LLV(s,n): return s.rolling(n,min_periods=1).min()

def compute_score(df):
    C,H,L,V = df["CLOSE"],df["HIGH"],df["LOW"],df["VOLUME"]
    score = 0

    # RSI
    delta = C.diff()
    gain  = delta.clip(lower=0).ewm(span=14,adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14,adjust=False).mean()
    rsi   = (100-(100/(1+gain/loss.replace(0,np.nan)))).iloc[-1]
    if   rsi < 30: score += 1
    elif rsi > 70: score -= 1

    # MACD
    macd = (EMA(C,12)-EMA(C,26)).iloc[-1]
    sig  = EMA(EMA(C,12)-EMA(C,26),9).iloc[-1]
    score += 1 if macd > sig else -1

    # MA trend
    ma20  = MA(C,20).iloc[-1]
    ma50  = MA(C,50).iloc[-1]
    ma200 = MA(C,200).iloc[-1]
    price = C.iloc[-1]
    score += 1 if price > ma200 else -1
    score += 1 if ma20 > ma50   else -1

    # Bollinger
    bb_mid = MA(C,20); bb_std = C.rolling(20).std()
    bb_pct = ((C - (bb_mid-2*bb_std)) / (4*bb_std).replace(0,np.nan)*100).iloc[-1]
    if   bb_pct < 20: score += 1
    elif bb_pct > 80: score -= 1

    # Stochastic
    sk = ((C-LLV(L,14))/(HHV(H,14)-LLV(L,14)).replace(0,np.nan)*100).iloc[-1]
    if   sk < 20: score += 1
    elif sk > 80: score -= 1

    # Volume
    vol_ratio = (V / V.rolling(20).mean()).iloc[-1]
    chg = C.pct_change().iloc[-1]
    if vol_ratio > 1.5 and chg > 0: score += 1
    elif vol_ratio > 1.5 and chg < 0: score -= 1

    return max(-5, min(5, score))

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_ticker(ticker):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=1y&corsDomain=finance.yahoo.com")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json",
        "Referer":    "https://finance.yahoo.com",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        res = data["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        df  = pd.DataFrame({
            "Date":   pd.to_datetime(res["timestamp"],unit="s").normalize(),
            "OPEN":   q["open"],  "HIGH":  q["high"],
            "LOW":    q["low"],   "CLOSE": q["close"],
            "VOLUME": q["volume"],
        }).dropna(subset=["CLOSE"]).reset_index(drop=True)
        df = df.set_index("Date")
        return df, None
    except Exception as e:
        return None, str(e)

def get_changes(df):
    c = df["CLOSE"]
    def chg(n):
        if len(c) > n:
            return (c.iloc[-1] / c.iloc[-n-1] - 1) * 100
        return np.nan
    return {"1D": chg(1), "1W": chg(5), "1M": chg(21), "3M": chg(63), "1Y": chg(252)}

# ── Build heatmap figure ──────────────────────────────────────────────────────
def make_heatmap(groups, values, title, fmt, colorscale, zmid=0):
    """
    groups: list of group names
    values: dict of {group: {ticker: (label, value)}}
    """
    all_tickers, all_labels, all_groups, all_values = [], [], [], []

    for grp in groups:
        for ticker, (label, val) in values[grp].items():
            all_tickers.append(ticker)
            all_labels.append(f"{ticker}<br>{label}")
            all_groups.append(grp)
            all_values.append(val if val is not None else np.nan)

    # Layout: group headers as y-axis, tickers as x within each group
    # Build as a grid — each group is a row, tickers are columns
    # Find max tickers in any group for grid width
    max_cols = max(len(values[g]) for g in groups)
    n_rows   = len(groups)

    z      = []
    text   = []
    y_labs = []
    x_labs = list(range(max_cols))

    for grp in groups:
        row_vals, row_text = [], []
        items = list(values[grp].items())
        for i in range(max_cols):
            if i < len(items):
                ticker, (label, val) = items[i]
                row_vals.append(val if val is not None else np.nan)
                row_text.append(f"<b>{ticker}</b><br>{label}<br>{fmt(val)}" if val is not None else f"<b>{ticker}</b><br>N/A")
            else:
                row_vals.append(np.nan)
                row_text.append("")
        z.append(row_vals)
        text.append(row_text)
        y_labs.append(grp)

    fig = go.Figure(go.Heatmap(
        z=z, text=text,
        texttemplate="%{text}",
        textfont=dict(size=9.5),
        x=x_labs, y=y_labs,
        colorscale=colorscale,
        zmid=zmid,
        showscale=True,
        colorbar=dict(thickness=14, len=0.8, tickfont=dict(size=10)),
        hovertemplate="%{text}<extra></extra>",
        xgap=3, ygap=3,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=max(500, n_rows * 75 + 120),
        margin=dict(l=180, r=20, t=50, b=30),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🌍 Global ETF Heatmap")

# Sidebar: group filter
st.sidebar.header("Settings")
all_groups  = list(ETF_UNIVERSE.keys())
sel_groups  = st.sidebar.multiselect(
    "Show groups", all_groups, default=all_groups
)
period_tab1 = st.sidebar.selectbox("Price change period", ["1D","1W","1M","3M","1Y"], index=2)
refresh     = st.sidebar.button("🔄 Refresh", type="primary", use_container_width=True)
if refresh:
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Data via Yahoo Finance · Cache: 30 min")

if not sel_groups:
    st.info("Select at least one group in the sidebar.")
    st.stop()

# ── Fetch all tickers ─────────────────────────────────────────────────────────
all_tickers = {t: name for g in sel_groups for t, name in ETF_UNIVERSE[g].items()}
# Deduplicate (some tickers appear in multiple groups)
seen = set()
unique_tickers = {}
for t, name in all_tickers.items():
    if t not in seen:
        unique_tickers[t] = name
        seen.add(t)

progress = st.progress(0, text="Fetching ETF data…")
ticker_data = {}
for i, ticker in enumerate(unique_tickers):
    progress.progress((i+1)/len(unique_tickers), text=f"Fetching {ticker}…")
    df, err = fetch_ticker(ticker)
    if df is not None and len(df) >= 30:
        ticker_data[ticker] = df
progress.empty()

st.caption(f"Loaded {len(ticker_data)}/{len(unique_tickers)} ETFs · "
           f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")

# ── Compute values ────────────────────────────────────────────────────────────
change_vals = {}
score_vals  = {}
for ticker in ticker_data:
    ch = get_changes(ticker_data[ticker])
    change_vals[ticker] = ch
    score_vals[ticker]  = compute_score(ticker_data[ticker])

# Build per-group value dicts
def build_group_vals(sel_groups, val_fn):
    result = {}
    for grp in sel_groups:
        result[grp] = {}
        for ticker, name in ETF_UNIVERSE[grp].items():
            if ticker in ticker_data:
                result[grp][ticker] = (name, val_fn(ticker))
    return result

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Price % Change", "🎯 Technical Score"])

with tab1:
    period = period_tab1
    grp_vals = build_group_vals(
        sel_groups,
        lambda t: change_vals[t].get(period)
    )
    # Summary metrics
    all_ch = [v for g in grp_vals.values() for _,v in g.values() if v is not None]
    if all_ch:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Best",   f"{max(all_ch):+.2f}%")
        c2.metric("Worst",  f"{min(all_ch):+.2f}%")
        c3.metric("Median", f"{np.median(all_ch):+.2f}%")
        c4.metric("% Green",f"{sum(1 for v in all_ch if v>0)/len(all_ch)*100:.0f}%")

    fig1 = make_heatmap(
        sel_groups, grp_vals,
        title=f"Global ETF Price Change — {period}",
        fmt=lambda v: f"{v:+.1f}%" if v is not None else "N/A",
        colorscale=[
            [0.0, "#1d4e89"], [0.2, "#2e86c1"],
            [0.4, "#85c1e9"], [0.5, "#f5f5f5"],
            [0.6, "#f0b27a"], [0.8, "#e67e22"],
            [1.0, "#873600"],
        ],
        zmid=0,
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    grp_scores = build_group_vals(
        sel_groups,
        lambda t: score_vals.get(t)
    )
    # Summary
    all_sc = [v for g in grp_scores.values() for _,v in g.values() if v is not None]
    if all_sc:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Strongest", f"+{max(all_sc)}")
        c2.metric("Weakest",   f"{min(all_sc)}")
        c3.metric("Avg score", f"{np.mean(all_sc):+.1f}")
        n_buy = sum(1 for s in all_sc if s >= 2)
        c4.metric("Buy signals", f"{n_buy}/{len(all_sc)}")

    fig2 = make_heatmap(
        sel_groups, grp_scores,
        title="Global ETF Technical Score (-5 to +5)",
        fmt=lambda v: f"{v:+d}" if v is not None else "N/A",
        colorscale=[
            [0.0,  "#1d4e89"], [0.2,  "#2e86c1"],
            [0.4,  "#85c1e9"], [0.5,  "#f5f5f5"],
            [0.6,  "#f0b27a"], [0.8,  "#e67e22"],
            [1.0,  "#873600"],
        ],
        zmid=0,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Score table
    with st.expander("View score details"):
        rows = []
        for grp in sel_groups:
            for ticker, name in ETF_UNIVERSE[grp].items():
                if ticker in score_vals:
                    sc = score_vals[ticker]
                    ch = change_vals[ticker]
                    rows.append({
                        "Group":  grp,
                        "Ticker": ticker,
                        "Name":   name,
                        "Score":  sc,
                        "1D %":   round(ch.get("1D",np.nan),2),
                        "1W %":   round(ch.get("1W",np.nan),2),
                        "1M %":   round(ch.get("1M",np.nan),2),
                    })
        if rows:
            df_tbl = pd.DataFrame(rows).sort_values("Score", ascending=False)
            st.dataframe(df_tbl, use_container_width=True, hide_index=True)
