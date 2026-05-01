"""
Quantitative Technical Analysis Watchlist
- Fixed watchlist + add/remove tickers
- Standard indicators: RSI, MACD, MA cross, Bollinger, Stochastic, ATR, Volume
- iFind signals: shown only when buy/sell/watch triggers
- Strength score: -5 (strong sell) to +5 (strong buy)
"""
import streamlit as st
import pandas as pd
import numpy as np
import urllib.request, json
from datetime import datetime

st.set_page_config(page_title="Watchlist", page_icon="📋", layout="wide")

# ── Default watchlist ─────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "NVDA","AAPL","MSFT",          # US equity
    "SPY","QQQ","GLD","TLT",       # ETFs
    "BTC-USD","ETH-USD",           # Crypto
    "GC=F","CL=F",                 # Commodities futures
    "EURUSD=X","USDJPY=X",         # FX
    "^GSPC","^VIX",                # Indices
]

# ── Indicator calculations ────────────────────────────────────────────────────
def HHV(s,n): return s.rolling(n,min_periods=1).max()
def LLV(s,n): return s.rolling(n,min_periods=1).min()
def EMA(s,n): return s.ewm(span=n,adjust=False).mean()
def MA(s,n):  return s.rolling(n,min_periods=1).mean()
def REF(s,n): return s.shift(n)
def SMA_w(s,n,m): return s.ewm(alpha=m/n,adjust=False).mean()
def IF(c,a,b): return pd.Series(np.where(c,a,b),index=c.index)
def CEILING(s): return np.ceil(s)
def FILTER(cond,n):
    r=pd.Series(False,index=cond.index); last=-n-1
    for i in range(len(cond)):
        if cond.iloc[i] and (i-last)>n: r.iloc[i]=True; last=i
    return r
def CROSS(a,b):
    if not isinstance(b,pd.Series): b=pd.Series(b,index=a.index)
    return (a>b)&(REF(a,1)<=REF(b,1))
def pct_rank(s,n=120):
    return s.rolling(n,min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]*100, raw=False)

def compute_all(df):
    C,H,L,V = df["CLOSE"],df["HIGH"],df["LOW"],df["VOLUME"]

    # ── Standard indicators ───────────────────────────────────────────────────
    # RSI (14)
    delta = C.diff()
    gain  = delta.clip(lower=0).ewm(span=14,adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14,adjust=False).mean()
    rsi   = 100 - (100/(1+gain/loss.replace(0,np.nan)))

    # MACD (12,26,9)
    macd_line   = EMA(C,12) - EMA(C,26)
    signal_line = EMA(macd_line,9)
    macd_hist   = macd_line - signal_line

    # Moving averages
    ma20  = MA(C,20);  ma50  = MA(C,50);  ma200 = MA(C,200)
    ema9  = EMA(C,9);  ema21 = EMA(C,21)

    # Bollinger Bands (20,2)
    bb_mid  = ma20
    bb_std  = C.rolling(20).std()
    bb_up   = bb_mid + 2*bb_std
    bb_lo   = bb_mid - 2*bb_std
    bb_pct  = (C - bb_lo)/(bb_up - bb_lo).replace(0,np.nan)*100  # 0=bottom, 100=top

    # Stochastic (14,3)
    low14  = LLV(L,14); high14 = HHV(H,14)
    stoch_k = (C-low14)/(high14-low14).replace(0,np.nan)*100
    stoch_d = stoch_k.rolling(3).mean()

    # Volume ratio (vs 20-day avg)
    vol_ratio = V / V.rolling(20).mean()

    # ATR (14) — for volatility context
    tr  = pd.concat([H-L,(H-REF(C,1)).abs(),(L-REF(C,1)).abs()],axis=1).max(axis=1)
    atr = tr.ewm(span=14,adjust=False).mean()

    # Trend regime
    ma60   = MA(C,60);  ma120 = MA(C,120)
    uptrend   = (ma20>ma60)&(ma60>ma120)
    downtrend = (ma20<ma60)&(ma60<ma120)
    ma60_slope= (ma60-REF(ma60,20))/REF(ma60,20)*100
    strong_up = uptrend&(ma60_slope>3)

    # ── iFind indicators ──────────────────────────────────────────────────────
    hhv60,llv60 = HHV(H,60),LLV(L,60)
    sanhu = 100*(hhv60-C)/(hhv60-llv60).replace(0,np.nan)

    hhv30,llv30 = HHV(H,30),LLV(L,30)
    RSV  = (C-llv30)/(hhv30-llv30).replace(0,np.nan)*100
    RSV1 = (C-MA(C,4))/MA(C,4).replace(0,np.nan)*100

    VAR2=REF(L,1); diff=L-VAR2
    VAR3=SMA_w(diff.abs(),3,1)/SMA_w(diff.clip(lower=0),3,1).replace(0,np.nan)*100
    VAR4=EMA(IF(C*1.3!=0,VAR3*10,VAR3/10),3)
    VAR5=LLV(L,30); VAR6=HHV(VAR4,30); VAR7=IF(MA(C,58)!=0,1,0)
    VAR8=EMA(IF(L<=VAR5,(VAR4+VAR6*2)/2,0),3)/618*VAR7

    K=SMA_w(RSV,5,1); D=SMA_w(K,3,1); J=3*K-2*D
    zhuli=EMA(J,6)

    hhv21,llv21=HHV(H,21),LLV(L,21); d21=(hhv21-llv21).replace(0,np.nan)
    AA4=(C-llv21)/d21*100; AA5=SMA_w(AA4,13,8)
    zoushi=CEILING(SMA_w(AA5,13,8))
    cdma=EMA(EMA(RSV1,7),7)*8

    # Percentile ranks
    sanhu_r=pct_rank(sanhu); zhuli_r=pct_rank(zhuli)
    zoushi_r=pct_rank(zoushi); cdma_r=pct_rank(cdma)

    # iFind buy signal
    buy_base=(J>REF(J,1))&(zoushi>=REF(zoushi,1))&(zoushi_r<20)
    ifind_buy = FILTER(buy_base,5)

    # iFind sell signal
    sell_base=(sanhu_r<15)&(zhuli_r>85)&(zoushi_r>80)&\
              (cdma<REF(cdma,1))&(cdma<REF(cdma,2))&\
              (zhuli<REF(zhuli,1))&(zhuli<REF(zhuli,2))
    recent_high=HHV(H,20); exhaustion=(C>=recent_high*0.97)&(C<REF(C,3))
    sell_up=sell_base&exhaustion
    sell_filtered=IF(strong_up,sell_up,sell_base).astype(bool)
    ifind_sell=FILTER(sell_filtered,20)

    # iFind watch signals
    ifind_watch_buy  = (pd.Series(90,index=C.index)>sanhu)&CROSS(pd.Series(90,index=C.index),sanhu)
    ifind_watch_top  = (sanhu_r<8)&(zhuli_r>92)&(zoushi_r>88)&\
                       (cdma_r<REF(cdma_r,1))&(zhuli<REF(zhuli,3))

    return {
        # Price
        "close":    C,
        "change_pct": C.pct_change()*100,

        # Standard
        "rsi":      rsi,
        "macd_hist":macd_hist,
        "macd_line":macd_line,
        "signal_line":signal_line,
        "bb_pct":   bb_pct,
        "stoch_k":  stoch_k,
        "stoch_d":  stoch_d,
        "ma20":     ma20, "ma50":ma50, "ma200":ma200,
        "vol_ratio":vol_ratio,
        "atr":      atr,

        # Trend
        "uptrend":  uptrend, "downtrend":downtrend, "strong_up":strong_up,

        # iFind
        "sanhu":    sanhu, "zhuli":zhuli, "zoushi":zoushi, "cdma":cdma,
        "sanhu_r":  sanhu_r, "zhuli_r":zhuli_r, "zoushi_r":zoushi_r,
        "ifind_buy":ifind_buy, "ifind_sell":ifind_sell,
        "ifind_watch_buy":ifind_watch_buy, "ifind_watch_top":ifind_watch_top,
    }

def score_row(ind, price):
    """
    Compute strength score -5 to +5.
    Each indicator contributes +1 (bullish), -1 (bearish), or 0 (neutral).
    Max 5 points each direction.
    """
    score = 0
    signals = []

    # ── Standard indicators (max ±5) ─────────────────────────────────────────
    # RSI
    r = ind["rsi"]
    if   r < 30:  score += 1; signals.append(("RSI oversold",    +1))
    elif r > 70:  score -= 1; signals.append(("RSI overbought",  -1))

    # MACD
    if ind["macd_hist"] > 0 and ind["macd_hist"] > ind["macd_hist"]:
        score += 1; signals.append(("MACD bullish", +1))
    if ind["macd_line"] > ind["signal_line"]:
        score += 1; signals.append(("MACD above signal", +1))
    else:
        score -= 1; signals.append(("MACD below signal", -1))

    # MA trend
    if price > ind["ma200"]:
        score += 1; signals.append(("Above MA200", +1))
    else:
        score -= 1; signals.append(("Below MA200", -1))
    if ind["ma20"] > ind["ma50"]:
        score += 1; signals.append(("MA20>MA50", +1))
    else:
        score -= 1; signals.append(("MA20<MA50", -1))

    # Bollinger
    bb = ind["bb_pct"]
    if   bb < 20: score += 1; signals.append(("Near BB lower",  +1))
    elif bb > 80: score -= 1; signals.append(("Near BB upper",  -1))

    # Stochastic
    sk = ind["stoch_k"]
    if   sk < 20: score += 1; signals.append(("Stoch oversold", +1))
    elif sk > 80: score -= 1; signals.append(("Stoch overbought",-1))

    # Volume confirmation
    if ind["vol_ratio"] > 1.5 and ind["change_pct"] > 0:
        score += 1; signals.append(("High vol breakout", +1))
    elif ind["vol_ratio"] > 1.5 and ind["change_pct"] < 0:
        score -= 1; signals.append(("High vol breakdown",-1))

    # Cap standard score at ±5
    score = max(-5, min(5, score))

    # ── iFind signals (bonus, shown separately) ───────────────────────────────
    ifind_tags = []
    if ind["ifind_buy"]:
        score = min(5, score + 1)
        ifind_tags.append("🟢 买入")
    if ind["ifind_watch_buy"]:
        score = min(5, score + 1)   # 关注低买 = early warning of buy, add +1
        ifind_tags.append("🔵 关注低买")
    if ind["ifind_sell"]:
        score = max(-5, score - 1)
        ifind_tags.append("🔴 卖出")
    if ind["ifind_watch_top"]:
        score = max(-5, score - 1)
        ifind_tags.append("⚠️ 观注顶")

    return score, signals, ifind_tags

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_ticker(ticker):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval=1d&range=2y&corsDomain=finance.yahoo.com")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "application/json",
        "Referer":    "https://finance.yahoo.com",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        res  = data["chart"]["result"][0]
        meta = res["meta"]
        q    = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Date":   pd.to_datetime(res["timestamp"],unit="s").normalize(),
            "OPEN":   q["open"],  "HIGH":  q["high"],
            "LOW":    q["low"],   "CLOSE": q["close"],
            "VOLUME": q["volume"],
        }).dropna(subset=["CLOSE"]).reset_index(drop=True).set_index("Date")
        name = meta.get("shortName") or ticker
        return df, name, None
    except Exception as e:
        return None, None, str(e)

def analyse_ticker(ticker):
    df, name, err = fetch_ticker(ticker)
    if err or df is None:
        return None
    ind = compute_all(df)
    # Get latest values
    latest = {k: float(v.iloc[-1]) if hasattr(v,"iloc") and not isinstance(v.iloc[-1],bool)
              else (bool(v.iloc[-1]) if hasattr(v,"iloc") else v)
              for k,v in ind.items()}
    score, signals, ifind_tags = score_row(latest, latest["close"])
    return {
        "ticker":     ticker,
        "name":       name[:22] if name else ticker,
        "price":      latest["close"],
        "chg%":       latest["change_pct"],
        "score":      score,
        "rsi":        latest["rsi"],
        "macd_hist":  latest["macd_hist"],
        "bb%":        latest["bb_pct"],
        "stoch":      latest["stoch_k"],
        "vol_ratio":  latest["vol_ratio"],
        "trend":      "↑ strong" if latest["strong_up"] else
                      ("↑" if latest["uptrend"] else ("↓" if latest["downtrend"] else "→")),
        "散户":       latest["sanhu"],
        "主力":       latest["zhuli"],
        "走势":       latest["zoushi"],
        "CDMA":       latest["cdma"],
        "iFind":      " | ".join(ifind_tags) if ifind_tags else "—",
        "signals":    signals,
    }

# ── Score formatting ──────────────────────────────────────────────────────────
def fmt_score(s):
    if   s >= 4:  return "🟢🟢 STRONG BUY"
    elif s >= 2:  return "🟢 BUY"
    elif s >= 1:  return "🟡 WEAK BUY"
    elif s == 0:  return "⚪ NEUTRAL"
    elif s >= -1: return "🟠 WEAK SELL"
    elif s >= -3: return "🔴 SELL"
    else:         return "🔴🔴 STRONG SELL"

def color_score(s):
    if   s >= 3:  return "background-color:#1a3a1a;color:#4ade80"
    elif s >= 1:  return "background-color:#2a3a1a;color:#a3e635"
    elif s == 0:  return "background-color:#2a2a2a;color:#aaa"
    elif s >= -2: return "background-color:#3a2a1a;color:#fb923c"
    else:         return "background-color:#3a1a1a;color:#f87171"

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📋 Technical Analysis Watchlist")

# Sidebar — watchlist management
st.sidebar.header("Watchlist")
if "tickers" not in st.session_state:
    st.session_state["tickers"] = DEFAULT_TICKERS.copy()

add_col, rem_col = st.sidebar.columns(2)
new_ticker = add_col.text_input("Add ticker", placeholder="e.g. NFLX").strip().upper()
if add_col.button("Add", use_container_width=True) and new_ticker:
    if new_ticker not in st.session_state["tickers"]:
        st.session_state["tickers"].append(new_ticker)
        st.rerun()

remove_ticker = rem_col.selectbox("Remove", ["—"] + st.session_state["tickers"])
if rem_col.button("Remove", use_container_width=True) and remove_ticker != "—":
    st.session_state["tickers"].remove(remove_ticker)
    st.rerun()

st.sidebar.markdown("**Current watchlist:**")
st.sidebar.caption("  |  ".join(st.session_state["tickers"]))
st.sidebar.caption(
    "Formats: stocks `AAPL` · crypto `BTC-USD` · "
    "FX `EURUSD=X` · futures `GC=F` · "
    "indices `^GSPC` · HK `0700.HK` · "
    "China `600519.SS`"
)

st.sidebar.divider()
sort_by   = st.sidebar.selectbox("Sort by", ["Score","RSI","Stoch","Vol ratio","趋势"])
show_signals = st.sidebar.checkbox("Show signal detail", value=False)
auto_refresh = st.sidebar.checkbox("Auto-refresh (15 min)", value=False)
if auto_refresh:
    import time; st.rerun() if (time.time() % 900 < 5) else None

refresh = st.sidebar.button("🔄 Refresh all", type="primary", use_container_width=True)
if refresh:
    st.cache_data.clear()

# ── Fetch and compute ─────────────────────────────────────────────────────────
tickers = st.session_state["tickers"]
results = []

progress = st.progress(0, text="Fetching data…")
for i, t in enumerate(tickers):
    progress.progress((i+1)/len(tickers), text=f"Analysing {t}…")
    row = analyse_ticker(t)
    if row:
        results.append(row)
progress.empty()

if not results:
    st.error("No data fetched. Check your internet connection.")
    st.stop()

# ── Sort ──────────────────────────────────────────────────────────────────────
sort_map = {"Score":"score","RSI":"rsi","Stoch":"stoch",
            "Vol ratio":"vol_ratio","趋势":"trend"}
results.sort(key=lambda x: x[sort_map[sort_by]], reverse=(sort_by!="趋势"))

# ── Summary metrics ───────────────────────────────────────────────────────────
scores   = [r["score"] for r in results]
n_buy    = sum(1 for s in scores if s >= 2)
n_sell   = sum(1 for s in scores if s <= -2)
n_neut   = len(scores) - n_buy - n_sell
avg_score= round(sum(scores)/len(scores), 1)
ifind_ct = sum(1 for r in results if r["iFind"] != "—")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Watching",   len(results))
c2.metric("Buy signals",  n_buy,  delta_color="normal")
c3.metric("Sell signals", n_sell, delta_color="inverse")
c4.metric("Avg score",    avg_score)
c5.metric("iFind alerts", ifind_ct)

st.divider()

# ── Main table ────────────────────────────────────────────────────────────────
st.markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*")

# Build display dataframe
rows = []
for r in results:
    rows.append({
        "Ticker":    r["ticker"],
        "Name":      r["name"],
        "Price":     f"${r['price']:.2f}",
        "Chg %":     f"{r['chg%']:+.2f}%",
        "Score":     r["score"],
        "Signal":    fmt_score(r["score"]),
        "Trend":     r["trend"],
        "RSI":       round(r["rsi"],1),
        "MACD hist": round(r["macd_hist"],3),
        "BB %":      round(r["bb%"],1),
        "Stoch K":   round(r["stoch"],1),
        "Vol×":      round(r["vol_ratio"],2),
        "散户":      round(r["散户"],1),
        "主力":      round(r["主力"],1),
        "走势":      round(r["走势"],1),
        "CDMA":      round(r["CDMA"],2),
        "iFind alert": r["iFind"],
    })

df_display = pd.DataFrame(rows)

def style_table(df):
    styled = df.style

    # Score column — color by value
    def score_cell(val):
        try:
            s = int(val)
            if   s >= 3: return "background-color:#1a3a1a;color:#4ade80;font-weight:500"
            elif s >= 1: return "background-color:#1e2e10;color:#a3e635"
            elif s == 0: return "color:#888"
            elif s >= -2:return "background-color:#2e1e10;color:#fb923c"
            else:        return "background-color:#2e1010;color:#f87171;font-weight:500"
        except: return ""

    styled = styled.map(score_cell, subset=["Score"])

    # RSI coloring
    def rsi_cell(val):
        try:
            v = float(val)
            if v < 30: return "color:#4ade80"
            if v > 70: return "color:#f87171"
            return ""
        except: return ""
    styled = styled.map(rsi_cell, subset=["RSI"])

    # Stoch coloring
    def stoch_cell(val):
        try:
            v = float(val)
            if v < 20: return "color:#4ade80"
            if v > 80: return "color:#f87171"
            return ""
        except: return ""
    styled = styled.map(stoch_cell, subset=["Stoch K"])

    # Chg% coloring
    def chg_cell(val):
        try:
            v = float(str(val).replace("%",""))
            if v > 0: return "color:#4ade80"
            if v < 0: return "color:#f87171"
            return ""
        except: return ""
    styled = styled.map(chg_cell, subset=["Chg %"])

    # iFind alert highlight
    def ifind_cell(val):
        if "买入" in str(val): return "color:#4ade80;font-weight:500"
        if "卖出" in str(val): return "color:#f87171;font-weight:500"
        if "关注" in str(val): return "color:#60a5fa"
        if "顶" in str(val):   return "color:#c084fc;font-weight:500"
        return "color:#555"
    styled = styled.map(ifind_cell, subset=["iFind alert"])

    return styled.hide(axis="index")

st.dataframe(
    style_table(df_display),
    use_container_width=True,
    height=min(60 + len(results)*38, 700),
)

# ── Signal detail (expandable per ticker) ─────────────────────────────────────
if show_signals:
    st.divider()
    st.subheader("Signal breakdown")
    for r in results:
        score = r["score"]
        color = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")
        with st.expander(f"{color} {r['ticker']} — {fmt_score(score)}  (score: {score:+d})"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Standard signals:**")
                for name, val in r["signals"]:
                    icon = "🟢" if val > 0 else "🔴"
                    st.markdown(f"{icon} {name} ({val:+d})")
            with sc2:
                st.markdown("**iFind alerts:**")
                if r["iFind"] != "—":
                    for tag in r["iFind"].split(" | "):
                        st.markdown(tag)
                else:
                    st.caption("No active iFind signals")
                st.markdown("**Indicator values:**")
                st.caption(
                    f"散户: {r['散户']:.1f}  |  主力: {r['主力']:.1f}  |  "
                    f"走势: {r['走势']:.0f}  |  CDMA: {r['CDMA']:.2f}"
                )
