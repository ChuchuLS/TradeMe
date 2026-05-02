"""
Relative Rotation Graph (RRG)
- ETFs vs benchmark (default SPY)
- Uses same ETF universe as heatmap
- Configurable tail length 1-12 weeks
- Quadrants: Leading / Weakening / Lagging / Improving
"""
import streamlit as st
import pandas as pd
import numpy as np
import urllib.request, json
import plotly.graph_objects as go

st.set_page_config(page_title="RRG", page_icon="🔄", layout="wide")

# ── ETF Universe (same as heatmap) ────────────────────────────────────────────
ETF_UNIVERSE = {
    "🇺🇸 US Broad":       {"SPY":"S&P 500","QQQ":"Nasdaq 100","DIA":"Dow Jones","IWM":"Russell 2000","VTI":"Total Market"},
    "🇺🇸 US Sectors":     {"XLK":"Technology","XLF":"Financials","XLV":"Health Care","XLE":"Energy","XLI":"Industrials","XLY":"Cons. Discret.","XLP":"Cons. Staples","XLU":"Utilities","XLB":"Materials","XLRE":"Real Estate","XLC":"Comm. Services"},
    "🌍 Europe":           {"EZU":"Eurozone","EWG":"Germany","EWU":"UK","EWF":"France","EWI":"Italy","EWP":"Spain","EWN":"Netherlands"},
    "🌏 Asia Pacific":     {"EWJ":"Japan","EWY":"S. Korea","EWT":"Taiwan","EWA":"Australia","EWH":"Hong Kong","EWS":"Singapore","INDA":"India","FXI":"China"},
    "🌐 Emerging Markets": {"EEM":"EM Broad","VWO":"EM Vanguard","EWZ":"Brazil","EWW":"Mexico","TUR":"Turkey"},
    "💵 Bonds":            {"TLT":"US 20Y+","IEF":"US 7-10Y","SHY":"US 1-3Y","HYG":"High Yield","LQD":"Corp IG","EMB":"EM Bonds","TIP":"TIPS"},
    "🥇 Commodities":      {"GLD":"Gold","SLV":"Silver","USO":"Oil WTI","UNG":"Nat Gas","PDBC":"Diversified"},
    "🔀 Alternatives":     {"VNQ":"US REITs","GDX":"Gold Miners","ARKK":"Innovation","SOXX":"Semiconductors","XBI":"Biotech","KWEB":"China Internet"},
}

ALL_ETFS = {t: n for g in ETF_UNIVERSE.values() for t, n in g.items()}

# ── RRG Math ──────────────────────────────────────────────────────────────────
def compute_rrg(prices: pd.DataFrame, benchmark: str, period: int = 52) -> pd.DataFrame:
    """
    Standard RRG calculation:
    1. RS-Ratio  = (asset/benchmark) relative strength, normalised to 100
    2. RS-Momentum = rate of change of RS-Ratio, normalised to 100
    Both axes centred at 100.
    """
    if benchmark not in prices.columns:
        return pd.DataFrame()

    bench = prices[benchmark]
    results = {}

    for ticker in prices.columns:
        if ticker == benchmark:
            continue
        if prices[ticker].isna().all():
            continue

        # Relative strength vs benchmark
        rs = prices[ticker] / bench

        # Smooth RS with 10-period EMA (standard RRG uses JdK RS-Ratio)
        rs_smooth = rs.ewm(span=10, adjust=False).mean()

        # Normalise to 100-based index over rolling `period` weeks
        rs_min = rs_smooth.rolling(period, min_periods=period//2).min()
        rs_max = rs_smooth.rolling(period, min_periods=period//2).max()
        rs_ratio = 100 + ((rs_smooth - rs_min) / (rs_max - rs_min).replace(0, np.nan) - 0.5) * 20

        # RS-Momentum = 1-period ROC of RS-Ratio, normalised similarly
        rs_mom_raw = rs_ratio.diff(1)
        # Use smaller min_periods (10) so momentum is available early in the series
        mom_min = rs_mom_raw.rolling(period, min_periods=10).min()
        mom_max = rs_mom_raw.rolling(period, min_periods=10).max()
        rs_mom = 100 + ((rs_mom_raw - mom_min) / (mom_max - mom_min).replace(0, np.nan) - 0.5) * 20

        results[ticker] = {"rs_ratio": rs_ratio, "rs_mom": rs_mom}

    return results

def quadrant(x, y):
    if x >= 100 and y >= 100: return "Leading",   "#e67e22"
    if x >= 100 and y <  100: return "Weakening",  "#2e86c1"
    if x <  100 and y <  100: return "Lagging",    "#1d4e89"
    return "Improving", "#85c1e9"

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def fetch_prices(tickers: tuple, benchmark: str) -> pd.DataFrame:
    all_t = list(set(list(tickers) + [benchmark]))
    frames = {}
    for ticker in all_t:
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
               f"?interval=1wk&range=2y&corsDomain=finance.yahoo.com")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json", "Referer": "https://finance.yahoo.com",
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as r:
                data = json.loads(r.read())
            res = data["chart"]["result"][0]
            q   = res["indicators"]["quote"][0]
            df  = pd.DataFrame({
                "Date":  pd.to_datetime(res["timestamp"], unit="s").normalize(),
                "Close": q["close"],
            }).dropna().set_index("Date")
            frames[ticker] = df["Close"]
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔄 Relative Rotation Graph")
st.caption("Quadrants: **Leading** (strong & gaining) · **Weakening** (strong but losing momentum) · "
           "**Lagging** (weak & losing) · **Improving** (weak but gaining momentum)")

# Sidebar
st.sidebar.header("Settings")
benchmark = st.sidebar.text_input("Benchmark", value="SPY",
                                   help="Any Yahoo Finance ticker: SPY, QQQ, ^GSPC, EEM etc.").strip().upper()
tail_weeks = st.sidebar.slider("Tail length (weeks)", 1, 12, 4)
norm_period = st.sidebar.slider("Normalisation period (weeks)", 20, 52, 52)

st.sidebar.divider()
st.sidebar.subheader("ETF selection")
use_groups = st.sidebar.multiselect(
    "Include groups", list(ETF_UNIVERSE.keys()),
    default=["🇺🇸 US Sectors", "🌍 Europe", "🌏 Asia Pacific", "💵 Bonds", "🥇 Commodities"]
)

# Custom ticker add/remove
st.sidebar.divider()
st.sidebar.subheader("Custom tickers")
if "rrg_custom" not in st.session_state:
    st.session_state["rrg_custom"] = []

c1, c2 = st.sidebar.columns(2)
new_t = c1.text_input("Add", placeholder="e.g. TSLA").strip().upper()
if c1.button("Add", key="add_rrg", use_container_width=True) and new_t:
    if new_t not in st.session_state["rrg_custom"]:
        st.session_state["rrg_custom"].append(new_t)
        st.rerun()

rm_t = c2.selectbox("Remove", ["—"] + st.session_state["rrg_custom"], key="rm_rrg")
if c2.button("Remove", key="rem_rrg", use_container_width=True) and rm_t != "—":
    st.session_state["rrg_custom"].remove(rm_t)
    st.rerun()

if st.session_state["rrg_custom"]:
    st.sidebar.caption("Custom: " + "  |  ".join(st.session_state["rrg_custom"]))

refresh = st.sidebar.button("🔄 Refresh", type="primary", use_container_width=True)
if refresh:
    st.cache_data.clear()
    st.rerun()

# Build ticker list
group_tickers = {t: ALL_ETFS.get(t, t)
                 for g in use_groups
                 for t in ETF_UNIVERSE.get(g, {}).keys()
                 if t != benchmark}
custom_tickers = {t: t for t in st.session_state["rrg_custom"] if t != benchmark}
all_plot = {**group_tickers, **custom_tickers}

if not all_plot:
    st.info("Select at least one group or add a custom ticker.")
    st.stop()

# Fetch
with st.spinner("Fetching weekly price data…"):
    prices = fetch_prices(tuple(all_plot.keys()), benchmark)

if prices.empty or benchmark not in prices.columns:
    st.error(f"Could not fetch benchmark **{benchmark}**. Check the ticker.")
    st.stop()

# Compute RRG
rrg_data = compute_rrg(prices, benchmark, period=norm_period)

if not rrg_data:
    st.error("Not enough data to compute RRG.")
    st.stop()

# ── Animate toggle ────────────────────────────────────────────────────────────
animate = st.toggle("▶ Animate rotation", value=False,
                    help="Step through each week to see how ETFs rotated over time")

# For animation we need enough history: tail + animation steps
anim_weeks = st.slider("Animation history (weeks)", 4, 26, 12) if animate else tail_weeks

def build_frame(rrg_data, all_plot, tail_w, week_offset=0):
    """Build traces for a single animation frame. week_offset=0 is most recent."""
    traces = []
    shapes = []
    annotations = []

    # Quadrant backgrounds
    for (x0,x1,y0,y1,color,label) in [
        (100,130, 100,130, "rgba(230,126, 34,0.08)", "LEADING"),
        (100,130,  70,100, "rgba( 46,134,193,0.08)", "WEAKENING"),
        ( 70,100,  70,100, "rgba( 29, 78,137,0.08)", "LAGGING"),
        ( 70,100, 100,130, "rgba(133,193,233,0.08)", "IMPROVING"),
    ]:
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                           fillcolor=color, line_width=0, layer="below"))
        annotations.append(dict(
            x=(x0+x1)/2, y=(y0+y1)/2, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, color=color.replace("0.08","0.5")),
            xref="x", yref="y"
        ))

    for ticker, rd in rrg_data.items():
        label = all_plot.get(ticker, ticker)
        xs = rd["rs_ratio"].dropna()
        ys = rd["rs_mom"].dropna()

        # Shift by offset for animation
        end_idx   = len(xs) - week_offset
        start_idx = max(0, end_idx - tail_w - 1)
        if end_idx < 2:
            continue

        tail_x = xs.iloc[start_idx:end_idx].tolist()
        tail_y = ys.iloc[start_idx:end_idx].tolist()
        if len(tail_x) < 1:
            continue
        # Remove any NaN values
        valid = [(x,y) for x,y in zip(tail_x,tail_y)
                 if not (pd.isna(x) or pd.isna(y))]
        if not valid:
            continue
        tail_x, tail_y = zip(*valid)
        tail_x, tail_y = list(tail_x), list(tail_y)

        curr_x, curr_y = tail_x[-1], tail_y[-1]
        quad, color = quadrant(curr_x, curr_y)

        # Tail line
        for i in range(len(tail_x)-1):
            alpha = 0.15 + 0.65 * (i / max(len(tail_x)-2, 1))
            traces.append(go.Scatter(
                x=[tail_x[i], tail_x[i+1]],
                y=[tail_y[i], tail_y[i+1]],
                mode="lines",
                line=dict(color=color, width=1.5),
                opacity=alpha,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Dot
        traces.append(go.Scatter(
            x=[curr_x], y=[curr_y],
            mode="markers+text",
            marker=dict(size=10, color=color, line=dict(width=1.5, color="#fff")),
            text=[ticker],
            textposition="top center",
            textfont=dict(size=10, color=color),
            name=f"{ticker} — {label}",
            hovertemplate=(
                f"<b>{ticker}</b> ({label})<br>"
                f"Quadrant: {quad}<br>"
                f"RS-Ratio: {curr_x:.1f}<br>"
                f"RS-Momentum: {curr_y:.1f}<extra></extra>"
            ),
            showlegend=True,
        ))

    # Legend markers
    for lname, lcolor in [("Leading","#e67e22"),("Weakening","#2e86c1"),
                           ("Lagging","#1d4e89"),("Improving","#85c1e9")]:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=lcolor, symbol="square"),
            name=lname, showlegend=True,
        ))

    return traces, shapes, annotations

base_layout = dict(
    height=700,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="RS-Ratio → (Relative Strength vs benchmark)",
               range=[70,130], showgrid=True,
               gridcolor="rgba(128,128,128,0.1)", tickfont=dict(size=10)),
    yaxis=dict(title="RS-Momentum → (Trend of Relative Strength)",
               range=[70,130], showgrid=True,
               gridcolor="rgba(128,128,128,0.1)", tickfont=dict(size=10)),
    legend=dict(orientation="v", x=1.01, y=1,
                font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    hovermode="closest",
    margin=dict(l=60, r=180, t=60, b=60),
)

if not animate:
    # ── Static chart ──────────────────────────────────────────────────────────
    traces, shapes, annotations = build_frame(rrg_data, all_plot, tail_weeks, 0)
    fig = go.Figure(data=traces)
    fig.update_layout(**base_layout, shapes=shapes, annotations=annotations)
    st.plotly_chart(fig, use_container_width=True)

else:
    # ── Animated chart ────────────────────────────────────────────────────────
    # Find max available weeks — use minimum across all tickers to ensure all have data
    valid_lengths = [
        len(rd["rs_ratio"].dropna())
        for rd in rrg_data.values()
        if len(rd["rs_ratio"].dropna()) > tail_weeks + 2
    ]
    if not valid_lengths:
        st.warning("Not enough history for animation. Try a shorter tail length.")
        st.stop()

    min_len    = min(valid_lengths)
    max_offset = min(min_len - tail_weeks - 2, anim_weeks)
    max_offset = max(1, max_offset)

    # Get date index from first valid ticker
    sample_rd  = next(
        rd for rd in rrg_data.values()
        if len(rd["rs_ratio"].dropna()) >= max_offset + tail_weeks + 2
    )
    date_idx = sample_rd["rs_ratio"].dropna().index

    # Build all frames oldest → newest
    all_frames   = []
    frame_labels = []
    for offset in range(max_offset, -1, -1):
        traces, shapes, annotations = build_frame(rrg_data, all_plot, tail_weeks, offset)
        if not traces:
            continue
        pos = -(offset + 1)
        frame_date = str(date_idx[pos].date()) if abs(pos) <= len(date_idx) else f"Week -{offset}"
        all_frames.append(go.Frame(
            data=traces,
            name=frame_date,
            layout=go.Layout(
                shapes=shapes,
                annotations=annotations + [dict(
                    x=0.5, y=1.04, xref="paper", yref="paper",
                    text=f"<b>{frame_date}</b>",
                    showarrow=False, font=dict(size=14),
                )]
            )
        ))
        frame_labels.append(frame_date)

    if not all_frames:
        st.warning("Could not build animation frames. Try reducing tail length or animation history.")
        st.stop()

    # Initial frame (oldest)
    init_traces, init_shapes, init_ann = build_frame(rrg_data, all_plot, tail_weeks, max_offset)
    fig = go.Figure(
        data=init_traces,
        frames=all_frames,
        layout=go.Layout(
            **base_layout,
            shapes=init_shapes,
            annotations=init_ann,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶  Play",
                         method="animate",
                         args=[None, dict(
                             frame=dict(duration=600, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=300, easing="cubic-in-out"),
                         )]),
                    dict(label="⏸  Pause",
                         method="animate",
                         args=[[None], dict(
                             frame=dict(duration=0, redraw=False),
                             mode="immediate",
                             transition=dict(duration=0),
                         )]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue=dict(prefix="Week: ", font=dict(size=11)),
                pad=dict(t=50, b=10),
                steps=[dict(
                    method="animate",
                    args=[[f.name], dict(
                        frame=dict(duration=600, redraw=True),
                        mode="immediate",
                        transition=dict(duration=300),
                    )],
                    label=f.name,
                ) for f in all_frames],
            )],
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Quadrant summary table ────────────────────────────────────────────────────
st.divider()
quad_rows = []
for ticker, rd in rrg_data.items():
    xs = rd["rs_ratio"].dropna()
    ys = rd["rs_mom"].dropna()
    if len(xs) < 2:
        continue
    curr_x, curr_y = xs.iloc[-1], ys.iloc[-1]
    prev_x, prev_y = xs.iloc[-2], ys.iloc[-2]
    quad, color = quadrant(curr_x, curr_y)
    label = all_plot.get(ticker, ticker)
    quad_rows.append({
        "Ticker":       ticker,
        "Name":         label,
        "Quadrant":     quad,
        "RS-Ratio":     round(curr_x, 2),
        "RS-Momentum":  round(curr_y, 2),
        "Ratio Δ":      round(curr_x - prev_x, 2),
        "Momentum Δ":   round(curr_y - prev_y, 2),
    })

if quad_rows:
    order = {"Leading":0, "Weakening":1, "Improving":2, "Lagging":3}
    df_q  = pd.DataFrame(quad_rows).sort_values(
        ["Quadrant","RS-Ratio"], key=lambda c: c.map(order) if c.name=="Quadrant" else -c
    )
    cols = st.columns(4)
    for i, (quad_name, color_hex) in enumerate([
        ("Leading","#e67e22"),("Weakening","#2e86c1"),
        ("Improving","#85c1e9"),("Lagging","#1d4e89")
    ]):
        sub = df_q[df_q["Quadrant"]==quad_name][["Ticker","Name","RS-Ratio","RS-Momentum"]]
        cols[i].markdown(f"**{quad_name}** ({len(sub)})")
        if not sub.empty:
            cols[i].dataframe(sub, hide_index=True, use_container_width=True)
        else:
            cols[i].caption("None")
