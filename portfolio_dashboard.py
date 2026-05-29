"""
Kronos-powered investment dashboard for a CAD $21,000 portfolio.
Run with: streamlit run portfolio_dashboard.py
"""

import os
import sys
import json
import warnings
import datetime
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# ── repo root so `model/` is importable ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    KRONOS_AVAILABLE = True
except Exception:
    KRONOS_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────────────────
CAD_USD = 0.734          # approximate exchange rate (May 2026)
BUDGET_CAD = 21_000
BUDGET_USD = BUDGET_CAD * CAD_USD

PORTFOLIO = {
    "RDW":  {"name": "Redwire Corporation",    "sector": "Space Infrastructure",  "risk": "High"},
    "MNTS": {"name": "Momentus Inc.",           "sector": "Space Transportation",  "risk": "Very High"},
    "OKLO": {"name": "Oklo Inc.",              "sector": "Nuclear Energy",         "risk": "High"},
    "ADTN": {"name": "ADTRAN Holdings",        "sector": "Networking Equipment",  "risk": "Medium"},
    "ACHR": {"name": "Archer Aviation",        "sector": "eVTOL / Air Taxi",      "risk": "Very High"},
    "CLSK": {"name": "CleanSpark Inc.",          "sector": "Bitcoin Mining",        "risk": "Very High"},
    "QBTS": {"name": "D-Wave Quantum",         "sector": "Quantum Computing",     "risk": "Very High"},
    "RGTI": {"name": "Rigetti Computing",      "sector": "Quantum Computing",     "risk": "Very High"},
}

# Risk-based weight suggestion (higher risk → smaller weight)
RISK_WEIGHTS = {"Medium": 0.20, "High": 0.14, "Very High": 0.10}

# Research summaries — sourced from adversarially verified deep-research (May 29, 2026)
# 14 claims passed 3-vote verification against primary SEC filings, NIST, DOE, and IR press releases.
RESEARCH = {
    "RDW": {
        "ticker_note": "⚠️ Stock surged ~86% in 6 days to ~$23 (May 29, 2026). All analyst targets were set before this spike.",
        "thesis": (
            "Redwire is a space infrastructure company providing structural components, solar arrays, and in-space "
            "manufacturing for NASA and DoD. FY2025 revenue was $335.4M (actual). Analysts project $475.29M in FY2026 "
            "(+41.72%) and $572.86M in FY2027 (+20.53%), corroborated by company guidance of $450–$500M for 2026 "
            "reaffirmed at its May 8, 2026 earnings call. Integration of Edge Autonomy is a key growth lever."
        ),
        "catalysts": (
            "NASA $25M single-award IDIQ contract (Aug 2025) with a $2.5M initial InSPA task order for ISS drug "
            "development; DoD satellite awards; Edge Autonomy synergies; ISS in-space manufacturing expansion; "
            "growing US government space budget."
        ),
        "risks": (
            "Stock is trading significantly above analyst consensus — average price target is $14.44 (−37% downside "
            "from ~$23), with the highest individual target at $22.00. FY2025 net loss was $226.6M on $335.4M revenue. "
            "Heavy acquisition debt; government contract concentration; dilutive equity; stock spike is momentum-driven "
            "and may revert sharply."
        ),
        "analyst": (
            "10 analysts (S&P Global pool): 8 Strong Buy, 1 Hold, 1 Sell — consensus 'Buy'. Average price target $14.44 "
            "(−37% downside from post-spike price of ~$23). Pre-spike, targets ranged $11–$22. The stock's rally has "
            "dramatically outpaced fundamental re-ratings; caution warranted entering at current levels."
        ),
    },
    "MNTS": {
        "ticker_note": "Going concern doubt officially resolved in Q1 2026 10-Q (filed May 13, 2026).",
        "thesis": (
            "Momentus provides in-space transportation via its Vigoride orbital transfer vehicles. Q1 2026 marked a "
            "breakout quarter: service revenue hit $3.215M — a ~10x year-over-year increase from $0.322M in Q1 2025. "
            "Revenue came from hosted payload services ($1.618M) and engineering project services ($1.597M). Management "
            "formally cleared prior going-concern doubt after raising ~$16.7M in Q1 2026, plus $5M private placement and "
            "~$10.5M in Class A stock sales post-quarter."
        ),
        "catalysts": (
            "Revenue acceleration (~10x YoY); going-concern doubt resolved; cash position $23.5M as of March 31, 2026; "
            "new hosted payload and engineering services contracts; potential DoD rideshare missions."
        ),
        "risks": (
            "Absolute revenue still very small ($3.2M/quarter); Q1 2026 operating cash outflow $5.814M implies ~4 "
            "quarters of runway at current burn before needing fresh capital; ongoing dilution risk; CFIUS regulatory "
            "oversight history; mission execution risk."
        ),
        "analyst": (
            "Highly speculative micro-cap. No formal sell-side coverage found. Positive signal: Q1 2026 10-Q "
            "(primary SEC filing) removes going-concern language — a genuine milestone for a company that carried "
            "this warning through Q3 2025. Position size accordingly."
        ),
    },
    "OKLO": {
        "ticker_note": "Pre-revenue company. ~$11.6B market cap at ~$66/share (May 2026).",
        "thesis": (
            "Oklo is developing the Aurora compact fission microreactor, backed by Sam Altman. The company has zero "
            "trailing revenue and reported a net loss of $128.92M (EPS −$0.83). The investment case rests entirely on "
            "future regulatory approval and power-purchase agreement execution. The nuclear sector tailwind is real — "
            "DOE's NRIC DOME facility at Idaho National Laboratory opened April 8, 2026, and three competing "
            "microreactor experiments (eVinci, Kaleidos, Antares R1) are targeting tests 'as early as 2026,' "
            "validating the broader nuclear microreactor thesis."
        ),
        "catalysts": (
            "NRC construction permit and operating licence; first power-purchase agreements with AI/hyperscaler data "
            "centers; DOE loan guarantee programme; sector validation from Idaho National Lab DOME facility opening; "
            "bipartisan political support for nuclear energy."
        ),
        "risks": (
            "Zero revenue; $128.92M annual net loss; regulatory timeline is multi-year and uncertain; nuclear public "
            "opposition; technology construction risk; at ~$11.6B market cap the valuation price in enormous future "
            "success, leaving little margin of safety."
        ),
        "analyst": (
            "23 analyst coverage — consensus 'Buy'; average price target ~$88.89 (~34% upside from ~$66). "
            "The large analyst following for a pre-revenue company reflects the sector narrative premium. "
            "This is the highest-conviction speculative long in the portfolio from an analyst standpoint."
        ),
    },
    "ADTN": {
        "ticker_note": "Most stable name in the portfolio. Avg analyst target $19.50 (+16% upside from ~$16.76).",
        "thesis": (
            "ADTRAN Holdings manufactures fiber broadband access equipment for telecommunications carriers and cable "
            "operators globally. It is the only revenue-generating, near-profitable name in this portfolio. Revenue is "
            "driven by fibre-to-the-home (FTTH) deployments, and the US BEAD programme (Broadband Equity, Access and "
            "Deployment) is expected to accelerate rural fibre buildout through 2027–2028."
        ),
        "catalysts": (
            "BEAD programme ($42.5B US federal broadband funding); 25G PON technology upgrade cycle; European fibre "
            "expansion (strong Germany/UK exposure); average analyst price target $19.50 — 16.35% upside from $16.76 "
            "(range $18–$23 across 9 analysts)."
        ),
        "risks": (
            "Customer concentration in major US carriers; competition from Nokia and Calix; margin compression from "
            "supply-chain costs; European telecom capex budget uncertainty; revenue timing dependent on government "
            "BEAD funding disbursement schedule."
        ),
        "analyst": (
            "9 analysts: 6 Strong Buy, 1 Buy, 2 Hold — consensus 'Buy'. Average price target $19.50 (range $18–$23), "
            "implying ~16% upside. Best risk-adjusted position in the portfolio; recommended as the anchor holding "
            "with the largest weight."
        ),
    },
    "ACHR": {
        "ticker_note": "Best-funded eVTOL company. $1.96B cash at year-end 2025. Target: first passenger flights 2026.",
        "thesis": (
            "Archer Aviation is building the Midnight eVTOL aircraft for urban air mobility. It holds the strongest "
            "balance sheet in the eVTOL sector: $1,964.7M in cash and short-term investments at December 31, 2025 "
            "(up $1,130.2M year-over-year, driven by ~$1.8B in equity raises). Archer achieved 100% FAA acceptance "
            "of Means of Compliance for Midnight — first eVTOL company to do so — and has been selected for the "
            "White House eVTOL Integration Pilot Program (eIPP) in Florida, New York, and Texas."
        ),
        "catalysts": (
            "Targeting first passenger-carrying flights in 2026 (company guidance); eIPP piloted operations 'later "
            "in 2026'; FAA Stage 4 type certification progress; UAE air taxi pilot programme; United Airlines "
            "partnership; DoD Agility Prime contract."
        ),
        "risks": (
            "FY2025 net loss $618.2M; full-year operating cash burn $432.9M — at this rate, even $1.96B cash gives "
            "roughly 4–5 years of runway, but capital raises will be needed before profitability. FAA type cert "
            "not expected before 2027–2028 per independent analysts; Archer was only ~15% through Stage 4 of FAA "
            "certification in early 2026 vs Joby at ~70%."
        ),
        "analyst": (
            "8 Wall Street analysts: 5 Buy, 2 Hold, 1 Sell — 'Moderate Buy'. Average price target $11.83 "
            "(~80% upside from $6.56). High target $18, low $8. Strong cash position reduces near-term "
            "dilution risk but profitability is years away."
        ),
    },
    "CLSK": {
        "ticker_note": "Bitcoin mining — returns are highly correlated to BTC price, not operational metrics.",
        "thesis": (
            "CleanSpark is a US Bitcoin miner operating large-scale ASIC facilities across Georgia, Wyoming, and "
            "other states. It markets itself as a low-cost, clean-energy miner. Revenue and margins are almost "
            "entirely a function of BTC price × mining efficiency (hash rate) ÷ energy cost. The April 2024 "
            "Bitcoin halving reduced block rewards from 6.25 to 3.125 BTC, compressing margins industry-wide. "
            "CleanSpark has been expanding hash rate capacity aggressively through acquisitions."
        ),
        "catalysts": (
            "Bitcoin price appreciation; hash rate expansion at lower cost-per-EH than peers; institutional BTC "
            "demand (spot ETF flows); acquisition of distressed mining assets; next halving cycle positioning."
        ),
        "risks": (
            "Bitcoin price volatility is the dominant risk — a 50% BTC drawdown can make mining uneconomical; "
            "rising network difficulty compresses per-coin margins; energy cost spikes (electricity price risk); "
            "regulatory risk on proof-of-work mining; dilutive equity offerings to fund expansion; this is "
            "effectively a leveraged BTC proxy, not a diversifying asset."
        ),
        "analyst": (
            "Speculative / crypto-correlated. Outperforms sharply in BTC bull markets but drawdowns can exceed "
            "BTC's own decline (2x–3x leverage effect). Best sized as a small satellite position for BTC "
            "upside exposure with operational leverage."
        ),
    },
    "QBTS": {
        "ticker_note": "BREAKING (May 21, 2026): $100M CHIPS Act LOI from US Dept of Commerce — confirmed by NIST.",
        "thesis": (
            "D-Wave Quantum is the only commercially deployed quantum computing company, using quantum annealing "
            "(not gate-based). Its Leap cloud platform offers real-time QPU access for combinatorial optimisation "
            "problems. FY2025 revenue was $24.59M (+178% YoY from $8.83M in 2024), but TTM revenue has since "
            "declined to ~$12.44M — signalling lumpy revenue recognition. The May 21, 2026 CHIPS Act LOI for "
            "$100M in federal funding is the most important near-term catalyst (confirmed by NIST primary source)."
        ),
        "catalysts": (
            "CHIPS and Science Act LOI for $100M federal funding (announced May 21, 2026, NIST confirmed); "
            "enterprise customer expansion in logistics and optimisation; Advantage2 system capabilities; "
            "Leap cloud platform subscription growth."
        ),
        "risks": (
            "Market cap ~$10.45B on ~$12.44M TTM revenue — implied P/S ratio over 800x; FY2025 net loss "
            "$355M (TTM loss $368M, up 146.8% vs 2024); annealing has limited problem scope vs gate-based "
            "quantum; IBM/Google/RGTI gate-based competition may eventually displace annealing for large "
            "problem sets; LOI ≠ disbursed funds."
        ),
        "analyst": (
            "Speculative. The $100M CHIPS Act LOI is a major government validation, but the valuation (800x P/S) "
            "already prices in years of growth. Revenue trajectory is volatile and the LOI timeline for actual "
            "fund disbursement is 3 years. Position sizing should be small given extreme valuation risk."
        ),
    },
    "RGTI": {
        "ticker_note": "BREAKING (May 21, 2026): $100M CHIPS Act LOI from US Dept of Commerce — confirmed by NIST.",
        "thesis": (
            "Rigetti Computing builds superconducting gate-based quantum processors and provides QPU cloud access. "
            "Its 84-qubit Ankaa-3 system is its most advanced system to date. Rigetti received a LOI for up to "
            "$100M under the CHIPS and Science Act (May 21, 2026, NIST confirmed), providing 3-year R&D funding "
            "for superconducting quantum computing — a material de-risking of its near-term cash needs."
        ),
        "catalysts": (
            "CHIPS Act LOI for up to $100M (3-year R&D funding, May 21, 2026, SEC 8-K filed); 84-qubit Ankaa-3 "
            "system performance improvements; error correction milestones; DARPA/DOE grants; enterprise cloud "
            "QPU customer wins."
        ),
        "risks": (
            "Pre-profitable with significant cash burn; IBM, Google, and IonQ outpace on qubit count and error "
            "rates; superconducting architecture requires near-absolute-zero cooling (operational complexity); "
            "LOI ≠ disbursed funds; 3-year government programme with milestone conditions."
        ),
        "analyst": (
            "Highly speculative. The CHIPS Act LOI is transformative for de-risking near-term finances and "
            "providing government credibility. Both QBTS and RGTI received identical $100M LOIs — worth noting "
            "the government is funding the space broadly, not picking a single winner between annealing and "
            "gate-based approaches."
        ),
    },
}

RISK_COLOR = {
    "Medium": "#2ecc71",
    "High": "#f39c12",
    "Very High": "#e74c3c",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kronos Portfolio Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── base overrides ─────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main           { background:#0e1117 !important; }
    [data-testid="stSidebar"]                      { background:#1e1e2e !important; }
    [data-testid="stSidebarUserContent"] label,
    [data-testid="stSidebarUserContent"] p         { color:#e2e8f0 !important; }

    /* ── typography ─────────────────────────────────────────────── */
    h1, h2, h3                                     { color:#a78bfa !important; }
    p, li, label, .stCaption                       { color:#cbd5e1 !important; }

    /* ── tabs ───────────────────────────────────────────────────── */
    [data-baseweb="tab-list"]                      { background:#1e1e2e !important; border-radius:8px; }
    [data-baseweb="tab"]                           { font-size:15px; color:#94a3b8 !important; }
    [aria-selected="true"][data-baseweb="tab"]     { color:#a78bfa !important; border-bottom:2px solid #a78bfa !important; }

    /* ── cards / containers ─────────────────────────────────────── */
    [data-testid="stMetric"]                       { background:#1e1e2e; border-radius:8px; padding:12px; }
    [data-testid="stMetricValue"]                  { color:#e2e8f0 !important; }
    [data-testid="stMetricDelta"]                  { font-size:13px; }
    div[data-testid="stDataFrame"]                 { background:#1e1e2e !important; }

    /* ── inputs ─────────────────────────────────────────────────── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stNumberInput"] > div            { background:#1e1e2e !important; color:#e2e8f0 !important; border-color:#334155 !important; }
    [data-testid="stButton"] > button              { background:#a78bfa !important; color:#0e1117 !important; font-weight:600; border:none; }

    /* ── alerts ─────────────────────────────────────────────────── */
    .stInfo    { background:#1e3a5f !important; color:#93c5fd !important; border-color:#3b82f6 !important; }
    .stSuccess { background:#14532d !important; color:#86efac !important; border-color:#22c55e !important; }
    .stWarning { background:#431407 !important; color:#fdba74 !important; border-color:#f97316 !important; }
    .stError   { background:#450a0a !important; color:#fca5a5 !important; border-color:#ef4444 !important; }

    /* ── risk labels ────────────────────────────────────────────── */
    .risk-high    { color:#f87171; font-weight:bold; }
    .risk-medium  { color:#fbbf24; font-weight:bold; }
    .risk-low     { color:#4ade80; font-weight:bold; }

    /* ── divider ────────────────────────────────────────────────── */
    hr { border-color:#334155 !important; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(str(Path(__file__).parent / "figures" / "logo.png"), width=80)
    st.title("Kronos Dashboard")
    st.markdown("---")

    st.subheader("Portfolio Settings")
    budget_cad = st.number_input("Budget (CAD $)", value=BUDGET_CAD, step=500)
    cad_usd_rate = st.number_input("CAD/USD Rate", value=CAD_USD, step=0.001, format="%.3f")
    budget_usd = budget_cad * cad_usd_rate
    st.caption(f"≈ USD ${budget_usd:,.0f}")

    st.markdown("---")
    st.subheader("Kronos Model")
    if KRONOS_AVAILABLE:
        model_choice = st.selectbox("Model", ["kronos-mini (4.1M)", "kronos-small (24.7M)", "kronos-base (102.3M)"])
        device_choice = st.selectbox("Device", ["cpu", "mps", "cuda"])
        load_model_btn = st.button("Load Kronos Model", type="primary", use_container_width=True)
    else:
        st.warning("Kronos model library not installed.\nRun: `pip install -r requirements.txt`")
        load_model_btn = False

    if "kronos_loaded" not in st.session_state:
        st.session_state.kronos_loaded = False
        st.session_state.predictor = None

    if KRONOS_AVAILABLE and load_model_btn:
        with st.spinner("Downloading & loading model…"):
            try:
                model_key = model_choice.split()[0].lower()
                MODEL_MAP = {
                    "kronos-mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k",  2048),
                    "kronos-small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
                    "kronos-base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
                }
                model_id, tok_id, ctx = MODEL_MAP[model_key]
                tok = KronosTokenizer.from_pretrained(tok_id)
                mdl = Kronos.from_pretrained(model_id)
                st.session_state.predictor = KronosPredictor(mdl, tok, device=device_choice, max_context=ctx)
                st.session_state.kronos_loaded = True
                st.session_state.model_key = model_key
                st.success(f"{model_choice} loaded on {device_choice}")
            except Exception as e:
                st.error(f"Load failed: {e}")

    if st.session_state.kronos_loaded:
        st.success("Kronos ready")

    st.markdown("---")
    st.subheader("Prediction Settings")
    lookback = st.slider("Lookback (trading days)", 100, 400, 200, 20)
    pred_len = st.slider("Forecast horizon (trading days)", 10, 60, 30, 5)
    temperature = st.slider("Temperature", 0.5, 2.0, 1.0, 0.1)
    top_p = st.slider("Top-p", 0.5, 1.0, 0.9, 0.05)

# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        # Flatten any MultiIndex columns
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = ["_".join(c).strip("_").lower() for c in hist.columns]
        else:
            hist.columns = [c.lower() for c in hist.columns]
        # Normalise date column name
        for candidate in ("date", "datetime", "timestamp"):
            if candidate in hist.columns:
                hist = hist.rename(columns={candidate: "timestamps"})
                break
        hist["timestamps"] = pd.to_datetime(hist["timestamps"]).dt.tz_localize(None)
        hist = hist.sort_values("timestamps").reset_index(drop=True)
        cols = [c for c in ["timestamps", "open", "high", "low", "close", "volume"] if c in hist.columns]
        return hist[cols]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def compute_allocation(budget_usd: float) -> pd.DataFrame:
    total_w = sum(RISK_WEIGHTS.get(v["risk"], 0.06) for v in PORTFOLIO.values())
    rows = []
    for ticker, meta in PORTFOLIO.items():
        w = RISK_WEIGHTS.get(meta["risk"], 0.06) / total_w
        usd = budget_usd * w
        cad = usd / cad_usd_rate
        rows.append({
            "Ticker": ticker,
            "Company": meta["name"],
            "Sector": meta["sector"],
            "Risk": meta["risk"],
            "Allocation %": round(w * 100, 1),
            "USD ($)": round(usd, 0),
            "CAD ($)": round(cad, 0),
        })
    return pd.DataFrame(rows)

def candlestick_chart(df: pd.DataFrame, ticker: str, pred_df: pd.DataFrame = None, future_dates=None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    # historical candles
    display = df.tail(lookback + 40)
    fig.add_trace(go.Candlestick(
        x=display["timestamps"], open=display["open"], high=display["high"],
        low=display["low"], close=display["close"],
        name="Historical", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # MA lines
    for ma, color in [(20, "#fbbf24"), (50, "#60a5fa")]:
        fig.add_trace(go.Scatter(
            x=display["timestamps"], y=display["close"].rolling(ma).mean(),
            name=f"MA{ma}", line=dict(color=color, width=1.2), opacity=0.8,
        ), row=1, col=1)

    # Kronos prediction
    if pred_df is not None and future_dates is not None and len(pred_df) > 0:
        fig.add_trace(go.Candlestick(
            x=future_dates, open=pred_df["open"], high=pred_df["high"],
            low=pred_df["low"], close=pred_df["close"],
            name="Kronos Forecast",
            increasing_line_color="#a78bfa", decreasing_line_color="#f87171",
        ), row=1, col=1)
        fig.add_vline(x=display["timestamps"].iloc[-1], line_dash="dash",
                      line_color="white", opacity=0.5, row=1, col=1)

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(display["close"], display["open"])]
    fig.add_trace(go.Bar(
        x=display["timestamps"], y=display["volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} — Price & Volume",
        template="plotly_dark", height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

FORECASTS_DIR = Path(__file__).parent / "forecasts"
FORECASTS_DIR.mkdir(exist_ok=True)

def save_forecast(ticker: str, name: str, last_close: float, last_date,
                  pred_df: pd.DataFrame, future_dates: list, model_key: str):
    """Append a single-ticker forecast into today's JSON snapshot."""
    fname = FORECASTS_DIR / f"{datetime.date.today().isoformat()}_{model_key}.json"
    if fname.exists():
        snap = json.loads(fname.read_text())
    else:
        snap = {
            "run_date":     datetime.date.today().isoformat(),
            "run_datetime": datetime.datetime.now().isoformat(timespec="seconds"),
            "model":        model_key,
            "lookback":     lookback,
            "pred_len":     pred_len,
            "tickers":      {},
        }
    rows = []
    for i, fdate in enumerate(future_dates):
        row = pred_df.iloc[i]
        rows.append({
            "date":   fdate.strftime("%Y-%m-%d") if hasattr(fdate, "strftime") else str(fdate)[:10],
            "open":   round(float(row["open"]),  4),
            "high":   round(float(row["high"]),  4),
            "low":    round(float(row["low"]),   4),
            "close":  round(float(row["close"]), 4),
            "volume": round(float(row.get("volume", 0)), 0),
        })
    snap["tickers"][ticker] = {
        "name":       name,
        "last_close": round(float(last_close), 4),
        "last_date":  last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)[:10],
        "forecast":   rows,
    }
    fname.write_text(json.dumps(snap, indent=2))

def run_kronos_prediction(df: pd.DataFrame, ticker: str):
    if not st.session_state.kronos_loaded or st.session_state.predictor is None:
        return None, None

    if len(df) < lookback + 5:
        return None, None

    predictor = st.session_state.predictor
    x_df = df.tail(lookback)[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_ts = df.tail(lookback)["timestamps"].reset_index(drop=True)
    if isinstance(x_ts, pd.DatetimeIndex):
        x_ts = pd.Series(x_ts)

    last_date = x_ts.iloc[-1]
    future_dates = []
    cur = last_date + datetime.timedelta(days=1)
    while len(future_dates) < pred_len:
        if cur.weekday() < 5:
            future_dates.append(cur)
        cur += datetime.timedelta(days=1)
    y_ts = pd.Series(future_dates)

    try:
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=temperature, top_p=top_p, sample_count=1,
        )
        # Auto-save to forecasts/
        model_key = st.session_state.get("model_key", "kronos-mini")
        save_forecast(
            ticker=ticker,
            name=PORTFOLIO[ticker]["name"],
            last_close=float(df["close"].iloc[-1]),
            last_date=last_date,
            pred_df=pred_df,
            future_dates=future_dates,
            model_key=model_key,
        )
        return pred_df, future_dates
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def format_large(n):
    if n is None:
        return "N/A"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.1f}M"
    return f"${n:,.0f}"

# ── main app ──────────────────────────────────────────────────────────────────
st.title("Kronos Investment Dashboard")
st.caption(f"Portfolio: CAD ${budget_cad:,}  ·  AI-powered by Kronos Foundation Model  ·  {datetime.date.today()}")

tabs = st.tabs(["Portfolio Overview", "Stock Analysis", "Kronos Forecast", "Forecast vs Actual", "Investment Research", "Allocation Strategy"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio Overview
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Portfolio Overview")

    alloc_df = compute_allocation(budget_usd)

    with st.spinner("Fetching live prices…"):
        prices = {}
        for ticker in PORTFOLIO:
            info = fetch_info(ticker)
            p = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
            prices[ticker] = p

    # Compute shares possible and market value
    rows_overview = []
    for _, row in alloc_df.iterrows():
        p = prices.get(row["Ticker"])
        shares = int(row["USD ($)"] / p) if p else 0
        mkt_val = shares * p if p else 0
        rows_overview.append({
            "Ticker": row["Ticker"],
            "Company": row["Company"],
            "Sector": row["Sector"],
            "Risk": row["Risk"],
            "Alloc %": f'{row["Allocation %"]}%',
            "Budget USD": f'${row["USD ($)"]:,.0f}',
            "Price": f"${p:.2f}" if p else "N/A",
            "Shares": shares,
            "Market Value": f"${mkt_val:,.0f}",
        })

    overview_df = pd.DataFrame(rows_overview)

    # Summary metrics
    total_invested = sum(
        int(alloc_df.loc[alloc_df["Ticker"] == t, "USD ($)"].values[0] /
            prices[t]) * prices[t]
        for t in PORTFOLIO if prices.get(t)
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Budget", f"CAD ${budget_cad:,}")
    c2.metric("≈ USD Value", f"${budget_usd:,.0f}")
    c3.metric("Stocks", len(PORTFOLIO))
    c4.metric("Sectors", len({m["sector"] for m in PORTFOLIO.values()}))

    st.markdown("---")

    col_tbl, col_pie = st.columns([3, 2])
    with col_tbl:
        st.markdown("**Suggested Position Sizes**")
        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk": st.column_config.TextColumn("Risk Level"),
                "Market Value": st.column_config.TextColumn("Est. Value"),
            },
        )
        st.caption("Shares are estimated floor values based on suggested allocation.")

    with col_pie:
        pie_colors = [RISK_COLOR.get(r, "#888") for r in alloc_df["Risk"]]
        fig_pie = go.Figure(go.Pie(
            labels=alloc_df["Ticker"], values=alloc_df["USD ($)"],
            marker_colors=pie_colors, hole=0.4,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(template="plotly_dark", height=380,
                              title="Allocation by Ticker")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Sector breakdown
    sector_df = alloc_df.groupby("Sector")["USD ($)"].sum().reset_index()
    fig_sector = go.Figure(go.Bar(
        x=sector_df["Sector"], y=sector_df["USD ($)"],
        marker_color="#60a5fa",
    ))
    fig_sector.update_layout(template="plotly_dark", showlegend=False, height=300,
                             title="Allocation by Sector",
                             yaxis_title="Budget (USD)")
    st.plotly_chart(fig_sector, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stock Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Individual Stock Analysis")

    selected = st.selectbox("Select Ticker", list(PORTFOLIO.keys()),
                             format_func=lambda t: f"{t} — {PORTFOLIO[t]['name']}")

    info = fetch_info(selected)
    df_ohlcv = fetch_ohlcv(selected)

    # Key metrics row
    price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", f"${price:.2f}" if price else "N/A")
    m2.metric("Market Cap", format_large(info.get("marketCap")))
    m3.metric("52W High", f'${info.get("fiftyTwoWeekHigh","N/A")}')
    m4.metric("52W Low", f'${info.get("fiftyTwoWeekLow","N/A")}')
    m5.metric("P/E Ratio", round(info.get("trailingPE", 0), 1) or "N/A")
    m6.metric("Beta", round(info.get("beta", 0), 2) or "N/A")

    st.caption(
        f'**{info.get("longName","—")}** · '
        f'{info.get("sector","—")} · '
        f'{info.get("country","—")} · '
        f'Exchange: {info.get("exchange","—")}'
    )

    if not df_ohlcv.empty:
        # Technical indicators
        close = df_ohlcv["close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df_ohlcv["rsi"] = 100 - (100 / (1 + rs))
        df_ohlcv["bb_mid"] = close.rolling(20).mean()
        df_ohlcv["bb_up"] = df_ohlcv["bb_mid"] + 2 * close.rolling(20).std()
        df_ohlcv["bb_lo"] = df_ohlcv["bb_mid"] - 2 * close.rolling(20).std()

        fig_main = candlestick_chart(df_ohlcv, selected)
        st.plotly_chart(fig_main, use_container_width=True)

        # RSI chart
        display = df_ohlcv.tail(lookback + 40)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=display["timestamps"], y=display["rsi"],
                                     name="RSI(14)", line=dict(color="#a78bfa", width=1.5)))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="#e74c3c", opacity=0.6)
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="#2ecc71", opacity=0.6)
        fig_rsi.update_layout(template="plotly_dark", height=200, title="RSI (14)",
                              yaxis=dict(range=[0, 100]),
                              margin=dict(l=40, r=20, t=40, b=20))
        st.plotly_chart(fig_rsi, use_container_width=True)

        # Volume & returns distribution
        cr, vr = st.columns(2)
        with cr:
            df_ohlcv["daily_ret"] = df_ohlcv["close"].pct_change() * 100
            ret_vals = df_ohlcv["daily_ret"].dropna()
            fig_ret = go.Figure(go.Histogram(x=ret_vals, nbinsx=60,
                                             marker_color="#60a5fa"))
            fig_ret.update_layout(template="plotly_dark", height=280, showlegend=False,
                                  title="Daily Return Distribution (%)",
                                  xaxis_title="%",
                                  margin=dict(l=40, r=20, t=40, b=20))
            st.plotly_chart(fig_ret, use_container_width=True)

        with vr:
            ann_vol = df_ohlcv["daily_ret"].std() * np.sqrt(252)
            cum_ret = (1 + df_ohlcv["daily_ret"] / 100).cumprod() - 1
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=df_ohlcv["timestamps"], y=cum_ret * 100,
                name="Cumulative Return", fill="tozeroy",
                line=dict(color="#a78bfa", width=1.5),
            ))
            fig_cum.update_layout(template="plotly_dark", height=280,
                                  title=f"Cumulative Return — Ann. Vol: {ann_vol:.1f}%",
                                  yaxis_title="%",
                                  margin=dict(l=40, r=20, t=40, b=20))
            st.plotly_chart(fig_cum, use_container_width=True)
    else:
        st.warning(f"No OHLCV data available for {selected}. Ticker may be incorrect or delisted.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Kronos Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Kronos AI Price Forecast")

    if not KRONOS_AVAILABLE:
        st.error("Kronos library not found. Install dependencies with:\n```\npip install -r requirements.txt\n```")
    elif not st.session_state.kronos_loaded:
        st.info("Load the Kronos model from the sidebar to enable AI forecasts.")
    else:
        fc_ticker = st.selectbox(
            "Select ticker to forecast",
            list(PORTFOLIO.keys()),
            format_func=lambda t: f"{t} — {PORTFOLIO[t]['name']}",
            key="fc_select",
        )

        df_fc = fetch_ohlcv(fc_ticker)

        if df_fc.empty:
            st.warning(f"No data for {fc_ticker}.")
        else:
            st.info(
                f"Using last **{lookback}** trading days as context → "
                f"forecasting next **{pred_len}** trading days "
                f"(T={temperature}, top_p={top_p})."
            )
            run_btn = st.button("Run Kronos Forecast", type="primary")

            if run_btn:
                with st.spinner(f"Running Kronos on {fc_ticker}…"):
                    pred_df, future_dates = run_kronos_prediction(df_fc, fc_ticker)

                if pred_df is not None:
                    st.success(f"Forecast complete — {pred_len} trading days predicted.")

                    fig_fc = candlestick_chart(df_fc, fc_ticker, pred_df, future_dates)
                    st.plotly_chart(fig_fc, use_container_width=True)

                    last_close = float(df_fc["close"].iloc[-1])
                    final_pred = float(pred_df["close"].iloc[-1])
                    change_pct = (final_pred / last_close - 1) * 100

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Last Close", f"${last_close:.2f}")
                    c2.metric(f"Predicted Close (T+{pred_len}d)", f"${final_pred:.2f}",
                              delta=f"{change_pct:+.2f}%")
                    c3.metric("Predicted High (max)", f"${float(pred_df['high'].max()):.2f}")

                    with st.expander("Raw forecast data"):
                        pred_df_display = pred_df.copy()
                        pred_df_display.index = [d.strftime("%Y-%m-%d") for d in future_dates]
                        st.dataframe(pred_df_display.round(3), use_container_width=True)
                else:
                    st.warning("Prediction returned no results. Check data length and model status.")

    # Batch overview (show all tickers)
    st.markdown("---")
    st.subheader("All-Ticker Forecast Summary")
    st.caption("Run individual forecasts above; table below shows last-close prices for reference.")

    summary_rows = []
    for t, meta in PORTFOLIO.items():
        p = prices.get(t, None)
        summary_rows.append({
            "Ticker": t,
            "Company": meta["name"],
            "Last Close (USD)": f"${p:.2f}" if p else "N/A",
            "Risk": meta["risk"],
            "Sector": meta["sector"],
            "Kronos Forecast": "Load model & run above",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Forecast vs Actual
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Forecast vs Actual — Live Market Comparison")
    st.caption("Select a saved forecast snapshot. Live prices are fetched from yfinance and overlaid against the Kronos predictions.")

    # ── Load available snapshots ───────────────────────────────────────────────
    snap_files = sorted(glob(str(FORECASTS_DIR / "*.json")), reverse=True)
    if not snap_files:
        st.info("No forecast snapshots found. Run forecasts in the Kronos Forecast tab first.")
    else:
        snap_labels = [Path(f).stem for f in snap_files]
        chosen_label = st.selectbox("Select forecast snapshot", snap_labels)
        chosen_file  = snap_files[snap_labels.index(chosen_label)]

        with open(chosen_file) as f:
            snap = json.load(f)

        run_date  = snap["run_date"]
        model_lbl = snap.get("model", "kronos")
        n_tickers = len(snap["tickers"])

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Forecast Date", run_date)
        col_b.metric("Model", model_lbl)
        col_c.metric("Tickers in snapshot", n_tickers)

        st.markdown("---")

        # ── Per-ticker comparison ──────────────────────────────────────────────
        fva_ticker = st.selectbox(
            "Select ticker to compare",
            list(snap["tickers"].keys()),
            format_func=lambda t: f"{t} — {snap['tickers'][t]['name']}",
            key="fva_sel",
        )

        tdata = snap["tickers"][fva_ticker]
        last_close  = tdata["last_close"]
        last_date   = pd.Timestamp(tdata["last_date"])
        pred_rows   = tdata["forecast"]
        pred_dates  = [pd.Timestamp(r["date"]) for r in pred_rows]
        pred_closes = [r["close"] for r in pred_rows]
        pred_highs  = [r["high"]  for r in pred_rows]
        pred_lows   = [r["low"]   for r in pred_rows]

        # Fetch actual prices since forecast date
        @st.cache_data(ttl=300, show_spinner=False)
        def fetch_actual(ticker, start_date):
            hist = yf.Ticker(ticker).history(start=start_date, auto_adjust=True)
            if hist.empty:
                return pd.DataFrame()
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            for col in ("date", "datetime"):
                if col in hist.columns:
                    hist = hist.rename(columns={col: "timestamps"})
                    break
            hist["timestamps"] = pd.to_datetime(hist["timestamps"]).dt.tz_localize(None)
            return hist[["timestamps","close"]].sort_values("timestamps").reset_index(drop=True)

        with st.spinner("Fetching live market data…"):
            actual_df = fetch_actual(fva_ticker, run_date)

        # ── Build comparison chart ─────────────────────────────────────────────
        fig_fva = go.Figure()

        # Kronos predicted close line
        fig_fva.add_trace(go.Scatter(
            x=pred_dates, y=pred_closes,
            name="Kronos Predicted Close",
            line=dict(color="#a78bfa", width=2.5, dash="dash"),
            mode="lines",
        ))

        # Predicted high/low band
        fig_fva.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1],
            y=pred_highs + pred_lows[::-1],
            fill="toself", fillcolor="rgba(167,139,250,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Predicted High/Low Band",
            hoverinfo="skip",
        ))

        # Actual close line
        if not actual_df.empty:
            fig_fva.add_trace(go.Scatter(
                x=actual_df["timestamps"], y=actual_df["close"],
                name="Actual Close (Live)",
                line=dict(color="#34d399", width=2.5),
                mode="lines+markers",
                marker=dict(size=5),
            ))

        # Last-close reference line
        fig_fva.add_hline(
            y=last_close, line_dash="dot", line_color="#f59e0b",
            annotation_text=f"Forecast base: ${last_close:.2f}",
            annotation_position="top left",
        )

        fig_fva.update_layout(
            title=f"{fva_ticker} — Kronos Forecast vs Live Market  (base date: {run_date})",
            template="plotly_dark",
            height=500,
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=40, r=20, t=60, b=40),
        )
        st.plotly_chart(fig_fva, use_container_width=True)

        # ── Accuracy metrics ───────────────────────────────────────────────────
        st.markdown("### Accuracy Metrics")

        if actual_df.empty:
            st.warning("No actual market data yet — forecast period may not have started.")
        else:
            # Match actual dates to predicted dates
            pred_series = pd.Series(pred_closes, index=pred_dates)
            act_series  = actual_df.set_index("timestamps")["close"]

            # Only compare days that exist in both
            common_dates = pred_series.index.intersection(act_series.index)
            days_elapsed = len(common_dates)
            days_remaining = len(pred_dates) - days_elapsed

            if days_elapsed == 0:
                st.info("Market hasn't opened yet for the first forecast day.")
            else:
                p_vals = pred_series.loc[common_dates].values
                a_vals = act_series.loc[common_dates].values

                mae  = float(np.mean(np.abs(a_vals - p_vals)))
                rmse = float(np.sqrt(np.mean((a_vals - p_vals)**2)))
                # Directional accuracy: did predicted direction vs base match actual direction vs base?
                pred_dirs = np.sign(p_vals - last_close)
                act_dirs  = np.sign(a_vals  - last_close)
                dir_acc   = float(np.mean(pred_dirs == act_dirs)) * 100

                latest_actual    = float(act_series.iloc[-1])
                latest_pred_date = common_dates[-1]
                latest_pred      = float(pred_series.loc[latest_pred_date])
                delta_pct        = (latest_actual / latest_pred - 1) * 100
                final_pred       = pred_closes[-1]
                final_pred_date  = pred_dates[-1]

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Days tracked", f"{days_elapsed} / {len(pred_dates)}")
                m2.metric("Days remaining", days_remaining)
                m3.metric("MAE", f"${mae:.3f}")
                m4.metric("RMSE", f"${rmse:.3f}")
                m5.metric("Directional Accuracy", f"{dir_acc:.0f}%")

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Forecast Base Price", f"${last_close:.2f}")
                c2.metric("Latest Actual", f"${latest_actual:.2f}",
                          delta=f"{(latest_actual/last_close-1)*100:+.1f}% vs base")
                c3.metric("Kronos Pred (same day)", f"${latest_pred:.2f}",
                          delta=f"{delta_pct:+.1f}% delta to actual")
                c4.metric(f"Final Pred (T+{len(pred_dates)}d)", f"${final_pred:.2f}",
                          delta=f"by {final_pred_date.strftime('%b %d')}")

                # Day-by-day table
                with st.expander("Day-by-day comparison table"):
                    rows = []
                    for d in common_dates:
                        pa = float(pred_series.loc[d])
                        aa = float(act_series.loc[d])
                        rows.append({
                            "Date": d.date(),
                            "Predicted Close": f"${pa:.2f}",
                            "Actual Close":    f"${aa:.2f}",
                            "Delta ($)":       f"${aa-pa:+.2f}",
                            "Delta (%)":       f"{(aa/pa-1)*100:+.1f}%",
                            "Direction ✓/✗":   "✓" if np.sign(pa-last_close)==np.sign(aa-last_close) else "✗",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Portfolio-wide snapshot summary ───────────────────────────────────
        st.markdown("---")
        st.markdown("### All-Ticker Snapshot Summary")
        summary_rows2 = []
        for t, td in snap["tickers"].items():
            fc_final = td["forecast"][-1]["close"]
            fc_high  = max(r["high"]  for r in td["forecast"])
            fc_low   = min(r["low"]   for r in td["forecast"])
            chg      = (fc_final / td["last_close"] - 1) * 100
            # Actual current price
            curr_p = prices.get(t)
            act_chg = (curr_p / td["last_close"] - 1) * 100 if curr_p else None
            summary_rows2.append({
                "Ticker":          t,
                "Company":         td["name"],
                "Base Price":      f"${td['last_close']:.2f}",
                "Pred T+30":       f"${fc_final:.2f}",
                "Pred Δ":          f"{'▲' if chg>=0 else '▼'}{abs(chg):.1f}%",
                "Pred High":       f"${fc_high:.2f}",
                "Pred Low":        f"${fc_low:.2f}",
                "Live Price":      f"${curr_p:.2f}" if curr_p else "—",
                "Live vs Base":    f"{'▲' if act_chg>=0 else '▼'}{abs(act_chg):.1f}%" if act_chg is not None else "—",
            })
        st.dataframe(pd.DataFrame(summary_rows2), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Investment Research
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Investment Research Summary")
    st.caption(
        "Data sourced from adversarially verified deep-research (May 29, 2026). "
        "14 claims passed 3-vote verification against primary SEC filings, NIST, DOE, and IR press releases."
    )

    research_ticker = st.selectbox(
        "Select company",
        list(PORTFOLIO.keys()),
        format_func=lambda t: f"{t} — {PORTFOLIO[t]['name']}",
        key="res_select",
    )

    meta = PORTFOLIO[research_ticker]
    res = RESEARCH[research_ticker]
    info_r = fetch_info(research_ticker)

    risk_color_map = {"Medium": "green", "High": "orange", "Very High": "red"}
    rc = risk_color_map.get(meta["risk"], "gray")

    # Show ticker note prominently if present
    if res.get("ticker_note"):
        st.info(res["ticker_note"])

    st.markdown(f"""
<div style='background:#1e1e2e;border-radius:10px;padding:20px;margin-bottom:16px;'>
    <h3 style='margin:0'>{research_ticker} — {meta["name"]}</h3>
    <p style='color:#94a3b8;margin:4px 0'>Sector: <b>{meta["sector"]}</b> &nbsp;|&nbsp;
    Risk: <b style='color:{rc}'>{meta["risk"]}</b></p>
    <p style='color:#94a3b8;margin:0'>Exchange: {info_r.get("exchange","—")} &nbsp;|&nbsp;
    Country: {info_r.get("country","—")}</p>
</div>
""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Investment Thesis**")
        st.info(res["thesis"])
        st.markdown("**Key Catalysts**")
        st.success(res["catalysts"])

    with cb:
        st.markdown("**Key Risks**")
        st.error(res["risks"])
        st.markdown("**Analyst Sentiment**")
        st.warning(res["analyst"])

    # Financials snapshot
    st.markdown("---")
    st.markdown("**Financial Snapshot (from yfinance)**")
    fin_cols = {
        "Market Cap": format_large(info_r.get("marketCap")),
        "Revenue (TTM)": format_large(info_r.get("totalRevenue")),
        "Gross Margin": f'{info_r.get("grossMargins",0)*100:.1f}%' if info_r.get("grossMargins") else "N/A",
        "Cash & Equiv.": format_large(info_r.get("totalCash")),
        "Total Debt": format_large(info_r.get("totalDebt")),
        "Employees": f'{info_r.get("fullTimeEmployees","N/A"):,}' if isinstance(info_r.get("fullTimeEmployees"), int) else "N/A",
    }
    fin_c = st.columns(len(fin_cols))
    for col, (label, val) in zip(fin_c, fin_cols.items()):
        col.metric(label, val)

    # Long description
    desc = info_r.get("longBusinessSummary", "")
    if desc:
        with st.expander("Business Description"):
            st.write(desc)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Allocation Strategy
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Allocation Strategy")

    alloc_df2 = compute_allocation(budget_usd)

    st.markdown("""
**Allocation Methodology**

Given this portfolio contains mostly high-risk, pre-revenue or early-stage companies,
a **risk-weighted allocation** is applied:
- `Medium` risk → 20% weight
- `High` risk → 14% weight
- `Very High` risk → 10% weight
- `Very High` (MNTS, ACHR, CLSK, QBTS, RGTI) → 10% weight each

This skews toward the more stable anchor position (ADTRAN / ADTN) while maintaining
meaningful exposure to the speculative names. Adjust weights to your own conviction.
""")

    # Adjustable weights
    st.markdown("---")
    st.subheader("Customize Weights")
    custom_weights = {}
    cols = st.columns(len(PORTFOLIO))
    for col, (ticker, meta) in zip(cols, PORTFOLIO.items()):
        default_w = alloc_df2.loc[alloc_df2["Ticker"] == ticker, "Allocation %"].values[0]
        custom_weights[ticker] = col.number_input(ticker, min_value=0.0, max_value=100.0,
                                                   value=float(default_w), step=0.5)

    total_w = sum(custom_weights.values())
    if abs(total_w - 100) > 0.1:
        st.warning(f"Weights sum to {total_w:.1f}% — adjust to reach 100%.")
    else:
        st.success("Weights sum to 100%")

    custom_rows = []
    for ticker, pct in custom_weights.items():
        usd = budget_usd * pct / 100
        p = prices.get(ticker)
        shares = int(usd / p) if p and p > 0 else 0
        custom_rows.append({
            "Ticker": ticker,
            "Company": PORTFOLIO[ticker]["name"],
            "Weight %": f"{pct:.1f}%",
            "USD ($)": f"${usd:,.0f}",
            "CAD ($)": f"${usd/cad_usd_rate:,.0f}",
            "Price": f"${p:.2f}" if p else "N/A",
            "Est. Shares": shares,
            "Risk": PORTFOLIO[ticker]["risk"],
        })

    st.dataframe(pd.DataFrame(custom_rows), use_container_width=True, hide_index=True)

    # Risk concentration
    risk_grp = alloc_df2.groupby("Risk")["USD ($)"].sum().reset_index()
    risk_colors = [RISK_COLOR.get(r, "#888") for r in risk_grp["Risk"]]
    fig_risk = go.Figure(go.Pie(
        labels=risk_grp["Risk"], values=risk_grp["USD ($)"],
        marker_colors=risk_colors, hole=0.4, textinfo="label+percent",
    ))
    fig_risk.update_layout(template="plotly_dark", height=350,
                           title="Portfolio Risk Concentration")
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("""
---
**Important Disclaimer**
This dashboard is for **educational and informational purposes only** and does not constitute financial advice.
All investments involve risk. The Kronos AI model predictions are based on historical patterns and should not
be used as the sole basis for investment decisions. Past performance does not guarantee future results.
Always consult a qualified financial advisor before investing.
""")
