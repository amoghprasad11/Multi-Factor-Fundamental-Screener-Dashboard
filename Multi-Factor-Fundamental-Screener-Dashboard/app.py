"""
Streamlit entry point for the Multi-Factor Fundamental Screener.

This file intentionally contains NO scoring or data-collection logic of
its own — it only orchestrates:

    S&P 500 universe -> financial data -> fundamental scoring -> ranking
                      (all via multi_factor_screener.py)
                                   |
                                   v
                    professional Plotly dashboard
                      (via professional_dashboard.py)
                                   |
                                   v
              search / filter / sort + individual stock drill-down
                          (this file, UI only)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Run in Google Colab:
    See README.md — Colab requires a tunneling helper (e.g. `localtunnel`)
    since Streamlit serves a live local web server rather than rendering
    inline like a notebook cell.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import multi_factor_screener as mfs
from professional_dashboard import build_professional_dashboard

st.set_page_config(
    page_title="Multi-Factor Fundamental Screener",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------
# Pipeline execution (cached so widget interactions don't re-run the scan)
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_run(max_tickers, min_market_cap, request_delay, cache_enabled) -> dict:
    """
    Thin, cached wrapper around `multi_factor_screener.run_pipeline`.
    Streamlit re-runs the whole script on every widget interaction, so
    without this cache every checkbox click would re-download the full
    universe. Cache key = the actual scan parameters, so changing them
    correctly triggers a fresh run.
    """
    config = mfs.Config(
        max_tickers=max_tickers,
        min_market_cap=min_market_cap,
        request_delay=request_delay,
        cache_enabled=cache_enabled,
    )
    return mfs.run_pipeline(config)


def score_badge(value: float) -> str:
    """Small emoji indicator alongside a 0-100 score for quick scanning."""
    if pd.isna(value):
        return "—"
    if value >= 80:
        return "🟢"
    if value >= 60:
        return "🟡"
    if value >= 40:
        return "🟠"
    return "🔴"


# -----------------------------------------------------------------------
# Sidebar — run controls + filters
# -----------------------------------------------------------------------
st.sidebar.title("⚙️ Screener Controls")

run_mode = st.sidebar.radio(
    "Universe size",
    ["Quick test (25 stocks)", "Medium (100 stocks)", "Full S&P 500"],
    index=1,
    help="Start with a smaller run to confirm everything works before "
         "pulling the full ~500-name universe (which takes several minutes).",
)
max_tickers_map = {"Quick test (25 stocks)": 25, "Medium (100 stocks)": 100, "Full S&P 500": None}
max_tickers = max_tickers_map[run_mode]

min_market_cap = st.sidebar.number_input(
    "Minimum market cap ($)", min_value=0, value=300_000_000, step=50_000_000, format="%d"
)
request_delay = st.sidebar.slider(
    "Request delay (seconds)", 0.0, 1.0, 0.25, 0.05,
    help="Delay between API calls. Increase if you hit rate limits.",
)
cache_enabled = st.sidebar.checkbox("Use on-disk cache", value=True)

run_clicked = st.sidebar.button("▶ Run Screener", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_clicked:
    with st.spinner(f"Screening {run_mode.lower()}... this can take a few minutes for larger universes."):
        try:
            st.session_state["results"] = cached_run(
                max_tickers, min_market_cap, request_delay, cache_enabled
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.session_state["results"] = None

results = st.session_state["results"]

# -----------------------------------------------------------------------
# Main area
# -----------------------------------------------------------------------
st.title("📊 Multi-Factor Fundamental Screener")
st.caption(
    "Ranks S&P 500 companies on Value, Quality, and Growth using sector-neutral "
    "scoring. Not a recommendation — a quantitative research starting point."
)

if results is None:
    st.info("Configure your run in the sidebar and click **Run Screener** to begin.")
    st.stop()

df = results["data"]
metrics = results.get("metrics", {})

# --- Score legend / methodology explainer ---
with st.expander("ℹ️ How to read these scores"):
    st.markdown(
        """
| Score range | Meaning |
|---|---|
| 90–100 | Exceptional |
| 80–89 | Strong |
| 70–79 | Above Average |
| 60–69 | Average |
| 50–59 | Below Average |
| Below 50 | Weak |

- **Value** — is the stock cheap relative to earnings, book value, and cash flow?
- **Quality** — how profitable and financially sound is the business (margins, returns on capital, leverage)?
- **Growth** — how fast is the business expanding (revenue, earnings)?
- **Overall Score** — a 0–100 percentile blend of Value, Quality, and Growth (weights configurable in `Config`).
- **Percentile** — where a stock ranks relative to *every other stock in this run*, not an absolute grade.

**Disclaimer:** this is a quantitative research tool, not investment advice.
Scores reflect historical/current fundamentals and do not predict future returns.
        """
    )

# --- KPI row ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Stocks Screened", len(df))
k2.metric("Top Ranked Stock", str(df.iloc[0]["ticker"]))
k3.metric("Top Score", f"{df.iloc[0]['overall_score_100']:.1f}")
k4.metric("Average Score", f"{df['overall_score_100'].mean():.1f}")
ic = metrics.get("information_coefficient")
k5.metric("Information Coefficient", f"{ic:.3f}" if ic is not None else "—")

st.divider()

# --- Dashboard ---
st.plotly_chart(build_professional_dashboard(results), use_container_width=True)

st.divider()

# --- Filters ---
st.subheader("🔍 Explore Results")
f1, f2, f3, f4 = st.columns([2, 2, 1.5, 1.5])

with f1:
    search = st.text_input("Search ticker or company name", "")
with f2:
    sectors = sorted(df["sector"].dropna().unique().tolist())
    selected_sectors = st.multiselect("Sector", sectors, default=[])
with f3:
    min_score = st.slider("Min overall score", 0, 100, 0)
with f4:
    sort_by = st.selectbox(
        "Sort by",
        ["overall_score_100", "value_score", "quality_score", "growth_score", "market_cap"],
        format_func=lambda c: {
            "overall_score_100": "Overall Score",
            "value_score": "Value",
            "quality_score": "Quality",
            "growth_score": "Growth",
            "market_cap": "Market Cap",
        }[c],
    )

filtered = df.copy()
if search:
    s = search.strip().upper()
    filtered = filtered[
        filtered["ticker"].str.upper().str.contains(s)
        | filtered["name"].str.upper().str.contains(s, na=False)
    ]
if selected_sectors:
    filtered = filtered[filtered["sector"].isin(selected_sectors)]
filtered = filtered[filtered["overall_score_100"].fillna(0) >= min_score]
filtered = filtered.sort_values(sort_by, ascending=False)

display_cols = {
    "rank": "Rank", "ticker": "Ticker", "name": "Company", "sector": "Sector",
    "overall_score_100": "Overall", "overall_percentile": "Percentile",
    "value_score": "Value", "quality_score": "Quality", "growth_score": "Growth",
    "market_cap": "Market Cap", "trailing_return_12m": "12M Return",
}
available_cols = [c for c in display_cols if c in filtered.columns]
table = filtered[available_cols].rename(columns=display_cols).reset_index(drop=True)

if "Overall" in table.columns:
    table.insert(1, "", table["Overall"].apply(score_badge))

st.dataframe(table, use_container_width=True, height=450)
st.caption(f"Showing {len(filtered)} of {len(df)} screened stocks.")

st.divider()

# --- Individual stock analysis ---
st.subheader("🔎 Individual Stock Analysis")
if filtered.empty:
    st.warning("No stocks match the current filters.")
else:
    pick = st.selectbox("Select a stock", filtered["ticker"].tolist())
    detail = mfs.get_stock_detail(df, pick)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### {detail['ticker']} — {detail['name']}")
        st.caption(detail["sector"])
        st.metric("Overall Score", f"{detail['overall_score_100']:.1f}", detail["score_band"])
        st.metric("Value", f"{detail['value_score']:.2f}" if pd.notna(detail["value_score"]) else "—")
        st.metric("Quality", f"{detail['quality_score']:.2f}" if pd.notna(detail["quality_score"]) else "—")
        st.metric("Growth", f"{detail['growth_score']:.2f}" if pd.notna(detail["growth_score"]) else "—")

    with c2:
        st.markdown("**Why did this stock rank here?**")
        if detail["strengths"]:
            st.markdown("**Strengths:**")
            for s in detail["strengths"]:
                st.markdown(f"- ✅ {s}")
        if detail["concerns"]:
            st.markdown("**Potential concerns:**")
            for c in detail["concerns"]:
                st.markdown(f"- ⚠️ {c}")
        if not detail["strengths"] and not detail["concerns"]:
            st.caption("No standout strengths or concerns at the current thresholds.")

        flags = detail["risk_flags"]
        if flags["risk_flags"] or flags["positive_flags"]:
            st.markdown("**Risk flags (not investment advice):**")
            for f in flags["risk_flags"]:
                st.markdown(f"- 🚩 {f}")
            for f in flags["positive_flags"]:
                st.markdown(f"- 👍 {f}")

    with st.expander("Full fundamentals"):
        fund_df = pd.DataFrame(
            [(k, v, mfs.METRIC_GLOSSARY.get(k, "")) for k, v in detail["fundamentals"].items()],
            columns=["Metric", "Value", "What it means"],
        )
        st.dataframe(fund_df, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Data: Yahoo Finance (yfinance) + a GitHub-hosted S&P 500 constituent list, "
    "with a Wikipedia fallback. This tool is for research/educational purposes "
    "only and does not constitute investment advice."
)
