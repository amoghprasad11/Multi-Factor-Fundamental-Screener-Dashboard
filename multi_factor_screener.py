# =============================================================================
# MULTI-FACTOR FUNDAMENTAL SCREENER & DASHBOARD
# -----------------------------------------------------------------------------
# A cross-sectional equity screener that ranks a stock universe on Value,
# Quality, and Growth factors, built to run end-to-end in Google Colab.
#
# Author:  (your name)
# License: MIT
#
# HOW TO USE IN COLAB:
#   1. Copy each "CELL N" block below into its own Colab cell, in order.
#   2. Run Cell 1 first to install dependencies.
#   3. Adjust CONFIG in Cell 3 (universe size, API keys, filters).
#   4. Run cells sequentially. The final cells produce the dashboard,
#      validation metrics, and exported report.
# =============================================================================


# %% [markdown]
# ## CELL 1 — Environment Setup
# Run once per Colab session. Installs all required packages.

# %%
# ==========================================================================
# CELL 1: ENVIRONMENT SETUP  (Colab shell command — remove '#COLAB#' prefix)
# ==========================================================================
#COLAB# !pip install -q yfinance pandas numpy scipy matplotlib seaborn plotly requests lxml


# %% [markdown]
# ## CELL 2 — Imports & Global Configuration

# %%
# ==========================================================================
# CELL 2: IMPORTS & CONFIGURATION
# ==========================================================================
from __future__ import annotations

import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("factor_screener")

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "yfinance is required. In Colab, run: !pip install yfinance"
    ) from exc

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110


@dataclass
class Config:
    """Central configuration for the screener. Edit these values, not the
    functions below, to change screener behavior."""

    # --- Universe ---
    max_tickers: Optional[int] = 100          # cap for speed/rate-limits; set None for full universe
    min_market_cap: float = 300_000_000        # exclude micro-caps / bad data below this

    # --- API / networking ---
    request_delay: float = 0.25                 # seconds between yfinance requests (politeness/rate-limit)
    max_retries: int = 3
    retry_backoff: float = 2.0                  # exponential backoff base (seconds)
    fmp_api_key: Optional[str] = None            # optional: set to enable Financial Modeling Prep enrichment

    # --- Statistics ---
    winsorize_limits: Tuple[float, float] = (0.02, 0.02)   # trim 2% tails each side

    # --- Output ---
    output_dir: str = "outputs"
    top_n_display: int = 20


CONFIG = Config()   # <-- adjust here (e.g. CONFIG.max_tickers = None for full S&P 500)


# %% [markdown]
# ## CELL 3 — Universe Construction
# Pulls the current S&P 500 constituent list (ticker, name, sector) from
# Wikipedia as a free, no-key-required source of index membership.

# %%
# ==========================================================================
# CELL 3: UNIVERSE CONSTRUCTION
# ==========================================================================
def get_sp500_universe() -> pd.DataFrame:
    """
    Scrape current S&P 500 constituents from Wikipedia.

    Returns
    -------
    pd.DataFrame with columns: ticker, name, sector, sub_industry

    Raises
    ------
    RuntimeError if the page cannot be parsed (schema drift, network error).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        df = df.rename(
            columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
            }
        )
        # yfinance uses '-' instead of '.' for share classes (e.g. BRK.B -> BRK-B)
        df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)
        keep = [c for c in ["ticker", "name", "sector", "sub_industry"] if c in df.columns]
        universe = df[keep].dropna(subset=["ticker"]).drop_duplicates(subset="ticker")
        logger.info(f"Universe built: {len(universe)} S&P 500 constituents")
        return universe.reset_index(drop=True)
    except Exception as exc:
        logger.error(f"Failed to fetch S&P 500 list from Wikipedia: {exc}")
        raise RuntimeError(
            "Could not build universe. Check network connectivity, or supply "
            "your own ticker list via a custom DataFrame with columns "
            "[ticker, name, sector, sub_industry]."
        ) from exc


# %% [markdown]
# ## CELL 4 — Data Collection Pipeline
# Fetches per-ticker fundamental snapshots and trailing price returns via
# yfinance, with retry/backoff error handling and rate limiting. Optionally
# enriches with Financial Modeling Prep (FMP) data if an API key is set.

# %%
# ==========================================================================
# CELL 4: DATA COLLECTION PIPELINE
# ==========================================================================
def _retry(fn, *args, max_retries: int = 3, backoff: float = 2.0, **kwargs):
    """Generic retry wrapper with exponential backoff for flaky network calls."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - intentionally broad for network flakiness
            last_exc = exc
            wait = backoff ** attempt
            logger.debug(f"Retry {attempt}/{max_retries} after error: {exc} (sleeping {wait:.1f}s)")
            time.sleep(wait)
    raise last_exc  # exhausted retries


def fetch_ticker_fundamentals(ticker: str, config: Config) -> Optional[Dict]:
    """
    Fetch a fundamentals snapshot for a single ticker via yfinance.

    Returns None (rather than raising) on failure, so a single bad ticker
    never crashes a multi-hundred-name universe pull.
    """
    try:
        tk = yf.Ticker(ticker)
        info = _retry(
            lambda: tk.info,
            max_retries=config.max_retries,
            backoff=config.retry_backoff,
        )
        if not info:
            logger.warning(f"{ticker}: empty info payload, skipping")
            return None

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            logger.warning(f"{ticker}: no price data, skipping")
            return None

        return {
            "ticker": ticker,
            "price": price,
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "free_cashflow": info.get("freeCashflow"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
        }
    except Exception as exc:
        logger.warning(f"{ticker}: fundamentals fetch failed ({exc})")
        return None


def fetch_trailing_returns(tickers: List[str], config: Config) -> pd.Series:
    """Batch-download ~13 months of daily closes and compute trailing 12M return."""
    if not tickers:
        return pd.Series(dtype=float)
    try:
        raw = yf.download(
            tickers, period="13mo", interval="1d",
            progress=False, group_by="ticker", threads=True, auto_adjust=True,
        )
    except Exception as exc:
        logger.error(f"Batch price download failed: {exc}")
        return pd.Series(dtype=float)

    returns: Dict[str, float] = {}
    for t in tickers:
        try:
            close = raw[t]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
            if len(close) < 2:
                continue
            returns[t] = float(close.iloc[-1] / close.iloc[0] - 1.0)
        except Exception:
            continue  # ticker missing from batch response; skip silently
    return pd.Series(returns, name="trailing_return_12m")


def enrich_with_fmp(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Optional enrichment step using Financial Modeling Prep's TTM ratios
    endpoint (adds a cross-checked ROIC and FCF-yield figure). Skipped
    entirely (with a log message) if no API key is configured.
    """
    if not config.fmp_api_key:
        logger.info("FMP_API_KEY not set — skipping optional FMP enrichment.")
        return df

    base_url = "https://financialmodelingprep.com/api/v3/ratios-ttm"
    rows = []
    for t in df["ticker"]:
        try:
            resp = requests.get(f"{base_url}/{t}", params={"apikey": config.fmp_api_key}, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if payload:
                rows.append({
                    "ticker": t,
                    "fmp_roic_ttm": payload[0].get("returnOnCapitalEmployedTTM"),
                    "fmp_fcf_yield_ttm": payload[0].get("freeCashFlowYieldTTM"),
                })
        except Exception as exc:
            logger.debug(f"FMP enrichment failed for {t}: {exc}")
        time.sleep(config.request_delay)

    if rows:
        fmp_df = pd.DataFrame(rows)
        df = df.merge(fmp_df, on="ticker", how="left")
        logger.info(f"FMP enrichment merged for {len(fmp_df)} tickers.")
    return df


def build_raw_dataset(universe: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Orchestrates the full collection pass: fundamentals + trailing returns."""
    tickers = universe["ticker"].tolist()
    if config.max_tickers:
        tickers = tickers[: config.max_tickers]

    logger.info(f"Fetching fundamentals for {len(tickers)} tickers (this can take a few minutes)...")
    records = []
    for i, t in enumerate(tickers, start=1):
        rec = fetch_ticker_fundamentals(t, config)
        if rec is not None:
            records.append(rec)
        if i % 25 == 0 or i == len(tickers):
            logger.info(f"  ... {i}/{len(tickers)} tickers processed ({len(records)} successful)")
        time.sleep(config.request_delay)

    fundamentals = pd.DataFrame(records)
    if fundamentals.empty:
        raise RuntimeError(
            "No fundamental data was collected. Check network connectivity "
            "and that ticker symbols are valid."
        )

    logger.info("Fetching trailing 12-month returns for validation metrics...")
    trailing = fetch_trailing_returns(fundamentals["ticker"].tolist(), config)
    fundamentals = fundamentals.merge(
        trailing, left_on="ticker", right_index=True, how="left"
    )

    merged = universe.merge(fundamentals, on="ticker", how="inner")
    merged = enrich_with_fmp(merged, config)
    logger.info(f"Raw dataset built: {len(merged)} tickers with usable data")
    return merged


# %% [markdown]
# ## CELL 5 — Data Cleaning & Feature Engineering
# Filters out low-quality/micro-cap rows, derives additional ratios, and
# prepares the metric set used by the scoring model.

# %%
# ==========================================================================
# CELL 5: DATA CLEANING & FEATURE ENGINEERING
# ==========================================================================
def clean_dataset(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Apply quality filters and derive engineered features."""
    df = df.copy()
    n_before = len(df)

    df = df[df["market_cap"].notna() & (df["market_cap"] >= config.min_market_cap)]
    df = df.drop_duplicates(subset="ticker")
    logger.info(f"Quality filters: {n_before} -> {len(df)} tickers "
                f"(min market cap ${config.min_market_cap:,.0f})")

    # --- Derived features ---
    df["fcf_yield"] = np.where(
        df["market_cap"] > 0, df["free_cashflow"] / df["market_cap"], np.nan
    )
    df["net_debt"] = df["total_debt"].fillna(0) - df["total_cash"].fillna(0)
    df["net_debt_to_mcap"] = np.where(
        df["market_cap"] > 0, df["net_debt"] / df["market_cap"], np.nan
    )

    if "sector" not in df.columns or df["sector"].isna().all():
        df["sector"] = "Unclassified"
    df["sector"] = df["sector"].fillna("Unclassified")

    return df.reset_index(drop=True)


def winsorize_series(s: pd.Series, limits: Tuple[float, float]) -> pd.Series:
    """Winsorize a series (clip extreme tails) while preserving the original index."""
    valid = s.dropna()
    if len(valid) < 5:
        return s
    clipped = stats.mstats.winsorize(valid.values, limits=limits)
    return pd.Series(clipped, index=valid.index).reindex(s.index)


def sector_zscore(df: pd.DataFrame, col: str, sector_col: str = "sector") -> pd.Series:
    """
    Compute a sector-relative z-score for a metric, so a bank's low P/E isn't
    penalized/rewarded against a software company's structurally different P/E.
    Sectors with too few members (<3) or zero variance fall back to 0 (neutral).
    """
    def _z(group: pd.Series) -> pd.Series:
        if group.dropna().shape[0] < 3 or group.std(skipna=True) in (0, None) or pd.isna(group.std()):
            return pd.Series(0.0, index=group.index)
        return (group - group.mean(skipna=True)) / group.std(skipna=True)

    return df.groupby(sector_col)[col].transform(_z)


# %% [markdown]
# ## CELL 6 — Core Scoring Model
# Combines individual metrics into Value / Quality / Growth composite
# factor scores via sector-neutral z-scoring, then blends into a single
# ranked composite score.

# %%
# ==========================================================================
# CELL 6: CORE SCORING MODEL
# ==========================================================================
# Metric -> sign mapping. +1 means "higher is better", -1 means "lower is better".
VALUE_METRICS: Dict[str, int] = {
    "trailing_pe": -1,
    "price_to_book": -1,
    "ev_to_ebitda": -1,
    "fcf_yield": 1,
}
QUALITY_METRICS: Dict[str, int] = {
    "roe": 1,
    "roa": 1,
    "operating_margin": 1,
    "gross_margin": 1,
    "debt_to_equity": -1,
    "current_ratio": 1,
}
GROWTH_METRICS: Dict[str, int] = {
    "revenue_growth": 1,
    "earnings_growth": 1,
}

FACTOR_WEIGHTS: Dict[str, float] = {"value_score": 1.0, "quality_score": 1.0, "growth_score": 1.0}


def _composite(df: pd.DataFrame, metrics: Dict[str, int], out_col: str) -> pd.DataFrame:
    """Build one composite factor score (e.g. 'value_score') from its component metrics."""
    z_components = []
    for col, sign in metrics.items():
        if col not in df.columns:
            logger.debug(f"Metric '{col}' not present in dataset; skipping in {out_col}")
            continue
        z_components.append(sign * sector_zscore(df, col))

    if not z_components:
        df[out_col] = np.nan
        return df

    stacked = pd.concat(z_components, axis=1)
    df[out_col] = stacked.mean(axis=1, skipna=True)
    return df


def compute_factor_scores(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    End-to-end scoring: winsorize raw metrics, compute sector-neutral
    z-scores, blend into Value/Quality/Growth composites, then combine
    into a single weighted composite score and rank.
    """
    df = df.copy()
    all_metrics = list(VALUE_METRICS) + list(QUALITY_METRICS) + list(GROWTH_METRICS)
    for col in set(all_metrics):
        if col in df.columns:
            df[col] = winsorize_series(df[col], config.winsorize_limits)

    df = _composite(df, VALUE_METRICS, "value_score")
    df = _composite(df, QUALITY_METRICS, "quality_score")
    df = _composite(df, GROWTH_METRICS, "growth_score")

    weighted = sum(df[c] * w for c, w in FACTOR_WEIGHTS.items())
    total_weight = sum(FACTOR_WEIGHTS.values())
    df["composite_score"] = weighted / total_weight

    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype("Int64")
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


# %% [markdown]
# ## CELL 7 — Signal Validation & Performance Metrics
# Sanity-checks the composite score against trailing 12-month returns
# (Information Coefficient + quintile spread). NOTE: this is a *trailing*,
# not a point-in-time forward-looking backtest — see caveat in the
# function docstring.

# %%
# ==========================================================================
# CELL 7: PERFORMANCE / VALIDATION METRICS
# ==========================================================================
def evaluate_signal(df: pd.DataFrame) -> Tuple[Dict, Optional[pd.Series]]:
    """
    Evaluate how well the composite score co-moves with trailing returns.

    IMPORTANT CAVEAT: Free fundamentals APIs return *current* snapshot data,
    not point-in-time historical fundamentals. This function therefore
    measures the CONCURRENT relationship between today's factor score and
    the past 12 months of returns — it is a plausibility/sanity check, NOT
    a leakage-free predictive backtest. A production backtest would require
    point-in-time fundamentals (e.g., Compustat/CRSP, WRDS, or a vendor with
    as-reported history) matched to FORWARD returns.
    """
    valid = df.dropna(subset=["composite_score", "trailing_return_12m"])
    if len(valid) < 10:
        logger.warning("Insufficient overlapping data to compute validation metrics.")
        return {}, None

    ic, p_value = stats.spearmanr(valid["composite_score"], valid["trailing_return_12m"])

    valid = valid.copy()
    n_bins = min(5, valid["composite_score"].nunique())
    valid["quintile"] = pd.qcut(valid["composite_score"], n_bins, labels=False, duplicates="drop")
    decile_returns = valid.groupby("quintile")["trailing_return_12m"].mean()

    top = decile_returns.iloc[-1]
    bottom = decile_returns.iloc[0]

    metrics = {
        "n_stocks": len(valid),
        "information_coefficient": round(float(ic), 4),
        "ic_p_value": round(float(p_value), 4),
        "top_quintile_return": round(float(top), 4),
        "bottom_quintile_return": round(float(bottom), 4),
        "quintile_spread": round(float(top - bottom), 4),
    }
    logger.info(
        f"Signal check — IC: {metrics['information_coefficient']:.3f} "
        f"(p={metrics['ic_p_value']:.3f}) | "
        f"Top-Bottom Quintile Spread: {metrics['quintile_spread']:.1%}"
    )
    return metrics, decile_returns


# %% [markdown]
# ## CELL 8 — Visualizations & Dashboard
# Produces a static matplotlib/seaborn report figure plus an interactive
# Plotly dashboard suitable for embedding in a notebook or exporting to HTML.

# %%
# ==========================================================================
# CELL 8: VISUALIZATIONS & DASHBOARD
# ==========================================================================
def plot_sector_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Average factor scores by sector — quick view of where the screen is finding opportunities."""
    pivot = df.groupby("sector")[["value_score", "quality_score", "growth_score", "composite_score"]].mean()
    pivot = pivot.sort_values("composite_score", ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                linewidths=0.5, cbar_kws={"label": "Avg. Z-Score"}, ax=ax)
    ax.set_title("Average Factor Scores by Sector", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_factor_scatter(df: pd.DataFrame) -> go.Figure:
    """Interactive Value vs. Quality scatter, sized by market cap, colored by sector."""
    plot_df = df.dropna(subset=["value_score", "quality_score"])
    fig = px.scatter(
        plot_df, x="value_score", y="quality_score",
        size="market_cap", color="sector", hover_name="ticker",
        hover_data={"composite_score": ":.2f", "market_cap": ":,.0f"},
        size_max=45, title="Value vs. Quality Factor Map (bubble size = market cap)",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(template="plotly_white", height=600)
    return fig


def plot_top_bottom_bar(df: pd.DataFrame, config: Config) -> plt.Figure:
    """Bar chart of the top-N ranked tickers by composite score."""
    top = df.head(config.top_n_display).sort_values("composite_score")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in top["composite_score"]]
    ax.barh(top["ticker"], top["composite_score"], color=colors)
    ax.set_title(f"Top {config.top_n_display} Ranked Stocks — Composite Factor Score",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Composite Score (sector-neutral z-score)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_factor_correlation(df: pd.DataFrame) -> plt.Figure:
    """Correlation matrix of the three composite factors + key raw metrics."""
    cols = ["value_score", "quality_score", "growth_score",
            "trailing_pe", "roe", "revenue_growth"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Factor & Metric Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_quintile_returns(decile_returns: Optional[pd.Series]) -> Optional[plt.Figure]:
    """Bar chart of trailing return by composite-score quintile (signal sanity check)."""
    if decile_returns is None or decile_returns.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("RdYlGn", len(decile_returns))
    ax.bar(range(len(decile_returns)), decile_returns.values * 100, color=colors)
    ax.set_xticks(range(len(decile_returns)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(decile_returns))])
    ax.set_ylabel("Avg. Trailing 12M Return (%)")
    ax.set_title("Trailing Return by Composite-Score Quintile\n(Q1=lowest score, Q5=highest)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig


def build_dashboard(df: pd.DataFrame, config: Config) -> go.Figure:
    """Single combined interactive Plotly dashboard (4 panels) for notebook display or HTML export."""
    top = df.head(config.top_n_display).sort_values("composite_score")
    sector_avg = df.groupby("sector")["composite_score"].mean().sort_values()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Value vs. Quality (bubble = market cap)",
            "Top Ranked Stocks — Composite Score",
            "Average Composite Score by Sector",
            "Composite Score Distribution",
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]],
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )

    scatter_df = df.dropna(subset=["value_score", "quality_score"])
    fig.add_trace(
        go.Scatter(
            x=scatter_df["value_score"], y=scatter_df["quality_score"],
            mode="markers", text=scatter_df["ticker"],
            marker=dict(
                size=np.clip(scatter_df["market_cap"] / scatter_df["market_cap"].max() * 40, 4, 40),
                color=scatter_df["composite_score"], colorscale="RdYlGn", showscale=True,
                colorbar=dict(title="Score", x=0.46),
            ),
            hovertemplate="%{text}<br>Value: %{x:.2f}<br>Quality: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(x=top["composite_score"], y=top["ticker"], orientation="h",
               marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in top["composite_score"]]),
        row=1, col=2,
    )

    fig.add_trace(
        go.Bar(x=sector_avg.values, y=sector_avg.index, orientation="h",
               marker_color="steelblue"),
        row=2, col=1,
    )

    fig.add_trace(
        go.Histogram(x=df["composite_score"].dropna(), nbinsx=30, marker_color="mediumpurple"),
        row=2, col=2,
    )

    fig.update_layout(
        height=850, showlegend=False, template="plotly_white",
        title_text="Multi-Factor Fundamental Screener — Dashboard", title_x=0.5,
    )
    return fig


# %% [markdown]
# ## CELL 9 — Export & Reporting
# Writes the ranked dataset to CSV and generates a plain-text research
# summary highlighting the top-ranked names, suitable for pasting into a
# README or research note.

# %%
# ==========================================================================
# CELL 9: EXPORT & REPORTING
# ==========================================================================
import os


def export_results(df: pd.DataFrame, metrics: Dict, config: Config) -> str:
    """Save the full ranked dataset and a markdown summary report to disk."""
    os.makedirs(config.output_dir, exist_ok=True)

    csv_path = os.path.join(config.output_dir, "factor_screen_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved full results to {csv_path}")

    report_path = os.path.join(config.output_dir, "screen_summary.md")
    top = df.head(config.top_n_display)

    lines = [
        "# Multi-Factor Fundamental Screen — Summary Report",
        "",
        f"Universe size: **{len(df)}** stocks passing quality filters.",
        "",
        "## Signal Validation",
        f"- Information Coefficient (Spearman): **{metrics.get('information_coefficient', 'n/a')}** "
        f"(p={metrics.get('ic_p_value', 'n/a')})",
        f"- Top-minus-bottom quintile spread (trailing 12M): "
        f"**{metrics.get('quintile_spread', 'n/a')}**",
        "",
        f"## Top {config.top_n_display} Ranked Stocks",
        "",
        "| Rank | Ticker | Sector | Composite | Value | Quality | Growth |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {row['rank']} | {row['ticker']} | {row.get('sector', 'n/a')} | "
            f"{row['composite_score']:.2f} | {row.get('value_score', np.nan):.2f} | "
            f"{row.get('quality_score', np.nan):.2f} | {row.get('growth_score', np.nan):.2f} |"
        )
    lines += [
        "",
        "_Note: composite scores are sector-neutral z-scores; validation is a "
        "trailing concurrent check, not a leakage-free forward backtest._",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved research summary to {report_path}")
    return report_path


# %% [markdown]
# ## CELL 10 — Main Orchestration
# Runs the full pipeline end-to-end with top-level error handling.

# %%
# ==========================================================================
# CELL 10: MAIN ORCHESTRATION
# ==========================================================================
def run_pipeline(config: Config = CONFIG) -> Dict:
    """
    Execute the full screener pipeline: universe -> collect -> clean ->
    score -> validate -> visualize -> export.

    Returns a dict of key artifacts (dataframe, metrics, figures) for
    interactive use in a notebook.
    """
    logger.info("=" * 70)
    logger.info("MULTI-FACTOR FUNDAMENTAL SCREENER — PIPELINE START")
    logger.info("=" * 70)

    try:
        universe = get_sp500_universe()
        raw = build_raw_dataset(universe, config)
        clean = clean_dataset(raw, config)
        scored = compute_factor_scores(clean, config)
        metrics, decile_returns = evaluate_signal(scored)

        figs = {
            "sector_heatmap": plot_sector_heatmap(scored),
            "factor_scatter": plot_factor_scatter(scored),
            "top_bottom_bar": plot_top_bottom_bar(scored, config),
            "factor_correlation": plot_factor_correlation(scored),
            "quintile_returns": plot_quintile_returns(decile_returns),
            "dashboard": build_dashboard(scored, config),
        }

        report_path = export_results(scored, metrics, config)

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"  Tickers screened : {len(scored)}")
        logger.info(f"  Report           : {report_path}")
        logger.info("=" * 70)

        return {"data": scored, "metrics": metrics, "figures": figs, "decile_returns": decile_returns}

    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        raise


# %% [markdown]
# ## CELL 11 — Run It
# Executes the pipeline. In Colab, the static figures render inline
# automatically; call `results["figures"]["dashboard"].show()` for the
# interactive Plotly dashboard.

# %%
# ==========================================================================
# CELL 11: RUN
# ==========================================================================
if __name__ == "__main__":
    results = run_pipeline(CONFIG)
    results["figures"]["dashboard"].show()
    print(results["data"].head(10)[["rank", "ticker", "sector", "composite_score"]])
