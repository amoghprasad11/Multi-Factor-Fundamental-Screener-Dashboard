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

    # --- Factor weights (must sum to 1.0; used for the 0-100 overall score) ---
    value_weight: float = 1 / 3
    quality_weight: float = 1 / 3
    growth_weight: float = 1 / 3

    # --- Caching (avoids re-downloading the same ticker within cache_ttl_hours) ---
    cache_enabled: bool = True
    cache_dir: str = ".cache"
    cache_ttl_hours: float = 24.0

    # --- Output ---
    output_dir: str = "outputs"
    top_n_display: int = 20


CONFIG = Config()   # <-- adjust here (e.g. CONFIG.max_tickers = None for full S&P 500)


# %% [markdown]
# ## CELL 3 — Universe Construction
# Builds the S&P 500 constituent list (ticker, name, sector, sub_industry)
# from a chain of free, no-key-required sources, in order of reliability:
#
#   1. A maintained GitHub-hosted CSV (datasets/s-and-p-500-companies) —
#      fast, stable, and does NOT trigger the "HTTP Error 403: Forbidden"
#      that `pd.read_html()` against Wikipedia frequently hits from cloud
#      environments like Google Colab (Wikipedia rejects requests that
#      don't send a browser-like User-Agent header, which `pd.read_html`
#      does not set by default).
#   2. Wikipedia, scraped via `requests` with an explicit User-Agent header
#      (fixes the 403 directly) — used only if source #1 fails.
#   3. A small hardcoded snapshot of large, stable constituents — a last
#      resort so the pipeline degrades gracefully instead of crashing.
#      This is intentionally logged loudly as DEGRADED MODE and should
#      not be relied on for a real screen.

# %%
# ==========================================================================
# CELL 3: UNIVERSE CONSTRUCTION
# ==========================================================================
_UNIVERSE_COLUMNS = ["ticker", "name", "sector", "sub_industry"]

# Emergency-only fallback: a small, deliberately conservative snapshot of
# large, long-standing S&P 500 members. This exists purely so the pipeline
# never hard-crashes if BOTH network sources below are unreachable; it is
# NOT a substitute for a real, current constituent list.
_STATIC_SP500_FALLBACK = [
    ("AAPL", "Apple Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals"),
    ("MSFT", "Microsoft Corporation", "Information Technology", "Systems Software"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "Broadline Retail"),
    ("NVDA", "NVIDIA Corporation", "Information Technology", "Semiconductors"),
    ("GOOGL", "Alphabet Inc.", "Communication Services", "Interactive Media & Services"),
    ("META", "Meta Platforms Inc.", "Communication Services", "Interactive Media & Services"),
    ("BRK-B", "Berkshire Hathaway", "Financials", "Multi-Sector Holdings"),
    ("JPM", "JPMorgan Chase & Co.", "Financials", "Diversified Banks"),
    ("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals"),
    ("XOM", "Exxon Mobil Corporation", "Energy", "Integrated Oil & Gas"),
    ("V", "Visa Inc.", "Financials", "Transaction & Payment Processing Services"),
    ("PG", "Procter & Gamble", "Consumer Staples", "Household Products"),
    ("UNH", "UnitedHealth Group", "Health Care", "Managed Health Care"),
    ("HD", "Home Depot Inc.", "Consumer Discretionary", "Home Improvement Retail"),
    ("MA", "Mastercard Inc.", "Financials", "Transaction & Payment Processing Services"),
]


def _normalize_tickers(df: pd.DataFrame, ticker_col: str = "ticker") -> pd.DataFrame:
    """yfinance expects '-' instead of '.' for share classes (BRK.B -> BRK-B)."""
    df = df.copy()
    df[ticker_col] = (
        df[ticker_col].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    )
    return df


def _fetch_sp500_from_github_dataset() -> pd.DataFrame:
    """Primary source: a maintained, plain-CSV mirror of S&P 500 membership."""
    url = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "master/data/constituents.csv"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    missing = [c for c in _UNIVERSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"GitHub dataset source missing expected columns: {missing}")
    return df[_UNIVERSE_COLUMNS]


def _fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """
    Fallback source: scrape Wikipedia directly, but — unlike a bare
    `pd.read_html(url)` call — send an explicit browser-like User-Agent.
    Wikipedia's servers reject the default Python/urllib user agent with a
    403, which is the exact failure this function exists to avoid.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(pd.io.common.StringIO(resp.text))
    df = tables[0]
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    missing = [c for c in _UNIVERSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Wikipedia source missing expected columns: {missing}")
    return df[_UNIVERSE_COLUMNS]


def _fetch_sp500_static_fallback() -> pd.DataFrame:
    """Last-resort emergency fallback — see module-level warning above."""
    logger.warning(
        "DEGRADED MODE: both live S&P 500 sources failed. Falling back to a "
        f"small static snapshot of {len(_STATIC_SP500_FALLBACK)} well-known "
        "constituents. This is NOT a current or complete universe — fix "
        "network access before relying on screen results."
    )
    return pd.DataFrame(_STATIC_SP500_FALLBACK, columns=_UNIVERSE_COLUMNS)


def get_sp500_universe() -> pd.DataFrame:
    """
    Build the S&P 500 constituent universe, trying each source in Cell 3's
    docstring in order until one succeeds.

    Returns
    -------
    pd.DataFrame with columns: ticker, name, sector, sub_industry.
    Missing sector/sub_industry values are filled with "Unclassified"
    rather than dropped, so a row with incomplete metadata still gets
    screened (see also `clean_dataset` in Cell 5).

    Raises
    ------
    RuntimeError only if ALL sources — including the static fallback —
    somehow fail, which should not happen in practice.
    """
    sources = [
        ("GitHub S&P 500 dataset", _fetch_sp500_from_github_dataset),
        ("Wikipedia (with User-Agent header)", _fetch_sp500_from_wikipedia),
        ("static fallback snapshot", _fetch_sp500_static_fallback),
    ]

    last_exc: Optional[Exception] = None
    for name, fetch_fn in sources:
        try:
            df = fetch_fn()
            df = _normalize_tickers(df)
            df["sector"] = df["sector"].fillna("Unclassified")
            df["sub_industry"] = df.get("sub_industry", pd.Series(dtype=str)).fillna("Unclassified")
            universe = df.dropna(subset=["ticker"]).drop_duplicates(subset="ticker").reset_index(drop=True)
            logger.info(f"Universe built via {name}: {len(universe)} constituents")
            return universe
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Universe source '{name}' failed ({exc}); trying next source...")

    raise RuntimeError(
        "Could not build universe from any source (GitHub dataset, Wikipedia, "
        "or static fallback). Check network connectivity, or supply your own "
        "ticker list via a custom DataFrame with columns "
        f"{_UNIVERSE_COLUMNS}."
    ) from last_exc


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


def _cache_path(ticker: str, config: Config) -> str:
    import os
    return os.path.join(config.cache_dir, f"{ticker.replace('/', '_')}.json")


def _cache_read(ticker: str, config: Config) -> Optional[Dict]:
    """Return a cached fundamentals record if present and still fresh, else None."""
    import os
    import json

    if not config.cache_enabled:
        return None
    path = _cache_path(ticker, config)
    if not os.path.exists(path):
        return None
    try:
        age_hours = (time.time() - os.path.getmtime(path)) / 3600.0
        if age_hours > config.cache_ttl_hours:
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None  # corrupt cache entry — treat as a miss, re-fetch


def _cache_write(ticker: str, record: Dict, config: Config) -> None:
    import os
    import json

    if not config.cache_enabled:
        return
    try:
        os.makedirs(config.cache_dir, exist_ok=True)
        with open(_cache_path(ticker, config), "w") as f:
            json.dump(record, f)
    except Exception as exc:
        logger.debug(f"{ticker}: failed to write cache ({exc})")


def fetch_ticker_fundamentals(ticker: str, config: Config) -> Optional[Dict]:
    """
    Fetch a fundamentals snapshot for a single ticker via yfinance, using an
    on-disk cache (see Config.cache_*) to avoid re-downloading the same
    ticker repeatedly across runs within `cache_ttl_hours`.

    Returns None (rather than raising) on failure, so a single bad ticker
    never crashes a multi-hundred-name universe pull.
    """
    cached = _cache_read(ticker, config)
    if cached is not None:
        return cached

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

        record = {
            "ticker": ticker,
            "price": price,
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "total_revenue": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            # --- Value metrics ---
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "ev_to_revenue": info.get("enterpriseToRevenue"),
            # --- Quality metrics ---
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "gross_margin": info.get("grossMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            # --- Growth metrics ---
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            # --- Cash flow / leverage inputs (used to derive further ratios) ---
            "free_cashflow": info.get("freeCashflow"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
        }
        _cache_write(ticker, record, config)
        return record
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
    # NOTE on scope: some commonly-requested ratios (true ROIC using invested
    # capital, net debt/EBITDA, interest coverage) require multi-statement
    # balance-sheet data yfinance's lightweight `.info` snapshot does not
    # reliably expose for the full universe. Rather than approximate them
    # with a shaky proxy and mislabel it, this pipeline derives only the
    # ratios it can compute honestly from available fields, and documents
    # the gap here and in the README/methodology section.
    def _col(name: str) -> pd.Series:
        """Return df[name] if present, else an all-NaN series of the right
        length/index — so a source that omits a field (e.g. yfinance not
        returning totalRevenue for some tickers) degrades that one derived
        ratio to NaN instead of crashing the whole pipeline."""
        if name in df.columns:
            return df[name]
        return pd.Series(np.nan, index=df.index)

    market_cap = _col("market_cap")
    free_cashflow = _col("free_cashflow")
    total_debt = _col("total_debt")
    total_cash = _col("total_cash")
    enterprise_value = _col("enterprise_value")
    trailing_pe = _col("trailing_pe")
    total_revenue = _col("total_revenue")

    df["fcf_yield"] = np.where(market_cap > 0, free_cashflow / market_cap, np.nan)
    df["net_debt"] = total_debt.fillna(0) - total_cash.fillna(0)
    df["net_debt_to_mcap"] = np.where(market_cap > 0, df["net_debt"] / market_cap, np.nan)
    df["ev_to_fcf"] = np.where(
        (free_cashflow > 0) & enterprise_value.notna(), enterprise_value / free_cashflow, np.nan
    )
    df["earnings_yield"] = np.where(trailing_pe > 0, 1.0 / trailing_pe, np.nan)
    df["fcf_margin"] = np.where(total_revenue > 0, free_cashflow / total_revenue, np.nan)

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
# A metric is silently skipped per-stock (not per-universe) if missing, via
# `sector_zscore` operating on whatever non-null values exist — see
# `_composite()`. This is what "handle missing data gracefully" means here:
# a stock with 4 of 6 quality metrics available still gets a quality score
# from those 4; it is not dropped from the screen.
VALUE_METRICS: Dict[str, int] = {
    "trailing_pe": -1,
    "price_to_book": -1,
    "ev_to_ebitda": -1,
    "ev_to_fcf": -1,
    "price_to_sales": -1,
    "fcf_yield": 1,
    "earnings_yield": 1,
}
QUALITY_METRICS: Dict[str, int] = {
    "roe": 1,
    "roa": 1,
    "operating_margin": 1,
    "gross_margin": 1,
    "fcf_margin": 1,
    "debt_to_equity": -1,
    "current_ratio": 1,
    "quick_ratio": 1,
}
GROWTH_METRICS: Dict[str, int] = {
    "revenue_growth": 1,
    "earnings_growth": 1,
    "earnings_quarterly_growth": 1,
}

# Legacy alias retained for backward compatibility with any external code
# that imported FACTOR_WEIGHTS directly. New code should set weights via
# `Config.value_weight` / `quality_weight` / `growth_weight` instead, since
# those are what `compute_factor_scores` actually reads.
FACTOR_WEIGHTS: Dict[str, float] = {"value_score": 1.0, "quality_score": 1.0, "growth_score": 1.0}

# Score bands for the 0-100 overall score, used both in the report and by
# `score_label()` for user-facing displays (e.g. the Streamlit app).
SCORE_BANDS = [
    (90, 100, "Exceptional"),
    (80, 90, "Strong"),
    (70, 80, "Above Average"),
    (60, 70, "Average"),
    (50, 60, "Below Average"),
    (0, 50, "Weak"),
]


def score_label(score_0_100: float) -> str:
    """Map a 0-100 overall score to a human-readable band label."""
    if pd.isna(score_0_100):
        return "N/A"
    for lo, hi, label in SCORE_BANDS:
        if lo <= score_0_100 <= hi:
            return label
    return "N/A"


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
    z-scores, blend into Value/Quality/Growth composites, combine into a
    single weighted composite z-score, and derive two user-facing views
    of the same ranking:

      - `composite_score`   : the raw sector-neutral z-score (existing
                               column, kept for backward compatibility —
                               anything downstream reading this column,
                               e.g. `professional_dashboard.py`, keeps
                               working unchanged).
      - `overall_score_100` : the SAME ranking expressed as a 0-100
                               percentile score, which is far easier for
                               a non-technical reader to interpret than a
                               raw z-score ("78" vs. "0.41 std devs").
                               Percentile rank is used instead of min-max
                               scaling because it is not distorted by one
                               extreme outlier stretching the whole scale.
      - `*_percentile`      : the same percentile treatment applied to
                               each individual factor (value/quality/growth).

    Factor weights come from `config.value_weight` / `quality_weight` /
    `growth_weight` (default: equal-weighted 1/3 each) rather than the
    module-level FACTOR_WEIGHTS constant, so weighting is configurable
    without editing code.
    """
    df = df.copy()
    all_metrics = list(VALUE_METRICS) + list(QUALITY_METRICS) + list(GROWTH_METRICS)
    for col in set(all_metrics):
        if col in df.columns:
            df[col] = winsorize_series(df[col], config.winsorize_limits)

    df = _composite(df, VALUE_METRICS, "value_score")
    df = _composite(df, QUALITY_METRICS, "quality_score")
    df = _composite(df, GROWTH_METRICS, "growth_score")

    weights = {
        "value_score": config.value_weight,
        "quality_score": config.quality_weight,
        "growth_score": config.growth_weight,
    }
    total_weight = sum(weights.values()) or 1.0
    df["composite_score"] = sum(df[c].fillna(0) * w for c, w in weights.items()) / total_weight

    # --- User-facing 0-100 view of the same ranking ---
    df["overall_percentile"] = (df["composite_score"].rank(pct=True) * 100)
    df["overall_score_100"] = df["overall_percentile"].round(1)
    df["value_percentile"] = (df["value_score"].rank(pct=True) * 100).round(1)
    df["quality_percentile"] = (df["quality_score"].rank(pct=True) * 100).round(1)
    df["growth_percentile"] = (df["growth_score"].rank(pct=True) * 100).round(1)
    df["score_band"] = df["overall_score_100"].apply(score_label)

    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype("Int64")
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


# %% [markdown]
# ## CELL 6B — Stock-Level Explanations, Glossary & Risk Flags
# Turns the numeric scores into plain-language output for an individual
# stock: which metrics are actually driving its rank, a short glossary for
# anyone still learning the metrics, and rule-based risk/strength flags.
# Every statement here is generated directly from that stock's own data —
# nothing is inferred or invented.

# %%
# ==========================================================================
# CELL 6B: STOCK-LEVEL EXPLANATIONS, GLOSSARY & RISK FLAGS
# ==========================================================================
METRIC_GLOSSARY: Dict[str, str] = {
    "trailing_pe": "Price / Earnings (trailing 12mo). How many dollars investors pay per dollar of last year's profit. Lower can mean cheaper, but also slower expected growth.",
    "price_to_book": "Price / Book Value. Compares market price to accounting net worth per share. Useful for asset-heavy businesses like banks.",
    "ev_to_ebitda": "Enterprise Value / EBITDA. A capital-structure-neutral valuation multiple — useful for comparing companies with different debt levels.",
    "ev_to_fcf": "Enterprise Value / Free Cash Flow. Similar to EV/EBITDA but based on actual cash generated, not accounting profit.",
    "price_to_sales": "Price / Revenue. Useful for valuing companies that aren't yet profitable.",
    "fcf_yield": "Free Cash Flow / Market Cap. The cash return a company generates relative to its price — higher generally means cheaper on a cash basis.",
    "earnings_yield": "1 / P/E. The inverse of the P/E ratio, useful for comparing against bond yields.",
    "roe": "Return on Equity. Net income as a % of shareholders' equity — how efficiently a company turns equity capital into profit.",
    "roa": "Return on Assets. Net income as a % of total assets — how efficiently a company turns its full asset base into profit.",
    "operating_margin": "Operating Income / Revenue. The % of each sales dollar left after core operating costs, before interest and taxes.",
    "gross_margin": "Gross Profit / Revenue. The % of each sales dollar left after direct production costs.",
    "fcf_margin": "Free Cash Flow / Revenue. The % of each sales dollar converted into actual free cash.",
    "debt_to_equity": "Total Debt / Equity. How much a company relies on debt versus shareholder capital — higher means more leverage and financial risk.",
    "current_ratio": "Current Assets / Current Liabilities. Ability to cover short-term obligations — above 1.0 is generally considered healthy.",
    "quick_ratio": "(Current Assets - Inventory) / Current Liabilities. A stricter short-term liquidity test than the current ratio.",
    "revenue_growth": "Year-over-year revenue growth rate.",
    "earnings_growth": "Year-over-year earnings growth rate.",
    "earnings_quarterly_growth": "Year-over-year earnings growth, most recent quarter.",
}

# Thresholds (in percentile terms, 0-100) used to generate rule-based
# strengths/concerns and risk flags below. These are deliberately simple
# and transparent rather than a black-box model, so every flag can be
# traced back to a specific metric and threshold.
_STRENGTH_PERCENTILE = 75
_CONCERN_PERCENTILE = 25


def get_stock_detail(df: pd.DataFrame, ticker: str) -> Dict:
    """
    Return a full, structured breakdown for a single ticker: identity,
    scores, percentiles, and every available raw metric. Used by both the
    Streamlit app (Cell 12) and directly in a notebook.

    Raises KeyError if the ticker is not present in `df` (e.g. it was
    filtered out during cleaning, or was never part of the run).
    """
    matches = df[df["ticker"] == ticker.upper()]
    if matches.empty:
        raise KeyError(f"'{ticker}' not found in the screened dataset.")
    row = matches.iloc[0]

    metric_cols = list(VALUE_METRICS) + list(QUALITY_METRICS) + list(GROWTH_METRICS)
    fundamentals = {
        col: row[col]
        for col in metric_cols
        if col in row.index and pd.notna(row[col])
    }

    return {
        "ticker": row["ticker"],
        "name": row.get("name", ""),
        "sector": row.get("sector", "Unclassified"),
        "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
        "overall_score_100": row.get("overall_score_100"),
        "overall_percentile": row.get("overall_percentile"),
        "score_band": row.get("score_band"),
        "value_score": row.get("value_score"),
        "value_percentile": row.get("value_percentile"),
        "quality_score": row.get("quality_score"),
        "quality_percentile": row.get("quality_percentile"),
        "growth_score": row.get("growth_score"),
        "growth_percentile": row.get("growth_percentile"),
        "fundamentals": fundamentals,
        "strengths": _generate_strengths(row),
        "concerns": _generate_concerns(row),
        "risk_flags": get_risk_flags(row),
    }


def _generate_strengths(row: pd.Series) -> List[str]:
    """Plain-language strengths, generated only from metrics this stock actually has."""
    strengths = []
    checks = [
        ("roe", "roe", "Strong return on equity"),
        ("roa", "roa", "Strong return on assets"),
        ("operating_margin", "operating_margin", "High operating margin"),
        ("gross_margin", "gross_margin", "High gross margin"),
        ("fcf_margin", "fcf_margin", "Strong free cash flow conversion"),
        ("revenue_growth", "revenue_growth", "Strong revenue growth"),
        ("earnings_growth", "earnings_growth", "Strong earnings growth"),
        ("fcf_yield", "fcf_yield", "Attractive free cash flow yield"),
        ("current_ratio", "current_ratio", "Healthy short-term liquidity"),
    ]
    for metric_col, _, label in checks:
        if metric_col in row.index and pd.notna(row[metric_col]):
            pct = _metric_percentile(row, metric_col)
            if pct is not None and pct >= _STRENGTH_PERCENTILE:
                strengths.append(label)
    # Low leverage is a "higher is better" strength in inverted form
    if "debt_to_equity" in row.index and pd.notna(row["debt_to_equity"]):
        pct = _metric_percentile(row, "debt_to_equity", invert=True)
        if pct is not None and pct >= _STRENGTH_PERCENTILE:
            strengths.append("Low leverage / healthy balance sheet")
    return strengths


def _generate_concerns(row: pd.Series) -> List[str]:
    """Plain-language concerns, generated only from metrics this stock actually has."""
    concerns = []
    checks = [
        ("trailing_pe", "High valuation (elevated P/E)"),
        ("price_to_book", "High valuation relative to book value"),
        ("ev_to_ebitda", "High valuation (elevated EV/EBITDA)"),
        ("debt_to_equity", "Elevated leverage"),
    ]
    for metric_col, label in checks:
        if metric_col in row.index and pd.notna(row[metric_col]):
            # these are all "lower is better" metrics, so a HIGH raw value
            # corresponds to a LOW inverted percentile
            pct = _metric_percentile(row, metric_col, invert=True)
            if pct is not None and pct <= _CONCERN_PERCENTILE:
                concerns.append(label)

    growth_checks = [("revenue_growth", "Weak revenue growth"), ("earnings_growth", "Weak earnings growth")]
    for metric_col, label in growth_checks:
        if metric_col in row.index and pd.notna(row[metric_col]):
            pct = _metric_percentile(row, metric_col)
            if pct is not None and pct <= _CONCERN_PERCENTILE:
                concerns.append(label)

    if "free_cashflow" in row.index and pd.notna(row["free_cashflow"]) and row["free_cashflow"] < 0:
        concerns.append("Negative free cash flow")

    return concerns


def get_risk_flags(row: pd.Series) -> Dict[str, List[str]]:
    """
    Rule-based risk/positive flags derived only from this stock's own data.
    Returned as {"risk_flags": [...], "positive_flags": [...]} — explicitly
    NOT an investment recommendation, just a transparent readout of which
    documented thresholds this stock crosses.
    """
    risk_flags, positive_flags = [], []

    if row.get("overall_score_100") is not None and pd.notna(row.get("overall_score_100")):
        if row["overall_score_100"] < 25:
            risk_flags.append("Unusually low overall composite score vs. peers")

    if "debt_to_equity" in row.index and pd.notna(row["debt_to_equity"]) and row["debt_to_equity"] > 200:
        risk_flags.append("High debt load (Debt/Equity > 200%)")
    elif "debt_to_equity" in row.index and pd.notna(row["debt_to_equity"]) and row["debt_to_equity"] < 50:
        positive_flags.append("Low leverage (Debt/Equity < 50%)")

    if "free_cashflow" in row.index and pd.notna(row["free_cashflow"]) and row["free_cashflow"] < 0:
        risk_flags.append("Negative free cash flow")

    if "current_ratio" in row.index and pd.notna(row["current_ratio"]) and row["current_ratio"] < 1.0:
        risk_flags.append("Current ratio below 1.0 (potential short-term liquidity strain)")

    if "roe" in row.index and pd.notna(row["roe"]) and row["roe"] < 0:
        risk_flags.append("Negative return on equity")

    if "gross_margin" in row.index and pd.notna(row["gross_margin"]) and row["gross_margin"] > 0.6:
        positive_flags.append("Strong gross margin (> 60%)")

    if "revenue_growth" in row.index and pd.notna(row["revenue_growth"]) and row["revenue_growth"] > 0.20:
        positive_flags.append("Strong revenue growth (> 20% YoY)")

    return {"risk_flags": risk_flags, "positive_flags": positive_flags}


def _metric_percentile(row: pd.Series, metric_col: str, invert: bool = False) -> Optional[float]:
    """
    Best-effort single-stock percentile lookup used by the explanation
    generators above. Falls back to the factor-level percentile the metric
    belongs to (value/quality/growth) since per-metric universe-wide
    percentiles aren't persisted as separate columns by default — this
    keeps the explanation directionally correct without bloating the
    output DataFrame with one percentile column per raw metric.
    """
    if metric_col in VALUE_METRICS:
        pct = row.get("value_percentile")
    elif metric_col in QUALITY_METRICS:
        pct = row.get("quality_percentile")
    elif metric_col in GROWTH_METRICS:
        pct = row.get("growth_percentile")
    else:
        return None
    if pct is None or pd.isna(pct):
        return None
    return (100 - pct) if invert else pct


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
