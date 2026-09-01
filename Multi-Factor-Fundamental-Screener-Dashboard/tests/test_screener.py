"""
Tests for the Multi-Factor Fundamental Screener.

These tests use small, mocked/synthetic DataFrames and do NOT hit the
network (no yfinance, no GitHub/Wikipedia calls) — they run instantly and
reliably in CI. Run with:

    pytest tests/test_screener.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import multi_factor_screener as mfs


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture
def config():
    return mfs.Config(max_tickers=None, min_market_cap=1_000_000, cache_enabled=False)


@pytest.fixture
def raw_universe_like_df():
    """A small synthetic dataset shaped like the output of build_raw_dataset()."""
    rng = np.random.default_rng(7)
    n = 60
    sectors = rng.choice(["Technology", "Financials", "Health Care", "Energy"], size=n)
    df = pd.DataFrame({
        "ticker": [f"SYN{i:03d}" for i in range(n)],
        "name": [f"Synthetic Co {i}" for i in range(n)],
        "sector": sectors,
        "sub_industry": "Synthetic Sub-Industry",
        "price": rng.uniform(10, 300, n),
        "market_cap": rng.lognormal(mean=21, sigma=1.4, size=n),
        "enterprise_value": rng.lognormal(mean=21, sigma=1.4, size=n),
        "total_revenue": rng.lognormal(mean=20, sigma=1.2, size=n),
        "trailing_pe": rng.uniform(5, 50, n),
        "price_to_book": rng.uniform(0.5, 12, n),
        "price_to_sales": rng.uniform(0.5, 15, n),
        "ev_to_ebitda": rng.uniform(3, 25, n),
        "roe": rng.normal(0.14, 0.12, n),
        "roa": rng.normal(0.07, 0.06, n),
        "operating_margin": rng.normal(0.14, 0.09, n),
        "gross_margin": rng.uniform(0.15, 0.75, n),
        "debt_to_equity": rng.uniform(0, 220, n),
        "current_ratio": rng.uniform(0.5, 3.5, n),
        "quick_ratio": rng.uniform(0.3, 3.0, n),
        "revenue_growth": rng.normal(0.08, 0.14, n),
        "earnings_growth": rng.normal(0.09, 0.22, n),
        "earnings_quarterly_growth": rng.normal(0.08, 0.25, n),
        "free_cashflow": rng.normal(4e8, 3e8, n),
        "total_cash": rng.lognormal(mean=19, sigma=1.1, size=n),
        "total_debt": rng.lognormal(mean=19, sigma=1.2, size=n),
        "beta": rng.normal(1.0, 0.35, n),
        "dividend_yield": rng.uniform(0, 0.04, n),
        "trailing_return_12m": rng.normal(0.10, 0.28, n),
    })
    return df


# -----------------------------------------------------------------------
# Universe / ticker normalization
# -----------------------------------------------------------------------
def test_normalize_tickers_converts_dots_to_dashes():
    df = pd.DataFrame({"ticker": ["BRK.B", "bf.b", " AAPL "]})
    out = mfs._normalize_tickers(df)
    assert list(out["ticker"]) == ["BRK-B", "BF-B", "AAPL"]


def test_static_fallback_has_required_schema():
    df = mfs._fetch_sp500_static_fallback()
    assert list(df.columns) == mfs._UNIVERSE_COLUMNS
    assert len(df) > 0
    assert df["ticker"].is_unique


def test_universe_columns_are_never_missing_after_normalization():
    df = mfs._fetch_sp500_static_fallback()
    df = mfs._normalize_tickers(df)
    df["sector"] = df["sector"].fillna("Unclassified")
    df["sub_industry"] = df["sub_industry"].fillna("Unclassified")
    assert df["sector"].isna().sum() == 0
    assert df["sub_industry"].isna().sum() == 0


# -----------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------
def test_clean_dataset_filters_by_market_cap(raw_universe_like_df, config):
    strict_config = mfs.Config(min_market_cap=1e12)  # deliberately huge, should filter most/all out
    cleaned = mfs.clean_dataset(raw_universe_like_df, strict_config)
    assert (cleaned["market_cap"] >= 1e12).all()


def test_clean_dataset_handles_missing_optional_columns(config):
    """A dataset missing e.g. total_revenue should not crash clean_dataset."""
    df = pd.DataFrame({
        "ticker": ["A", "B"],
        "market_cap": [5e9, 6e9],
        "free_cashflow": [1e8, np.nan],
        "total_debt": [1e8, 2e8],
        "total_cash": [5e7, np.nan],
        "enterprise_value": [5e9, np.nan],
        "trailing_pe": [15, np.nan],
        # total_revenue intentionally omitted
    })
    cleaned = mfs.clean_dataset(df, config)
    assert "fcf_margin" in cleaned.columns
    assert cleaned["fcf_margin"].isna().all()  # can't compute without revenue -> NaN, not a crash


def test_clean_dataset_fills_missing_sector(config):
    df = pd.DataFrame({"ticker": ["A"], "market_cap": [5e9], "sector": [None]})
    cleaned = mfs.clean_dataset(df, config)
    assert cleaned["sector"].iloc[0] == "Unclassified"


# -----------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------
def test_compute_factor_scores_produces_expected_columns(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    for col in [
        "value_score", "quality_score", "growth_score", "composite_score",
        "overall_score_100", "overall_percentile",
        "value_percentile", "quality_percentile", "growth_percentile",
        "score_band", "rank",
    ]:
        assert col in scored.columns, f"missing column: {col}"


def test_overall_score_100_is_within_0_100(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    valid = scored["overall_score_100"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_percentiles_within_0_100(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    for col in ["value_percentile", "quality_percentile", "growth_percentile", "overall_percentile"]:
        valid = scored[col].dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), f"{col} out of range"


def test_rank_1_has_highest_composite_score(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    top_row = scored[scored["rank"] == 1].iloc[0]
    assert top_row["composite_score"] == scored["composite_score"].max()


def test_custom_factor_weights_change_ranking(raw_universe_like_df):
    cleaned_a = mfs.clean_dataset(raw_universe_like_df, mfs.Config())
    value_only = mfs.Config(value_weight=1.0, quality_weight=0.0, growth_weight=0.0)
    growth_only = mfs.Config(value_weight=0.0, quality_weight=0.0, growth_weight=1.0)
    scored_value = mfs.compute_factor_scores(cleaned_a, value_only)
    scored_growth = mfs.compute_factor_scores(cleaned_a, growth_only)
    # Different weighting schemes should (almost certainly, on random data)
    # produce a different #1 ranked stock.
    assert scored_value.iloc[0]["ticker"] != scored_growth.iloc[0]["ticker"] or \
        scored_value["composite_score"].tolist() != scored_growth["composite_score"].tolist()


def test_score_label_bands():
    assert mfs.score_label(95) == "Exceptional"
    assert mfs.score_label(85) == "Strong"
    assert mfs.score_label(75) == "Above Average"
    assert mfs.score_label(65) == "Average"
    assert mfs.score_label(55) == "Below Average"
    assert mfs.score_label(20) == "Weak"
    assert mfs.score_label(float("nan")) == "N/A"


# -----------------------------------------------------------------------
# Stock detail / explanations / risk flags
# -----------------------------------------------------------------------
def test_get_stock_detail_returns_expected_shape(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    ticker = scored.iloc[0]["ticker"]
    detail = mfs.get_stock_detail(scored, ticker)
    assert detail["ticker"] == ticker
    assert isinstance(detail["fundamentals"], dict)
    assert isinstance(detail["strengths"], list)
    assert isinstance(detail["concerns"], list)
    assert "risk_flags" in detail and "positive_flags" in detail["risk_flags"]


def test_get_stock_detail_raises_for_unknown_ticker(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    with pytest.raises(KeyError):
        mfs.get_stock_detail(scored, "NOT_A_REAL_TICKER")


def test_risk_flags_are_data_driven_not_invented():
    """A row with clean fundamentals should raise no risk flags."""
    row = pd.Series({
        "overall_score_100": 80, "debt_to_equity": 20, "free_cashflow": 5e8,
        "current_ratio": 2.0, "roe": 0.2, "gross_margin": 0.5, "revenue_growth": 0.05,
    })
    flags = mfs.get_risk_flags(row)
    assert flags["risk_flags"] == []


def test_risk_flags_detect_high_leverage():
    row = pd.Series({
        "overall_score_100": 50, "debt_to_equity": 300, "free_cashflow": 1e8,
        "current_ratio": 1.5, "roe": 0.1, "gross_margin": 0.4, "revenue_growth": 0.03,
    })
    flags = mfs.get_risk_flags(row)
    assert any("debt" in f.lower() for f in flags["risk_flags"])


def test_metric_glossary_is_populated():
    assert len(mfs.METRIC_GLOSSARY) >= 15
    assert "roe" in mfs.METRIC_GLOSSARY


# -----------------------------------------------------------------------
# Signal validation
# -----------------------------------------------------------------------
def test_evaluate_signal_returns_ic_and_spread(raw_universe_like_df, config):
    cleaned = mfs.clean_dataset(raw_universe_like_df, config)
    scored = mfs.compute_factor_scores(cleaned, config)
    metrics, decile_returns = mfs.evaluate_signal(scored)
    assert "information_coefficient" in metrics
    assert "quintile_spread" in metrics
    assert decile_returns is not None


def test_evaluate_signal_handles_insufficient_data():
    tiny_df = pd.DataFrame({
        "composite_score": [0.5, 0.6],
        "trailing_return_12m": [0.1, np.nan],
    })
    metrics, decile_returns = mfs.evaluate_signal(tiny_df)
    assert metrics == {}
    assert decile_returns is None
