# Multi-Factor Fundamental Screener & Dashboard

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amoghprasad11/Multi-Factor-Fundamental-Screener-Dashboard/blob/main/notebooks/screener_demo.ipynb)

A cross-sectional equity screener that ranks the S&P 500 on **Value**, **Quality**, and **Growth** factors using sector-neutral scoring, with both a polished interactive Plotly dashboard and a full Streamlit web app for searching, filtering, and drilling into individual stocks.

---

## Features

- Full S&P 500 universe construction with a **resilient multi-source fallback chain** (no more `HTTP Error 403` from Wikipedia scraping)
- Sector-neutral Value / Quality / Growth scoring across 15+ fundamental metrics
- An intuitive **0–100 overall score** and percentile ranking, in addition to the underlying z-scores
- Configurable factor weights (default: equal-weighted 1/3 Value, 1/3 Quality, 1/3 Growth)
- On-disk caching to avoid re-downloading the same ticker repeatedly
- A polished dark-theme interactive Plotly dashboard (`professional_dashboard.py`)
- A Streamlit app (`app.py`) with search, sector filtering, score/column sorting, a sortable results table, and per-stock drill-down
- Automatically generated, data-driven **"why did this stock rank here?"** strengths/concerns and risk flags for any individual stock
- A built-in metric glossary and score-band legend for anyone still learning fundamental analysis
- A pytest suite that runs with no network access (mocked data)

## How the Model Works

### Universe construction
The S&P 500 constituent list is built from a chain of sources, tried in order until one succeeds:
1. A maintained GitHub-hosted CSV mirror of S&P 500 membership (fast, stable, does not 403 in cloud environments).
2. Wikipedia, scraped via `requests` with an explicit browser User-Agent header (this — not the source itself — is what fixes the 403 Wikipedia frequently returns to Colab).
3. A small hardcoded snapshot of well-known large caps, used only as a last resort and logged loudly as degraded mode.

### Value methodology
Cheapness relative to earnings, book value, sales, EV, and cash flow: trailing P/E, Price/Book, Price/Sales, EV/EBITDA, EV/FCF, FCF Yield, Earnings Yield. Lower-is-better metrics are sign-flipped before combining.

### Quality methodology
Profitability, capital efficiency, and balance-sheet health: ROE, ROA, Operating Margin, Gross Margin, FCF Margin, Debt/Equity (inverted), Current Ratio, Quick Ratio.

### Growth methodology
Revenue Growth, Earnings Growth, and most-recent-quarter Earnings Growth (year-over-year).

> **Scope note:** a few commonly-requested ratios — true ROIC (which requires an invested-capital calculation from the full balance sheet), Net Debt/EBITDA, and Interest Coverage — are **not** computed here, because `yfinance`'s lightweight snapshot endpoint doesn't reliably expose the inputs needed across the full universe. Rather than fake these with a shaky proxy, they're simply omitted; this is documented directly in the code (see `clean_dataset()`).

### Composite scoring
Each raw metric is **sector-neutral z-scored** — a bank's P/E is only ever compared against other Financials, not against a software company's structurally different P/E. Metrics are combined into Value/Quality/Growth composites, then blended into an overall score using `Config.value_weight` / `quality_weight` / `growth_weight` (default: equal-weighted).

The composite is exposed two ways:
- `composite_score` — the raw sector-neutral z-score (kept for backward compatibility with `professional_dashboard.py`).
- `overall_score_100` — the **same ranking** expressed as a 0–100 percentile score, which is far easier to read than a raw z-score. Percentile rank is used instead of min-max scaling because one extreme outlier can't distort the whole scale.

| Score | Meaning |
|---|---|
| 90–100 | Exceptional |
| 80–89 | Strong |
| 70–79 | Above Average |
| 60–69 | Average |
| 50–59 | Below Average |
| Below 50 | Weak |

A missing metric doesn't disqualify a stock — its factor score is computed from whatever metrics *are* available for it.

### Data sources

| Source | Used for | API key required? |
|---|---|---|
| GitHub-hosted S&P 500 CSV | Primary universe list | No |
| Wikipedia (with proper headers) | Universe fallback | No |
| Yahoo Finance (`yfinance`) | Prices, market cap, all fundamentals | No |
| Financial Modeling Prep | Optional supplementary ROIC/FCF-yield | Yes (optional) |

---

## Installation

### Local

```bash
git clone https://github.com/amoghprasad11/Multi-Factor-Fundamental-Screener-Dashboard.git
cd Multi-Factor-Fundamental-Screener-Dashboard
pip install -r requirements.txt
```

### Google Colab (notebook)

Click the **Open in Colab** badge at the top of this README, or open [`notebooks/screener_demo.ipynb`](notebooks/screener_demo.ipynb) directly — it's a real, runnable notebook (not just a copy-paste script). Run the cells top to bottom; Cell 1 installs dependencies.

Alternatively, from a blank Colab notebook:

```python
!git clone https://github.com/amoghprasad11/Multi-Factor-Fundamental-Screener-Dashboard.git
%cd Multi-Factor-Fundamental-Screener-Dashboard
!pip install -q -r requirements.txt

from multi_factor_screener import run_pipeline, CONFIG
CONFIG.max_tickers = 25   # small test run first
results = run_pipeline(CONFIG)

from professional_dashboard import build_professional_dashboard
build_professional_dashboard(results).show()
```

Set `CONFIG.max_tickers = None` for a full S&P 500 run once the small test succeeds.

---

## Running the Streamlit App

```bash
streamlit run app.py
```

This opens the interactive web app locally with:
- Sidebar controls for universe size, minimum market cap, and request delay
- KPI cards (stocks screened, top stock, average score, information coefficient)
- The full interactive dashboard
- A searchable, filterable, sortable results table
- A per-stock detail view with factor breakdown, full fundamentals, a plain-language "why did this stock rank here?" explanation, and data-driven risk flags

**Running Streamlit from Colab:** Colab doesn't serve local web apps directly. Use a tunneling tool such as `localtunnel`:

```python
!npm install -g localtunnel
!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```

For local use, `streamlit run app.py` alone is enough.

---

## Configuration

All tunable parameters live in the `Config` dataclass in `multi_factor_screener.py`:

```python
CONFIG.max_tickers = None          # None = full S&P 500; int = capped test run
CONFIG.min_market_cap = 300_000_000
CONFIG.value_weight = 1/3          # factor weights, must sum to 1.0
CONFIG.quality_weight = 1/3
CONFIG.growth_weight = 1/3
CONFIG.cache_enabled = True        # on-disk cache, avoids re-downloading tickers
CONFIG.cache_ttl_hours = 24.0
CONFIG.request_delay = 0.25        # seconds between API calls
```

Switch between **test mode** (`max_tickers=25` or `100`) and **full mode** (`max_tickers=None`) by changing this one value.

---

## Project Structure

```
Multi-Factor-Fundamental-Screener-Dashboard/
│
├── multi_factor_screener.py     # data pipeline + scoring engine (Colab-cell-formatted)
├── professional_dashboard.py    # polished interactive Plotly dashboard
├── app.py                       # Streamlit app (imports the two files above; no duplicated logic)
├── requirements.txt
├── README.md
├── LICENSE
│
├── tests/
│   └── test_screener.py         # no-network pytest suite
│
├── notebooks/
│   └── screener_demo.ipynb      # real, runnable Colab notebook
│
└── data/
    └── .gitkeep
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/test_screener.py -v
```

All tests run against small synthetic/mocked DataFrames — no network access required.

---

## Limitations

- **Signal validation is a sanity check, not a backtest.** Free fundamentals APIs return current snapshot data, not point-in-time history, so `evaluate_signal()` measures the *concurrent* relationship between today's score and trailing 12-month returns — not a leakage-free forward-looking backtest. A genuine backtest needs point-in-time fundamentals (e.g. via WRDS/Compustat) matched to forward returns.
- True ROIC, Net Debt/EBITDA, and Interest Coverage are not computed (see Growth/Quality methodology above).
- Data quality depends entirely on what Yahoo Finance's `.info` endpoint returns for each ticker at request time; some tickers will have incomplete fundamentals, which the scoring model handles gracefully rather than excluding the stock entirely.
- Yahoo Finance is an unofficial, rate-limited API; very large or very frequent runs may trigger temporary throttling. Increase `Config.request_delay` if this happens.

## Disclaimer

This project is a **quantitative research and educational tool**. Composite scores, percentiles, strengths/concerns, and risk flags are generated purely from historical/current fundamental data using transparent, rule-based logic — they are **not investment advice** and do not guarantee future performance. Always do your own research and consult a qualified financial advisor before making investment decisions.

## License

MIT — see [LICENSE](LICENSE).
