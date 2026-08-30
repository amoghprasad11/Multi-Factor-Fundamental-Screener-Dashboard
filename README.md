# Multi-Factor-Fundamental-Screener-Dashboard

A cross-sectional equity screener that ranks the S&P 500 on Value, Quality, and Growth factors using sector-neutral z-scoring, with an interactive research dashboard.

## Google Colab

The repository includes a polished Plotly dashboard in `professional_dashboard.py`.

After cloning the repository and running the screener pipeline so that `results` exists, use:

```python
from professional_dashboard import build_professional_dashboard

dashboard = build_professional_dashboard(results)
dashboard.show()
```

For the detailed ranked list underneath the dashboard:

```python
top25 = (
    results["data"]
    .sort_values("composite_score", ascending=False)
    .head(25)
    .copy()
)

display(top25[
    [
        "rank", "ticker", "name", "sector",
        "composite_score", "value_score",
        "quality_score", "growth_score"
    ]
])
```

## Full S&P 500

Set the universe cap to `None` before running the pipeline:

```python
CONFIG.max_tickers = None
```

The pipeline currently uses a free S&P 500 constituent source and yfinance for market/fundamental data. Network/API availability can affect how many companies return usable data.

## Dashboard Features

- Full-universe stock count and top-ranked stock KPI cards
- Value vs. Quality map with market-cap-sized bubbles
- Top 15 and Top 10 ranked stocks
- Sector-average composite scores
- Composite-score distribution
- Value / Quality / Growth factor comparison
- Number of stocks by sector
- Composite score vs. trailing 12-month return
- Interactive hover details
- Display-only clipping of extreme 12-month returns so the main relationship remains readable
- Built-in explanation panel for easier interpretation

## Important Research Note

The composite score is a quantitative ranking signal, not a guarantee of future performance. The current validation compares today's factor scores with trailing 12-month returns; it is not a leakage-free point-in-time forward backtest.

## License

MIT
