# Hosted

This repository now includes a practical **Daily Stock Bot** you can run from the terminal to generate a market briefing.

## What it does

`daily_stock_bot.py` builds a daily watchlist report for chosen US tickers:

- Pulls ~30+ trading days of price history from Stooq
- Computes 1-day/1-week change and 30-day volatility
- Pulls recent Google News RSS headlines per ticker
- Scores news sentiment with VADER
- Produces a simple actionable recommendation (`BUY CANDIDATE`, `HOLD / REVIEW`, `WATCH / HIGH RISK`)

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with defaults:

```bash
python daily_stock_bot.py
```

Run with your own tickers:

```bash
python daily_stock_bot.py --tickers TSLA META AMD NFLX
```

Optional flags:

- `--news-limit 5` (default: 5)
- `--timeout 15` (default: 15 seconds)

## Notes

- If news fetch fails, the bot still generates a report using price action only.
- Output is a text/markdown briefing designed for daily review and quick decision support.
- This is a signal tool, **not financial advice**.
