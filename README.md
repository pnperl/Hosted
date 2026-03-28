# Hosted

## India Daily Market Bot

This repo provides `daily_stock_bot.py`, a simple bot focused on India markets.

### What it covers (last 24 hours)
- Overall Indian stock market news
- Major India/national business headlines
- Gujarat headlines
- Jamnagar headlines
- Rajkot headlines
- Nifty50 news + price outlook
- Crude, Gold, Silver India price outlook (INR converted)

### Run

```bash
pip install -r requirements.txt
python daily_stock_bot.py
```

### Useful options

```bash
python daily_stock_bot.py --hours 24 --news-limit 4
python daily_stock_bot.py --loop-every-hours 24
python daily_stock_bot.py --print-cron
```

### Daily schedule (every 24 hours)
- Continuous mode: `--loop-every-hours 24`
- Or cron mode (recommended):
  `0 8 * * * cd /workspace/Hosted && /usr/bin/python3 daily_stock_bot.py >> market_bot.log 2>&1`

### Notes
- The bot filters headlines to the latest window (`--hours`, default 24).
- If live feeds are blocked/unavailable, it automatically falls back to demo data so output still works.
- Educational signal tool only, not financial advice.
