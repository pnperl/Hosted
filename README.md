# Hosted

## Daily India Stock Bot (Gemini + Telegram)

This project now supports running a daily market brief bot via:

```bash
python src/main.py
```

### What it does
- Uses Gemini to generate a concise India market brief (simple language).
- Sends the brief to Telegram (if bot token/chat id are configured).
- Falls back to a safe built-in brief if Gemini is unavailable.

### Required environment variables
- `GEMINI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- Optional: `GEMINI_MODEL` (default: `gemini-1.5-flash`)

### Run daily every 24 hours
Cron example:

```bash
0 8 * * * cd /workspace/Hosted && /usr/bin/python3 src/main.py >> market_bot.log 2>&1
```

### Dependency note
The Gemini client is implemented with compatibility for both:
- new `google-genai` SDK
- legacy `google-generativeai` SDK

This avoids failures like `AttributeError: module 'google.generativeai' has no attribute 'GenerativeModel'` on differing runner images.
