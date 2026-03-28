from __future__ import annotations

from datetime import datetime, timezone

from gemini_client import get_gemini_prediction
from telegram_client import send_telegram_message


def fallback_brief() -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "India Daily Market Brief (Fallback)\n"
        f"Updated: {now}\n"
        "- Indian equities: mixed-to-stable tone; watch Nifty support/resistance levels.\n"
        "- National headlines: monitor RBI policy signals and inflation commentary.\n"
        "- Gujarat: industrial capex and export-linked sectors remain in focus.\n"
        "- Jamnagar: refinery and energy chain updates can drive local sentiment.\n"
        "- Rajkot: engineering/MSME order flow remains a key local indicator.\n"
        "- Nifty50 outlook: sideways with stock-specific opportunities.\n"
        "- Crude outlook (India): mildly firm prices can pressure OMC margins.\n"
        "- Gold outlook (India): resilient on global risk and USD volatility.\n"
        "- Silver outlook (India): volatile, follows industrial + precious metals trend."
    )


def main() -> None:
    try:
        message = get_gemini_prediction()
    except Exception as exc:
        message = f"{fallback_brief()}\n\nNote: Gemini unavailable ({exc})."

    sent = False
    try:
        sent = send_telegram_message(message)
    except Exception as exc:
        print(f"Telegram send failed: {exc}")

    print(message)
    if sent:
        print("\nTelegram: message sent")
    else:
        print("\nTelegram: skipped (missing credentials or send failed)")


if __name__ == "__main__":
    main()
