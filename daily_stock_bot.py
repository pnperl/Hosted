#!/usr/bin/env python3
"""India-focused daily market bot with 24-hour headlines and simple outlooks."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from statistics import mean
from typing import Iterable
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

GOOGLE_NEWS_RSS_BASE_URL = "https://news.google.com/rss/search"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"

NEWS_QUERIES = {
    "Indian Market": "Indian stock market NSE BSE",
    "India National": "India business economy policy",
    "Gujarat": "Gujarat business industry",
    "Jamnagar": "Jamnagar industry business news",
    "Rajkot": "Rajkot business industry news",
    "Nifty50": "Nifty 50 stocks outlook",
}

NIFTY_SYMBOLS = [
    "^NSEI",
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "ITC.NS",
    "LT.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
]

COMMODITY_SYMBOLS = {
    "Crude Oil (Brent/WTI proxy)": "CL=F",  # USD per barrel
    "Gold": "GC=F",  # USD per troy ounce
    "Silver": "SI=F",  # USD per troy ounce
    "USD/INR": "INR=X",
}


@dataclass
class Headline:
    title: str
    published_at: datetime


@dataclass
class Quote:
    symbol: str
    price: float
    change_pct: float
    currency: str


@dataclass
class CommodityINR:
    name: str
    raw_usd_price: float
    inr_price: float
    unit: str
    change_pct: float
    outlook: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="India daily market/news bot")
    parser.add_argument("--hours", type=int, default=24, help="News lookback window in hours (default: 24)")
    parser.add_argument("--news-limit", type=int, default=4, help="Max headlines per section (default: 4)")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds (default: 15)")
    parser.add_argument(
        "--loop-every-hours",
        type=int,
        default=0,
        help="Run continuously every N hours (use 24 for daily). 0 = run once.",
    )
    parser.add_argument(
        "--print-cron",
        action="store_true",
        help="Print a cron example to run once every 24h.",
    )
    return parser.parse_args()


def fetch_google_news(query: str, hours: int, limit: int, timeout: int) -> list[Headline]:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)
    encoded_query = quote_plus(query)
    url = f"{GOOGLE_NEWS_RSS_BASE_URL}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    headlines: list[Headline] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        published_raw = (item.findtext("pubDate") or "").strip()
        if not title or not published_raw:
            continue
        try:
            published_at = parsedate_to_datetime(published_raw).astimezone(timezone.utc)
        except (TypeError, ValueError, AttributeError):
            continue
        if published_at < since:
            continue
        headlines.append(Headline(title=title, published_at=published_at))

    headlines.sort(key=lambda h: h.published_at, reverse=True)
    return headlines[:limit]


def fetch_quotes(symbols: Iterable[str], timeout: int) -> dict[str, Quote]:
    symbols = list(symbols)
    if not symbols:
        return {}

    params = {"symbols": ",".join(symbols)}
    response = requests.get(YAHOO_QUOTE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    out: dict[str, Quote] = {}
    for row in payload.get("quoteResponse", {}).get("result", []):
        symbol = row.get("symbol")
        price = row.get("regularMarketPrice")
        change_pct = row.get("regularMarketChangePercent")
        currency = row.get("currency") or "USD"
        if symbol is None or price is None or change_pct is None:
            continue
        out[symbol] = Quote(symbol=symbol, price=float(price), change_pct=float(change_pct), currency=currency)

    if not out:
        raise ValueError("No quote data returned")
    return out


def sentiment_outlook(texts: list[str], analyzer: SentimentIntensityAnalyzer) -> str:
    if not texts:
        return "Neutral"
    score = mean(analyzer.polarity_scores(text)["compound"] for text in texts)
    if score > 0.2:
        return "Positive"
    if score < -0.2:
        return "Cautious"
    return "Neutral"


def price_outlook(change_pct: float) -> str:
    if change_pct >= 1.2:
        return "Bullish"
    if change_pct <= -1.2:
        return "Bearish"
    return "Sideways"


def commodity_inr_outlook(quotes: dict[str, Quote]) -> list[CommodityINR]:
    usdinr = quotes.get("INR=X")
    if not usdinr:
        raise ValueError("USD/INR quote missing")

    fx = usdinr.price

    out: list[CommodityINR] = []
    crude = quotes.get("CL=F")
    gold = quotes.get("GC=F")
    silver = quotes.get("SI=F")

    if crude:
        inr_barrel = crude.price * fx
        out.append(
            CommodityINR(
                name="Crude Oil",
                raw_usd_price=crude.price,
                inr_price=inr_barrel,
                unit="₹/barrel",
                change_pct=crude.change_pct,
                outlook=price_outlook(crude.change_pct),
            )
        )
    if gold:
        inr_oz = gold.price * fx
        out.append(
            CommodityINR(
                name="Gold",
                raw_usd_price=gold.price,
                inr_price=inr_oz,
                unit="₹/oz",
                change_pct=gold.change_pct,
                outlook=price_outlook(gold.change_pct),
            )
        )
    if silver:
        inr_kg = silver.price * fx * 32.1507466
        out.append(
            CommodityINR(
                name="Silver",
                raw_usd_price=silver.price,
                inr_price=inr_kg,
                unit="₹/kg",
                change_pct=silver.change_pct,
                outlook=price_outlook(silver.change_pct),
            )
        )
    return out


def render_report(
    generated_at: datetime,
    section_news: dict[str, list[Headline]],
    section_outlook: dict[str, str],
    nifty_quotes: dict[str, Quote],
    commodity_rows: list[CommodityINR],
    fallback_notes: list[str],
    hours: int,
) -> str:
    lines = [
        "# India Daily Market Bot",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Coverage window: last {hours} hours",
        "",
        "## Market + Regional Headlines",
    ]

    for section, headlines in section_news.items():
        lines.append(f"### {section} (Outlook: {section_outlook.get(section, 'Neutral')})")
        if headlines:
            for h in headlines:
                lines.append(f"- {h.title}")
        else:
            lines.append("- No fresh headlines found in the selected window.")

    lines.extend(["", "## Nifty50 Price Snapshot + Outlook", "| Symbol | Price | Change % | Outlook |", "|---|---:|---:|---|"])

    for symbol in NIFTY_SYMBOLS:
        q = nifty_quotes.get(symbol)
        if not q:
            lines.append(f"| {symbol} | NA | NA | NA |")
            continue
        lines.append(f"| {symbol} | {q.price:,.2f} {q.currency} | {q.change_pct:+.2f}% | {price_outlook(q.change_pct)} |")

    lines.extend(["", "## India Commodity Prices + Outlook", "| Commodity | USD Price | India Price | Day Change | Outlook |", "|---|---:|---:|---:|---|"])
    for row in commodity_rows:
        lines.append(
            f"| {row.name} | ${row.raw_usd_price:,.2f} | {row.inr_price:,.2f} {row.unit} | {row.change_pct:+.2f}% | {row.outlook} |"
        )

    lines.extend(
        [
            "",
            "## Daily run",
            "- Run every 24 hours: `python daily_stock_bot.py`",
            "- Continuous mode: `python daily_stock_bot.py --loop-every-hours 24`",
        ]
    )

    if fallback_notes:
        lines.extend(["", "## Notes", *[f"- {n}" for n in fallback_notes]])

    return "\n".join(lines)


def build_demo_news(hours: int) -> tuple[dict[str, list[Headline]], dict[str, str]]:
    now = datetime.now(timezone.utc)
    section_news: dict[str, list[Headline]] = {}
    section_outlook: dict[str, str] = {}
    for section in NEWS_QUERIES:
        section_news[section] = [
            Headline(title=f"{section}: Demo headline (live feed unavailable)", published_at=now - timedelta(hours=min(2, hours))),
            Headline(title=f"{section}: Policy and market reaction under watch", published_at=now - timedelta(hours=min(6, hours))),
        ]
        section_outlook[section] = "Neutral"
    return section_news, section_outlook


def build_demo_quotes() -> tuple[dict[str, Quote], list[CommodityINR]]:
    quotes = {
        "^NSEI": Quote("^NSEI", 22350.0, 0.6, "INR"),
        "RELIANCE.NS": Quote("RELIANCE.NS", 2901.0, 0.9, "INR"),
        "TCS.NS": Quote("TCS.NS", 4150.0, -0.2, "INR"),
        "HDFCBANK.NS": Quote("HDFCBANK.NS", 1575.0, 0.3, "INR"),
        "ICICIBANK.NS": Quote("ICICIBANK.NS", 1180.0, 1.1, "INR"),
        "INFY.NS": Quote("INFY.NS", 1685.0, -0.7, "INR"),
        "ITC.NS": Quote("ITC.NS", 437.0, 0.4, "INR"),
        "LT.NS": Quote("LT.NS", 3700.0, 0.5, "INR"),
        "SBIN.NS": Quote("SBIN.NS", 790.0, 0.8, "INR"),
        "BHARTIARTL.NS": Quote("BHARTIARTL.NS", 1235.0, -0.1, "INR"),
        "CL=F": Quote("CL=F", 80.0, 0.7, "USD"),
        "GC=F": Quote("GC=F", 2190.0, 0.4, "USD"),
        "SI=F": Quote("SI=F", 25.5, -0.5, "USD"),
        "INR=X": Quote("INR=X", 83.2, 0.0, "INR"),
    }
    return quotes, commodity_inr_outlook(quotes)


def run_once(args: argparse.Namespace) -> str:
    analyzer = SentimentIntensityAnalyzer()
    now = datetime.now(timezone.utc)

    notes: list[str] = []

    # News sections
    section_news: dict[str, list[Headline]] = {}
    section_outlook: dict[str, str] = {}
    try:
        for section, query in NEWS_QUERIES.items():
            headlines = fetch_google_news(query=query, hours=args.hours, limit=args.news_limit, timeout=args.timeout)
            section_news[section] = headlines
            section_outlook[section] = sentiment_outlook([h.title for h in headlines], analyzer)
    except (requests.RequestException, ET.ParseError, ValueError) as exc:
        section_news, section_outlook = build_demo_news(args.hours)
        notes.append(f"Live news unavailable; demo headlines used. ({exc})")

    # Quotes
    try:
        quote_symbols = list(set(NIFTY_SYMBOLS + list(COMMODITY_SYMBOLS.values())))
        all_quotes = fetch_quotes(quote_symbols, timeout=args.timeout)
        nifty_quotes = {s: all_quotes[s] for s in NIFTY_SYMBOLS if s in all_quotes}
        commodity_rows = commodity_inr_outlook(all_quotes)
    except (requests.RequestException, ValueError, KeyError) as exc:
        all_quotes, commodity_rows = build_demo_quotes()
        nifty_quotes = {s: all_quotes[s] for s in NIFTY_SYMBOLS if s in all_quotes}
        notes.append(f"Live prices unavailable; demo prices used. ({exc})")

    return render_report(
        generated_at=now,
        section_news=section_news,
        section_outlook=section_outlook,
        nifty_quotes=nifty_quotes,
        commodity_rows=commodity_rows,
        fallback_notes=notes,
        hours=args.hours,
    )


def main() -> int:
    args = parse_args()

    if args.print_cron:
        print("Cron example (run daily at 08:00):")
        print("0 8 * * * cd /workspace/Hosted && /usr/bin/python3 daily_stock_bot.py >> market_bot.log 2>&1")
        return 0

    if args.loop_every_hours <= 0:
        print(run_once(args))
        return 0

    while True:
        print(run_once(args))
        print("\n--- waiting for next run ---\n")
        time.sleep(max(1, args.loop_every_hours) * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
