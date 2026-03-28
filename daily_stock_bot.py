#!/usr/bin/env python3
"""Daily stock bot: fetches recent stock data + sentiment and prints a practical briefing."""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from statistics import mean, pstdev
from typing import Iterable
from urllib.parse import quote_plus

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

STOOQ_DAILY_CSV = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
GOOGLE_NEWS_RSS_BASE_URL = "https://news.google.com/rss/search"


@dataclass
class StockSnapshot:
    symbol: str
    price: float
    day_change_pct: float
    week_change_pct: float
    volatility_30d_pct: float
    momentum_score: float
    sentiment_score: float
    headlines: list[str]
    recommendation: str
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a daily stock bot briefing")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
        help="Ticker symbols to analyze (default: AAPL MSFT NVDA AMZN GOOGL)",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=5,
        help="How many recent headlines to consider per ticker",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="HTTP timeout in seconds",
    )
    return parser.parse_args()


def fetch_stooq_closes(symbol: str, timeout: int = 15) -> list[float]:
    url = STOOQ_DAILY_CSV.format(symbol=symbol.lower())
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    rows = list(csv.DictReader(io.StringIO(response.text)))
    closes: list[float] = []
    for row in rows:
        close_raw = (row.get("Close") or "").strip()
        if not close_raw:
            continue
        try:
            closes.append(float(close_raw))
        except ValueError:
            continue

    if len(closes) < 31:
        raise ValueError(f"Not enough daily history for {symbol}")
    return closes


def fetch_google_news_headlines(symbol: str, limit: int, timeout: int = 15) -> list[str]:
    query = quote_plus(f"{symbol} stock")
    url = f"{GOOGLE_NEWS_RSS_BASE_URL}?q={query}&hl=en-US&gl=US&ceid=US:en"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # Light-weight RSS parsing without external deps.
    text = response.text
    parts = text.split("<item>")[1:]
    headlines: list[tuple[datetime, str]] = []

    for item in parts:
        title = _extract_xml_tag(item, "title")
        pub_date = _extract_xml_tag(item, "pubDate")
        if not title:
            continue

        parsed_date = None
        if pub_date:
            try:
                parsed_date = parsedate_to_datetime(pub_date).astimezone(timezone.utc)
            except (TypeError, ValueError):
                parsed_date = None

        headlines.append((parsed_date or datetime.min.replace(tzinfo=timezone.utc), _clean_rss_text(title)))

    headlines.sort(key=lambda pair: pair[0], reverse=True)
    return [headline for _, headline in headlines[:limit]]


def _extract_xml_tag(xml_text: str, tag: str) -> str:
    start = f"<{tag}>"
    end = f"</{tag}>"
    if start not in xml_text or end not in xml_text:
        return ""
    return xml_text.split(start, 1)[1].split(end, 1)[0].strip()


def _clean_rss_text(value: str) -> str:
    # Minimal cleanup for entities commonly seen in RSS titles.
    return (
        value.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )


def build_snapshot(symbol: str, analyzer: SentimentIntensityAnalyzer, news_limit: int, timeout: int) -> StockSnapshot:
    closes = fetch_stooq_closes(symbol=symbol, timeout=timeout)

    latest = closes[-1]
    prev_day = closes[-2]
    prev_week = closes[-6]
    closes_30d = closes[-30:]

    day_change_pct = _pct_change(latest, prev_day)
    week_change_pct = _pct_change(latest, prev_week)
    volatility_30d_pct = _volatility_pct(closes_30d)

    headlines: list[str] = []
    sentiment_score = 0.0
    try:
        headlines = fetch_google_news_headlines(symbol=symbol, limit=news_limit, timeout=timeout)
        if headlines:
            sentiment_score = mean(analyzer.polarity_scores(h)["compound"] for h in headlines)
    except requests.RequestException:
        # Graceful degradation: still provide price-based recommendation.
        pass

    momentum_score = _clamp(
        (week_change_pct * 2.0) - (volatility_30d_pct * 0.8) + (sentiment_score * 20.0),
        -100,
        100,
    )

    recommendation = _recommendation(momentum_score, day_change_pct, volatility_30d_pct)
    note = ""
    if not headlines:
        note = "News unavailable; recommendation based on price action only."

    return StockSnapshot(
        symbol=symbol.upper(),
        price=latest,
        day_change_pct=day_change_pct,
        week_change_pct=week_change_pct,
        volatility_30d_pct=volatility_30d_pct,
        momentum_score=momentum_score,
        sentiment_score=sentiment_score,
        headlines=headlines,
        recommendation=recommendation,
        note=note,
    )


def build_demo_snapshot(symbol: str) -> StockSnapshot:
    """Offline fallback profile so the bot remains useful in restricted environments."""
    symbol = symbol.upper()
    seed = sum(ord(c) for c in symbol)
    base_price = 80 + (seed % 220) + ((seed % 13) / 10)
    day_change_pct = ((seed % 9) - 4) * 0.7
    week_change_pct = ((seed % 17) - 8) * 0.9
    volatility_30d_pct = 3.5 + (seed % 8) * 0.8
    sentiment_score = ((seed % 11) - 5) / 10
    momentum_score = _clamp((week_change_pct * 2.0) - (volatility_30d_pct * 0.8) + (sentiment_score * 20.0), -100, 100)
    recommendation = _recommendation(momentum_score, day_change_pct, volatility_30d_pct)

    headlines = [
        f"{symbol} earnings outlook update and analyst commentary",
        f"{symbol} sector demand trends remain in focus",
        f"{symbol} valuation debate continues ahead of next session",
    ]

    return StockSnapshot(
        symbol=symbol,
        price=base_price,
        day_change_pct=day_change_pct,
        week_change_pct=week_change_pct,
        volatility_30d_pct=volatility_30d_pct,
        momentum_score=momentum_score,
        sentiment_score=sentiment_score,
        headlines=headlines,
        recommendation=recommendation,
        note="Demo profile used because live market sources were unreachable in this environment.",
    )


def _pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100


def _volatility_pct(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    avg = mean(values)
    if avg == 0:
        return 0.0
    return (pstdev(values) / avg) * 100


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _recommendation(momentum: float, day_change: float, vol: float) -> str:
    if momentum >= 25 and day_change >= -1.5 and vol <= 7:
        return "BUY CANDIDATE"
    if momentum <= -25 or vol >= 12:
        return "WATCH / HIGH RISK"
    return "HOLD / REVIEW"


def render_report(snapshots: list[StockSnapshot]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Daily Stock Bot Briefing",
        f"Generated: {now}",
        "",
        "| Ticker | Price | 1D % | 1W % | 30D Vol % | Sentiment | Momentum | Recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for s in snapshots:
        lines.append(
            f"| {s.symbol} | ${s.price:,.2f} | {s.day_change_pct:+.2f}% | {s.week_change_pct:+.2f}% | "
            f"{s.volatility_30d_pct:.2f}% | {s.sentiment_score:+.2f} | {s.momentum_score:+.1f} | {s.recommendation} |"
        )

    lines.append("")
    lines.append("## Top headlines")
    for s in snapshots:
        lines.append(f"### {s.symbol}")
        if s.note:
            lines.append(f"- _{s.note}_")
        if s.headlines:
            for headline in s.headlines[:3]:
                lines.append(f"- {headline}")
        else:
            lines.append("- No recent headlines fetched.")

    best = max(snapshots, key=lambda s: s.momentum_score)
    worst = min(snapshots, key=lambda s: s.momentum_score)
    lines.extend(
        [
            "",
            "## Action summary",
            f"- Strongest momentum: **{best.symbol}** ({best.momentum_score:+.1f}).",
            f"- Weakest momentum: **{worst.symbol}** ({worst.momentum_score:+.1f}).",
            "- Use this as a watchlist signal, not financial advice.",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    analyzer = SentimentIntensityAnalyzer()

    snapshots: list[StockSnapshot] = []
    failures: list[str] = []

    for ticker in args.tickers:
        try:
            snapshots.append(
                build_snapshot(
                    symbol=ticker,
                    analyzer=analyzer,
                    news_limit=args.news_limit,
                    timeout=args.timeout,
                )
            )
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"{ticker.upper()}: {exc}")

    if not snapshots:
        print("Live sources were unavailable. Falling back to demo market profiles.\n")
        snapshots = [build_demo_snapshot(ticker) for ticker in args.tickers]

    print(render_report(snapshots))

    if failures:
        print("\n## Partial failures")
        for failure in failures:
            print(f"- {failure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
