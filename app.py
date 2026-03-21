import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================
# 1. THE "ONCE A DAY" LOGIC (CACHE)
# ==========================================
LEADERS = {
    "Putin": "Russia Putin",
    "Trump": "Donald Trump",
    "Modi": "Narendra Modi",
    "Kim Jong Un": "Kim Jong Un North Korea",
    "Xi Jinping": "Xi Jinping China",
    "Khamenei": "Ayatollah Khamenei",
    "Zelenskyy": "Zelenskyy Ukraine",
    "Macron": "Macron France",
    "Scholz": "Olaf Scholz Germany",
    "Netanyahu": "Netanyahu Israel",
}

TAIL_WEEKS = 4
WEEK_LENGTH_DAYS = 7
GOOGLE_NEWS_RSS_BASE_URL = "https://news.google.com/rss/search"


def clean_text(value):
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def generate_demo_data():
    base_profiles = {
        "Putin": (78, 84),
        "Trump": (64, 92),
        "Modi": (42, 73),
        "Kim Jong Un": (88, 58),
        "Xi Jinping": (54, 90),
        "Khamenei": (81, 61),
        "Zelenskyy": (37, 76),
        "Macron": (34, 68),
        "Scholz": (29, 63),
        "Netanyahu": (74, 71),
    }

    history_rows = []
    today = datetime.now(timezone.utc)
    for leader in LEADERS:
        aggression_base, influence_base = base_profiles[leader]
        for weeks_ago in range(TAIL_WEEKS - 1, -1, -1):
            drift = TAIL_WEEKS - weeks_ago - 1
            history_rows.append(
                {
                    "Leader": leader,
                    "Date": today - timedelta(days=weeks_ago * WEEK_LENGTH_DAYS),
                    "Aggression": max(0, min(100, aggression_base + (drift * 2.8) - 4.2)),
                    "Influence": max(5, min(100, influence_base + (drift * 1.7) - 2.5)),
                    "Volume": max(10, int(influence_base + (drift * 3))),
                    "IsDemo": True,
                }
            )

    history_df = pd.DataFrame(history_rows).sort_values(["Leader", "Date"]).reset_index(drop=True)

    # FIX #5: Use groupby.tail(1) instead of equality datetime match
    current_df = (
        history_df.sort_values("Date")
        .groupby("Leader")
        .tail(1)
        .sort_values("Leader")
        .reset_index(drop=True)
    )

    return current_df, history_df


def analyze_articles(articles, analyzer):
    if not articles:
        return None

    raw_volume = len(articles)
    compound_scores = [
        analyzer.polarity_scores(f"{article['title']} {article['description']}")["compound"]
        for article in articles
    ]
    avg_sentiment = np.mean(compound_scores) if compound_scores else 0

    # FIX #3: Clamp aggression score to [0, 100] in case avg_sentiment ever
    # drifts outside VADER's guaranteed [-1, +1] range (e.g. after refactor).
    aggression_score = max(0, min(100, ((avg_sentiment * -1) + 1) * 50))

    return {
        "Aggression": aggression_score,
        "Volume": raw_volume,
    }


def fetch_google_news_articles(query):
    encoded_query = quote_plus(f"{query} when:{TAIL_WEEKS * WEEK_LENGTH_DAYS}d")
    url = (
        f"{GOOGLE_NEWS_RSS_BASE_URL}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
    )

    response = requests.get(url, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    articles = []
    for item in root.findall("./channel/item"):
        published_at = item.findtext("pubDate", default="")
        try:
            published_dt = parsedate_to_datetime(published_at).astimezone(timezone.utc)
        except (TypeError, ValueError, AttributeError):
            continue

        articles.append(
            {
                "title": clean_text(item.findtext("title", default="")),
                "description": clean_text(item.findtext("description", default="")),
                "published_at": published_dt,
            }
        )

    return articles


# This command tells the server: "Run the function below ONLY if 24 hours have passed
# since the last run. Otherwise, show the saved data."
@st.cache_data(ttl=86400)
def fetch_data_once_a_day():
    analyzer = SentimentIntensityAnalyzer()
    history_results = []
    today = datetime.now(timezone.utc)
    missing_leaders = []  # FIX #2: track leaders with zero articles across all windows

    for name, query in LEADERS.items():
        try:
            articles = fetch_google_news_articles(query)
        except (requests.RequestException, ET.ParseError) as e:
            # FIX #8: Surface per-leader fetch errors in a structured way
            st.warning(f"Could not fetch news for {name}: {e}")
            articles = []

        leader_had_data = False
        for weeks_ago in range(TAIL_WEEKS - 1, -1, -1):
            window_start = today - timedelta(days=(weeks_ago + 1) * WEEK_LENGTH_DAYS)
            # FIX #6: Add 1 second to window_end for the most-recent window so
            # articles published right at fetch-time are not excluded.
            raw_window_end = today - timedelta(days=weeks_ago * WEEK_LENGTH_DAYS)
            window_end = raw_window_end + (timedelta(seconds=1) if weeks_ago == 0 else timedelta(0))

            window_articles = [
                article
                for article in articles
                if window_start <= article["published_at"] < window_end
            ]

            analyzed = analyze_articles(window_articles, analyzer)
            if not analyzed:
                # FIX #2: Insert a neutral placeholder row so the leader always
                # appears on the chart, even when a week window has no coverage.
                history_results.append(
                    {
                        "Leader": name,
                        "Date": raw_window_end,
                        "Aggression": 50.0,   # neutral
                        "Volume": 0,
                        "IsDemo": False,
                    }
                )
                continue

            leader_had_data = True
            history_results.append(
                {
                    "Leader": name,
                    "Date": raw_window_end,
                    **analyzed,
                    "IsDemo": False,
                }
            )

        if not leader_had_data:
            missing_leaders.append(name)

    history_df = pd.DataFrame(history_results)
    if history_df.empty:
        return generate_demo_data()

    if missing_leaders:
        st.info(
            f"No live articles found for: {', '.join(missing_leaders)}. "
            "Their positions are shown at neutral (50) aggression."
        )

    # FIX #4: Normalise Influence per-leader (relative to each leader's own
    # max volume) so a historical spike for one leader does not compress
    # every other leader's current-week influence score.
    def normalize_influence(group):
        leader_max = group["Volume"].max()
        if leader_max == 0:
            group["Influence"] = 5.0
        else:
            group["Influence"] = (group["Volume"] / leader_max) * 95 + 5
        return group

    history_df = history_df.groupby("Leader", group_keys=False).apply(normalize_influence)

    # FIX #5: Use groupby.tail(1) instead of equality datetime match
    current_df = (
        history_df.sort_values("Date")
        .groupby("Leader")
        .tail(1)
        .sort_values("Leader")
        .reset_index(drop=True)
    )

    return current_df, history_df.sort_values(["Leader", "Date"]).reset_index(drop=True)


# ==========================================
# 2. THE VISUALIZATION (OUTPUT)
# ==========================================
def plot_chart(df, history_df):
    # Set futuristic background for the plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="#050816")
    ax.set_facecolor("#050816")

    background_gradient = np.linspace(0, 1, 256).reshape(1, -1)
    neon_cmap = LinearSegmentedColormap.from_list(
        "neon_grid", ["#050816", "#0A1030", "#111B4D", "#1D2D6B"]
    )
    ax.imshow(
        background_gradient,
        extent=(0, 100, 0, 100),
        origin="lower",
        aspect="auto",
        cmap=neon_cmap,
        alpha=0.55,
        zorder=0,
    )

    # Quadrants
    ax.add_patch(patches.Rectangle((50, 50), 50, 50, color="#FF4D8D", alpha=0.12))
    ax.add_patch(patches.Rectangle((0, 50), 50, 50, color="#00F5A0", alpha=0.10))
    ax.add_patch(patches.Rectangle((50, 0), 50, 50, color="#FFC857", alpha=0.10))
    ax.add_patch(patches.Rectangle((0, 0), 50, 50, color="#8A8FB2", alpha=0.10))

    ax.axhline(50, color="#3BE7FF", linewidth=1.1, alpha=0.25, zorder=1)
    ax.axvline(50, color="#3BE7FF", linewidth=1.1, alpha=0.25, zorder=1)

    # FIX #11: cm.get_cmap is deprecated — use matplotlib.colormaps instead
    colors = plt.colormaps["cool"].resampled(len(df))

    # FIX #7: Use enumerate so color index is always sequential 0..N-1,
    # regardless of the DataFrame's actual index values.
    for idx, (_, row) in enumerate(df.iterrows()):
        color = colors(idx)
        leader_history = history_df[history_df["Leader"] == row["Leader"]].sort_values("Date")

        if len(leader_history) > 1:
            ax.plot(
                leader_history["Aggression"],
                leader_history["Influence"],
                color=color,
                linewidth=2.6,
                alpha=0.55,
                zorder=2,
            )
            ax.scatter(
                leader_history.iloc[:-1]["Aggression"],
                leader_history.iloc[:-1]["Influence"],
                color=[color],
                s=80,
                alpha=0.30,
                zorder=3,
            )

        ax.scatter(
            row["Aggression"],
            row["Influence"],
            color=color,
            s=520,
            alpha=0.18,
            linewidths=0,
            zorder=4,
        )
        ax.scatter(
            row["Aggression"],
            row["Influence"],
            color=color,
            s=280,
            edgecolors="#DFFBFF",
            linewidths=1.4,
            alpha=0.98,
            zorder=5,
        )
        ax.text(
            row["Aggression"] + 2,
            row["Influence"],
            row["Leader"].upper(),
            color="#E9FCFF",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("PEACEFUL <------------------> HOSTILE", color="#9CF6FF")
    ax.set_ylabel("LOCAL <---------------------> GLOBAL", color="#9CF6FF")
    ax.set_title(
        f"Global Sentiment Watch (Updated Daily, {TAIL_WEEKS}-Week Tails)",
        color="white",
    )
    ax.tick_params(colors="#9CF6FF")
    ax.grid(True, linestyle=":", alpha=0.18, color="#3BE7FF")

    for spine in ax.spines.values():
        spine.set_color("#3BE7FF")
        spine.set_alpha(0.25)

    return fig


FUTURISTIC_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top, #16224f 0%, #091126 35%, #030712 100%);
    color: #e7fbff;
}
[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}
.hero-panel {
    padding: 1.4rem 1.6rem;
    border: 1px solid rgba(59, 231, 255, 0.28);
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(9, 17, 38, 0.92), rgba(22, 34, 79, 0.75));
    box-shadow: 0 0 24px rgba(59, 231, 255, 0.16);
    margin-bottom: 1rem;
}
.hero-kicker {
    color: #3BE7FF;
    text-transform: uppercase;
    letter-spacing: 0.18rem;
    font-size: 0.78rem;
    margin-bottom: 0.35rem;
}
.hero-title {
    color: #F4FBFF;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
}
.hero-copy {
    color: #C8F7FF;
    margin-top: 0.65rem;
    line-height: 1.5;
}
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 1.4rem;
}
.metric-card {
    padding: 0.9rem 1rem;
    border-radius: 16px;
    background: rgba(5, 8, 22, 0.72);
    border: 1px solid rgba(156, 246, 255, 0.18);
}
.metric-label {
    color: #7CEEFF;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12rem;
}
.metric-value {
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.15rem;
}
</style>
"""


# ==========================================
# 3. THE WEBSITE LAYOUT
# ==========================================
st.set_page_config(page_title="Chaos Watcher", layout="centered")
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-kicker">Neural Geopolitics Feed</div>
        <h1 class="hero-title">🌍 Global Leaders Sentiment Analysis</h1>
        <p class="hero-copy">
            Futuristic signal mapping for geopolitical coverage, refreshed every 24 hours with
            a rolling weekly tail covering the last 4 weeks for each leader.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df, history_df = fetch_data_once_a_day()
    # FIX #1: Use column membership check instead of DataFrame.get(), which is
    # unreliable for this purpose and would incorrectly flag live data as demo
    # when the IsDemo column is absent.
    using_demo_data = history_df["IsDemo"].all() if "IsDemo" in history_df.columns else False
except Exception as e:
    st.warning(
        "Google News RSS is unavailable right now, so demo trend data is being shown instead. "
        f"({e})"
    )
    df, history_df = generate_demo_data()
    using_demo_data = True

if not df.empty:
    if using_demo_data:
        st.info("Displaying demo leader history because live Google News RSS results were unavailable in this environment.")

    strongest_signal = df.loc[df["Influence"].idxmax(), "Leader"]
    highest_alert = df.loc[df["Aggression"].idxmax(), "Leader"]
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card">
                <div class="metric-label">Tracked Leaders</div>
                <div class="metric-value">{len(df)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Strongest Reach</div>
                <div class="metric-value">{strongest_signal}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Highest Alert</div>
                <div class="metric-value">{highest_alert}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(plot_chart(df, history_df))
    # FIX #10: Round display scores to 1 decimal place for readability
    st.dataframe(df[["Leader", "Aggression", "Influence"]].round(1))
else:
    st.error("No live or demo data could be prepared.")
