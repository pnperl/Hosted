import os

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

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
NEWS_API_SECRET_NAME = "NEWSAPI_KEY"


def get_newsapi_key():
    return st.secrets.get(NEWS_API_SECRET_NAME) or os.environ.get(NEWS_API_SECRET_NAME)


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
    today = datetime.now()
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
    latest_dates = history_df.groupby("Leader")["Date"].transform("max")
    current_df = (
        history_df[history_df["Date"] == latest_dates]
        .sort_values("Leader")
        .reset_index(drop=True)
    )

    return current_df, history_df


def analyze_articles(articles, analyzer):
    if not articles:
        return None

    raw_volume = len(articles)
    compound_scores = [
        analyzer.polarity_scores(
            (article.get("title") or "") + " " + (article.get("description") or "")
        )["compound"]
        for article in articles
    ]
    avg_sentiment = np.mean(compound_scores) if compound_scores else 0
    aggression_score = ((avg_sentiment * -1) + 1) * 50

    return {
        "Aggression": aggression_score,
        "Volume": raw_volume,
    }


# This command tells the server: "Run the function below ONLY if 24 hours have passed
# since the last run. Otherwise, show the saved data."
@st.cache_data(ttl=86400)
def fetch_data_once_a_day():
    api_key = get_newsapi_key()
    if not api_key:
        raise RuntimeError(
            f"Missing {NEWS_API_SECRET_NAME}. Add it to Streamlit secrets or the environment."
        )

    analyzer = SentimentIntensityAnalyzer()
    history_results = []

    today = datetime.now()

    for name, query in LEADERS.items():
        for weeks_ago in range(TAIL_WEEKS - 1, -1, -1):
            window_start = today - timedelta(days=(weeks_ago + 1) * WEEK_LENGTH_DAYS)
            window_end = today - timedelta(days=weeks_ago * WEEK_LENGTH_DAYS)
            from_date = window_start.strftime("%Y-%m-%d")
            to_date = window_end.strftime("%Y-%m-%d")
            url = (
                "https://newsapi.org/v2/everything?"
                f"q={query}&from={from_date}&to={to_date}&sortBy=publishedAt"
                f"&language=en&pageSize=100&apiKey={api_key}"
            )

            try:
                response = requests.get(url, timeout=15).json()
                articles = response.get("articles", [])
            except requests.RequestException:
                articles = []

            analyzed = analyze_articles(articles, analyzer)
            if not analyzed:
                continue

            history_results.append(
                {
                    "Leader": name,
                    "Date": to_date,
                    **analyzed,
                    "IsDemo": False,
                }
            )

    history_df = pd.DataFrame(history_results)
    if history_df.empty:
        return generate_demo_data()

    max_vol = history_df["Volume"].max()
    history_df["Influence"] = (history_df["Volume"] / max_vol) * 95 + 5
    history_df["Date"] = pd.to_datetime(history_df["Date"])

    latest_dates = history_df.groupby("Leader")["Date"].transform("max")
    current_df = (
        history_df[history_df["Date"] == latest_dates]
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
    ax.add_patch(patches.Rectangle((50, 50), 50, 50, color="#FF4D8D", alpha=0.12))  # War
    ax.add_patch(patches.Rectangle((0, 50), 50, 50, color="#00F5A0", alpha=0.10))  # Peace
    ax.add_patch(patches.Rectangle((50, 0), 50, 50, color="#FFC857", alpha=0.10))  # Rogue
    ax.add_patch(patches.Rectangle((0, 0), 50, 50, color="#8A8FB2", alpha=0.10))  # Isolated

    ax.axhline(50, color="#3BE7FF", linewidth=1.1, alpha=0.25, zorder=1)
    ax.axvline(50, color="#3BE7FF", linewidth=1.1, alpha=0.25, zorder=1)

    # Plot historical tails and current points
    colors = cm.get_cmap("cool", len(df))
    for i, row in df.iterrows():
        color = colors(i)
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

    # Labels and Grid
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

# Load data (This uses the cache!)
try:
    df, history_df = fetch_data_once_a_day()
    using_demo_data = bool(history_df.get("IsDemo", pd.Series([False])).all())
except Exception as e:
    st.warning(f"Live NewsAPI data is unavailable right now, so demo trend data is being shown instead. ({e})")
    df, history_df = generate_demo_data()
    using_demo_data = True

if not df.empty:
    if using_demo_data:
        st.info("Displaying demo leader history because live NewsAPI results were unavailable in this environment.")

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
    st.dataframe(df[["Leader", "Aggression", "Influence"]])
else:
    st.error("No live or demo data could be prepared.")
