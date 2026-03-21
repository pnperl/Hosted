import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
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
API_KEY = "3f767f5d796b470e96e93b4df8707df7"


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
                f"&language=en&pageSize=100&apiKey={API_KEY}"
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
                }
            )

    history_df = pd.DataFrame(history_results)
    if history_df.empty:
        return pd.DataFrame(), pd.DataFrame()

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
    # Set dark background for the plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    # Quadrants
    ax.add_patch(patches.Rectangle((50, 50), 50, 50, color="#8B0000", alpha=0.3))  # War
    ax.add_patch(patches.Rectangle((0, 50), 50, 50, color="#006400", alpha=0.3))  # Peace
    ax.add_patch(patches.Rectangle((50, 0), 50, 50, color="#FF8C00", alpha=0.3))  # Rogue
    ax.add_patch(patches.Rectangle((0, 0), 50, 50, color="#696969", alpha=0.3))  # Isolated

    # Plot historical tails and current points
    colors = cm.get_cmap("tab10", len(df))
    for i, row in df.iterrows():
        color = colors(i)
        leader_history = history_df[history_df["Leader"] == row["Leader"]].sort_values("Date")

        if len(leader_history) > 1:
            ax.plot(
                leader_history["Aggression"],
                leader_history["Influence"],
                color=color,
                linewidth=2,
                alpha=0.4,
                zorder=2,
            )
            ax.scatter(
                leader_history.iloc[:-1]["Aggression"],
                leader_history.iloc[:-1]["Influence"],
                color=[color],
                s=70,
                alpha=0.25,
                zorder=3,
            )

        ax.scatter(
            row["Aggression"],
            row["Influence"],
            color=color,
            s=300,
            edgecolors="white",
            alpha=0.95,
            zorder=4,
        )
        ax.text(
            row["Aggression"] + 2,
            row["Influence"],
            row["Leader"].upper(),
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    # Labels and Grid
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("PEACEFUL <------------------> HOSTILE", color="white")
    ax.set_ylabel("LOCAL <---------------------> GLOBAL", color="white")
    ax.set_title(
        f"Global Sentiment Watch (Updated Daily, {TAIL_WEEKS}-Week Tails)",
        color="white",
    )
    ax.grid(True, linestyle=":", alpha=0.3)

    return fig


# ==========================================
# 3. THE WEBSITE LAYOUT
# ==========================================
st.set_page_config(page_title="Chaos Watcher", layout="centered")
st.title("🌍 Global Leaders Sentiment Analysis")
st.write(
    "This chart updates automatically once every 24 hours to track global news sentiment, "
    "including a rolling weekly tail covering the last 4 weeks for each leader."
)

# Load data (This uses the cache!)
try:
    df, history_df = fetch_data_once_a_day()
    if not df.empty:
        st.pyplot(plot_chart(df, history_df))
        st.dataframe(df[["Leader", "Aggression", "Influence"]])
    else:
        st.error("No data found. NewsAPI might be blocking cloud requests (Free Tier limitation).")
except Exception as e:
    st.error(f"An error occurred: {e}")
