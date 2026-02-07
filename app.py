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
# This command tells the server: "Run the function below ONLY if 24 hours have passed 
# since the last run. Otherwise, show the saved data."
@st.cache_data(ttl=86400) 
def fetch_data_once_a_day():
    # Your original configuration
    API_KEY = "3f767f5d796b470e96e93b4df8707df7"
    LEADERS = {
        "Putin": "Russia Putin", "Trump": "Donald Trump", "Modi": "Narendra Modi",
        "Kim Jong Un": "Kim Jong Un North Korea", "Xi Jinping": "Xi Jinping China",
        "Khamenei": "Ayatollah Khamenei", "Zelenskyy": "Zelenskyy Ukraine",
        "Macron": "Macron France", "Scholz": "Olaf Scholz Germany",
        "Netanyahu": "Netanyahu Israel"
    }
    
    analyzer = SentimentIntensityAnalyzer()
    results = []
    from_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

    # Fetching Data
    for name, query in LEADERS.items():
        url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=publishedAt&language=en&apiKey={API_KEY}"
        try:
            response = requests.get(url).json()
            articles = response.get('articles', [])
        except:
            articles = []

        if not articles: continue

        # Analyze Volume & Sentiment
        raw_volume = len(articles)
        compound_scores = [analyzer.polarity_scores((a.get('title') or "") + " " + (a.get('description') or ""))['compound'] for a in articles]
        avg_sentiment = np.mean(compound_scores) if compound_scores else 0
        
        # Calculate Coordinates
        aggression_score = ((avg_sentiment * -1) + 1) * 50 
        
        results.append({
            'Leader': name,
            'Aggression': aggression_score, 
            'Volume': raw_volume
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    if not df.empty:
        max_vol = df['Volume'].max()
        df['Influence'] = (df['Volume'] / max_vol) * 95 + 5
        
    return df

# ==========================================
# 2. THE VISUALIZATION (OUTPUT)
# ==========================================
def plot_chart(df):
    # Set dark background for the plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    # Quadrants
    ax.add_patch(patches.Rectangle((50, 50), 50, 50, color='#8B0000', alpha=0.3)) # War
    ax.add_patch(patches.Rectangle((0, 50), 50, 50, color='#006400', alpha=0.3))  # Peace
    ax.add_patch(patches.Rectangle((50, 0), 50, 50, color='#FF8C00', alpha=0.3))  # Rogue
    ax.add_patch(patches.Rectangle((0, 0), 50, 50, color='#696969', alpha=0.3))   # Isolated

    # Plot Points
    colors = cm.get_cmap('tab10', len(df))
    for i, row in df.iterrows():
        color = colors(i)
        ax.scatter(row['Aggression'], row['Influence'], color=color, s=300, edgecolors='white', alpha=0.9)
        ax.text(row['Aggression']+2, row['Influence'], row['Leader'].upper(), color='white', fontsize=9, fontweight='bold')

    # Labels and Grid
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel('PEACEFUL <------------------> HOSTILE', color='white')
    ax.set_ylabel('LOCAL <---------------------> GLOBAL', color='white')
    ax.set_title(f"Global Sentiment Watch (Updated Daily)", color='white')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    return fig

# ==========================================
# 3. THE WEBSITE LAYOUT
# ==========================================
st.set_page_config(page_title="Chaos Watcher", layout="centered")
st.title("🌍 Global Leaders Sentiment Analysis")
st.write("This chart updates automatically once every 24 hours to track global news sentiment.")

# Load data (This uses the cache!)
try:
    df = fetch_data_once_a_day()
    if not df.empty:
        st.pyplot(plot_chart(df))
        st.dataframe(df[['Leader', 'Aggression', 'Influence']])
    else:
        st.error("No data found. NewsAPI might be blocking cloud requests (Free Tier limitation).")
except Exception as e:
    st.error(f"An error occurred: {e}")