import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime

# --- CONFIGURATION ---
TRAINING_DAYS = 365 * 10  # 10 Years for a solid 'Normal' baseline
TICKERS = ["^VIX", "^VIX9D", "^GSPC"]

def get_market_data():
    # Fetch data
    df = yf.download(TICKERS, period="12y")['Close']
    df = df.dropna()
    
    # CALCULATE SPREAD: Short-term minus Long-term
    # In Backwardation (Panic), VIX9D > VIX, so spread goes POSITIVE
    df['spread'] = df['^VIX9D'] - df['^VIX']
    
    # Calculate Log Returns for the HMM
    df['returns'] = np.log(df['^GSPC'] / df['^GSPC'].shift(1))
    return df.dropna()

def train_hmm(data):
    # Features: Log Returns and the VIX9D-VIX Spread
    X = data[['returns', 'spread']].values
    
    # 2-State Model: State 0 (Calm), State 1 (Panic)
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Ensure State 1 is always the 'Panic' state (higher spread mean)
    if model.means_[0][1] > model.means_[1][1]:
        # Swap if State 0 actually has the higher spread (backwardation)
        model.means_ = model.means_[::-1]
        model.covars_ = model.covars_[::-1]
        model.transmat_ = model.transmat_[::-1, ::-1]

    # Get Probabilities
    probs = model.predict_proba(X)
    panic_prob = probs[:, 1]  # Probability of State 1 (Panic)
    
    return panic_prob, model

def save_to_json(df, panic_prob):
    # Prepare the last 500 days for the chart
    plot_df = df.tail(500).copy()
    plot_df['panic_prob'] = panic_prob[-500:] * 100  # Scale to 0-100
    
    latest = plot_df.iloc[-1]
    
    output = {
        "summary": {
            "date": latest.name.strftime('%Y-%m-%d'),
            "vix_close": round(latest['^VIX'], 2),
            "vix_spread": round(latest['spread'], 2),
            "panic_probability": round(latest['panic_prob'], 2),
            "regime": "HIGH-VOLATILITY" if latest['panic_prob'] > 70 else "CALM"
        },
        "chart_data": {
            "labels": plot_df.index.strftime('%Y-%m-%d').tolist(),
            "vix": plot_df['^VIX'].tolist(),
            "spread": plot_df['spread'].tolist(),
            "probability": plot_df['panic_prob'].tolist()
        }
    }
    
    with open('chart_data.json', 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    df = get_market_data()
    panic_prob, model = train_hmm(df)
    save_to_json(df, panic_prob)
    print("Market Regime Analysis Complete.")
