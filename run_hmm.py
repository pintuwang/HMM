import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
import time
from datetime import datetime

# --- CONFIGURATION ---
TRAINING_DAYS = 365 * 10 
TICKERS = ["^VIX", "^VIX9D", "^GSPC"]

def get_market_data():
    # Attempt to download up to 3 times in case of 'database is locked' errors
    for attempt in range(3):
        try:
            # We add auto_adjust=True to handle the yfinance warning
            df = yf.download(TICKERS, period="12y", auto_adjust=True)['Close']
            if not df.empty and '^GSPC' in df.columns:
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5) # Wait 5 seconds before retrying
    
    if df.empty:
        raise ValueError("Could not download data from Yahoo Finance after 3 attempts.")

    df = df.dropna()
    # VIX9D - VIX: Positive = Panic, Negative = Calm
    df['spread'] = df['^VIX9D'] - df['^VIX']
    df['returns'] = np.log(df['^GSPC'] / df['^GSPC'].shift(1))
    return df.dropna()

def train_hmm(data):
    # Features: Log Returns and the VIX9D-VIX Spread
    X = data[['returns', 'spread']].values
    
    if len(X) < 100:
        raise ValueError(f"Not enough data points. Found {len(X)}")

    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Force State 1 to be the high-spread (Panic) state
    # We check the mean of the 'spread' feature (index 1)
    if model.means_[0][1] > model.means_[1][1]:
        model.means_ = model.means_[::-1]
        model.covars_ = model.covars_[::-1]
        model.transmat_ = model.transmat_[::-1, ::-1]

    probs = model.predict_proba(X)
    return probs[:, 1]

if __name__ == "__main__":
    try:
        df = get_market_data()
        panic_prob = train_hmm(df)
        
        # Prepare last 500 days
        plot_df = df.tail(500).copy()
        plot_df['prob'] = panic_prob[-500:] * 100
        
        latest = plot_df.iloc[-1]
        
        # Force everything to basic Python types for JSON
        output = {
            "date": str(latest.name.strftime('%Y-%m-%d')),
            "vix_close": float(latest['^VIX']),
            "vix_spread": float(latest['spread']),
            "panic_probability": float(latest['prob']),
            "dates": [str(d) for d in plot_df.index.strftime('%Y-%m-%d')],
            "vix_values": [float(v) for v in plot_df['^VIX']],
            "spread_values": [float(s) for s in plot_df['spread']],
            "prob_values": [float(p) for p in plot_df['prob']]
        }
        
        # Write to docs/chart_data.json
        with open('docs/chart_data.json', 'w') as f:
            json.dump(output, f)
        print("SUCCESS: Data written to docs/chart_data.json")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit(1) # Ensure the GitHub Action fails clearly
