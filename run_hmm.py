import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import json
from datetime import datetime

# --- CONFIGURATION ---
TRAINING_DAYS = 365 * 10 
TICKERS = ["^VIX", "^VIX9D", "^GSPC"]

def get_market_data():
    df = yf.download(TICKERS, period="12y")['Close']
    df = df.dropna()
    # VIX9D - VIX: Positive = Panic, Negative = Calm
    df['spread'] = df['^VIX9D'] - df['^VIX']
    df['returns'] = np.log(df['^GSPC'] / df['^GSPC'].shift(1))
    return df.dropna()

def train_hmm(data):
    X = data[['returns', 'spread']].values
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    
    # Force State 1 to be the high-spread (Panic) state
    if model.means_[0][1] > model.means_[1][1]:
        model.means_ = model.means_[::-1]
        model.covars_ = model.covars_[::-1]
        model.transmat_ = model.transmat_[::-1, ::-1]

    probs = model.predict_proba(X)
    return probs[:, 1]

if __name__ == "__main__":
    df = get_market_data()
    panic_prob = train_hmm(df)
    
    # Prepare last 500 days
    plot_df = df.tail(500).copy()
    plot_df['prob'] = panic_prob[-500:] * 100
    
    latest = plot_df.iloc[-1]
    
    # FLAT STRUCTURE - Easiest for JS to read
    output = {
        "date": latest.name.strftime('%Y-%m-%d'),
        "vix_close": float(latest['^VIX']),
        "vix_spread": float(latest['spread']),
        "panic_probability": float(latest['prob']),
        "dates": plot_df.index.strftime('%Y-%m-%d').tolist(),
        "vix_values": plot_df['^VIX'].tolist(),
        "spread_values": plot_df['spread'].tolist(),
        "prob_values": plot_df['prob'].tolist()
    }
    
    with open('docs/chart_data.json', 'w') as f:
        json.dump(output, f)
    print("SUCCESS: Data written to docs/chart_data.json")
