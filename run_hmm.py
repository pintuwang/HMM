# run_hmm.py - FINAL WORKING VERSION

import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from datetime import date, timedelta
import os
import warnings
import json
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CHART_DATA_FILE = 'docs/chart_data.json'
CHART_PERIOD_DAYS = 365 * 1 # Last 1 year for display
N_COMPONENTS = 2 # 2 Regimes: Calm and Panic
TRAINING_DAYS = 365 * 3 # Use 3 years of data for training

# *** FIX: Replaced non-functional ^VXST with ^VIX3M ***
TICKERS = ["^GSPC", "^VIX", "^VIX3M"] 

def fetch_historical_data(tickers, period_days):
    """Fetches historical data for multiple tickers using yfinance."""
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)
    
    print(f"DEBUG: Fetching data for {', '.join(tickers)} from {start_date} to {end_date}...")
    
    try:
        # Fetch the data and keep only the 'Close' prices
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        return data.dropna()
    except Exception as e:
        print(f"ERROR: yfinance data fetch failed: {e}", file=sys.stderr)
        return pd.DataFrame()


def train_and_predict_hmm(df):
    """Trains the HMM and calculates the probability of the Panic Regime."""
    
    if len(df) < 50:
        print(f"ERROR: Not enough data points ({len(df)}) for HMM training.", file=sys.stderr)
        return None
    
    # X now includes Log_Return and the VIX_Spread
    X = df[['Log_Return', 'VIX_Spread']].values

    # Initialize the HMM
    model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type="diag", n_iter=500, tol=0.0001)
    
    # Initialize means and covariances (re-used from previous turns)
    model.means_ = np.array([[0.0003, 1.5], [-0.0005, -2.0]])
    model.covars_ = np.array([[0.00005, 0.5], [0.0005, 5.0]])
    
    # Train the model
    try:
        model.fit(X)
    except Exception as e:
        print(f"ERROR: HMM fit failed: {e}. Skipping update.", file=sys.stderr)
        return None
    
    # Determine Panic State Index (lowest VIX Spread mean = Backwardation/Panic)
    vix_spread_means = model.means_[:, 1]
    panic_state_index = np.argmin(vix_spread_means)
    
    # Assign the correct probability columns
    df['P_Panic'] = model.predict_proba(X)[:, panic_state_index]
    
    return df

def update_historical_data():
    """Main function to run the process."""
    
    try:
        print("DEBUG: Starting HMM data update process...")
        
        data = fetch_historical_data(TICKERS, TRAINING_DAYS)
        
        if data.empty:
            print("FATAL ERROR: Data fetching resulted in an empty DataFrame. Cannot proceed.", file=sys.stderr)
            return

        sp_df = data['^GSPC'].to_frame(name='Close')
        
        # *** FIX: Renaming ^VIX3M column ***
        vix_df = data[['^VIX', '^VIX3M']].rename(columns={'^VIX': 'VIX_Close', '^VIX3M': 'VIX3M_Close'})
        
        # 1. Calculate Log Returns
        sp_df['Log_Return'] = np.log(sp_df['Close'] / sp_df['Close'].shift(1))
        
        # 2. Feature Engineering: Calculate VIX Term Structure Proxy (Spread)
        # *** FIX: Using VIX3M for spread calculation ***
        vix_df['VIX_Spread'] = vix_df['VIX_Close'] - vix_df['VIX3M_Close']
        
        # 3. Merge dataframes
        df = pd.merge(
            sp_df[['Log_Return', 'Close']],
            vix_df[['VIX_Close', 'VIX_Spread']],
            left_index=True, 
            right_index=True, 
            how='inner'
        ).dropna()
        
        if df.empty:
            print("FATAL ERROR: Data merge/cleaning resulted in an empty DataFrame. Cannot proceed.", file=sys.stderr)
            return

        # --- HMM Execution ---
        df_with_hmm = train_and_predict_hmm(df)
        
        if df_with_hmm is None or df_with_hmm.empty:
            print("FATAL ERROR: HMM failed or returned empty. JSON not generated.", file=sys.stderr)
            return

        # --- JSON Generation ---
        display_days = int(CHART_PERIOD_DAYS * 0.7) 
        df_chart = df_with_hmm.tail(display_days).copy()
        
        output_df = pd.DataFrame({
            'Date': df_chart.index.strftime('%Y-%m-%d'),
            'VIX_Spread': df_chart['VIX_Spread'].round(2),
            'VIX_Close': df_chart['VIX_Close'].round(2),
            'P_Panic': df_chart['P_Panic'].round(4) * 100, # Convert to %
        })

        last_spread = df_chart.iloc[-1]['VIX_Spread']
        last_p_panic = df_chart.iloc[-1]['P_Panic']
        regime = "LOW-VOLATILITY" if last_p_panic < 0.5 else "HIGH-VOLATILITY"
        spread_status = "Backwardation (Acute Stress)" if last_spread < 0 else "Contango (Healthy)"
        
        chart_json = {
            'dates': output_df['Date'].tolist(),
            'p_panic': output_df['P_Panic'].tolist(),
            'vix_close': output_df['VIX_Close'].tolist(),
            'vix_spread': output_df['VIX_Spread'].tolist(),
            'last_reading': {
                'date': output_df.iloc[-1]['Date'],
                'vix_close': float(output_df.iloc[-1]['VIX_Close']),
                'vix_spread': float(output_df.iloc[-1]['VIX_Spread']), 
                'spread_status': spread_status,
                'p_panic': float(output_df.iloc[-1]['P_Panic']),
                'regime': regime
            }
        }

        os.makedirs(os.path.dirname(CHART_DATA_FILE), exist_ok=True)
        with open(CHART_DATA_FILE, 'w') as f:
            json.dump(chart_json, f, indent=4)
            
        print(f"SUCCESS: Updated {CHART_DATA_FILE} with {len(output_df)} data points.")

    except Exception as e:
        print(f"FATAL UNHANDLED ERROR IN SCRIPT: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == '__main__':
    update_historical_data()
