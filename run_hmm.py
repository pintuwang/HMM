# run_hmm.py - Final, Error-Checked Version

import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from datetime import date, timedelta
import os
import warnings
import json

# Suppress warnings from yfinance/pandas/hmmlearn during execution
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# IMPORTANT: This path MUST match the location where your website expects to load the file.
# Since your GitHub Pages is likely reading from the repository root, 'docs/chart_data.json'
# is the correct path if your website root is NOT the 'docs' folder.
CHART_DATA_FILE = 'docs/chart_data.json' 
CHART_PERIOD_DAYS = 365 * 1 # Last 1 year for display
N_COMPONENTS = 2 # 2 Regimes: Calm and Panic
TRAINING_DAYS = 365 * 3 # Use 3 years of data for training

def fetch_historical_data(tickers, period_days):
    """Fetches historical data for multiple tickers using yfinance."""
    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)
    
    print(f"Fetching data for {', '.join(tickers)} from {start_date} to {end_date}...")
    
    try:
        # Fetch the data and keep only the 'Close' prices
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        return data.dropna()
    except Exception as e:
        print(f"ERROR: yfinance data fetch failed: {e}")
        return pd.DataFrame()


def train_and_predict_hmm(df):
    """Trains the HMM and calculates the probability of the Panic Regime."""
    
    # CRITICAL CHECK: Ensure we have enough data points
    if len(df) < 50:
        print(f"ERROR: Not enough data points ({len(df)}) for HMM training.")
        return None
    
    X = df[['Log_Return', 'VIX_Spread']].values

    # Initialize the HMM
    model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type="diag", n_iter=500, tol=0.0001)
    
    # Initialize means and covariances
    model.means_ = np.array([[0.0003, 1.5], [-0.0005, -2.0]])
    model.covars_ = np.array([[0.00005, 0.5], [0.0005, 5.0]])
    
    # Train the model
    try:
        model.fit(X)
    except Exception as e:
        print(f"ERROR: HMM fit failed: {e}. Skipping update.")
        return None
    
    # Determine Panic State Index (lowest VIX Spread mean = Backwardation/Panic)
    vix_spread_means = model.means_[:, 1]
    panic_state_index = np.argmin(vix_spread_means)
    
    # Assign the correct probability columns
    df['P_Panic'] = model.predict_proba(X)[:, panic_state_index]
    
    return df

def update_historical_data():
    """Main function to run the process."""
    
    print("Starting HMM data update process...")
    
    TICKERS = ["^GSPC", "^VIX", "^VXST"]
    data = fetch_historical_data(TICKERS, TRAINING_DAYS)
    
    if data.empty:
        print("FATAL ERROR: Data fetching resulted in an empty DataFrame. Cannot proceed.")
        return

    sp_df = data['^GSPC'].to_frame(name='Close')
    vix_df = data[['^VIX', '^VXST']].rename(columns={'^VIX': 'VIX_Close', '^VXST': 'VXST_Close'})
    
    # Feature Engineering
    sp_df['Log_Return'] = np.log(sp_df['Close'] / sp_df['Close'].shift(1))
    vix_df['VIX_Spread'] = vix_df['VIX_Close'] - vix_df['VXST_Close']
    
    # Merge dataframes
    df = pd.merge(
        sp_df[['Log_Return', 'Close']],
        vix_df[['VIX_Close', 'VIX_Spread']],
        left_index=True, 
        right_index=True, 
        how='inner'
    ).dropna()
    
    if df.empty:
        print("FATAL ERROR: Data merge/cleaning resulted in an empty DataFrame. Cannot proceed.")
        return

    df_with_hmm = train_and_predict_hmm(df)
    
    if df_with_hmm is None or df_with_hmm.empty:
        print("FATAL ERROR: HMM training failed or returned an empty DataFrame. JSON not generated.")
        return

    # --- Prepare Data for Charting (JSON) ---
    
    display_days = int(CHART_PERIOD_DAYS * 0.7) 
    df_chart = df_with_hmm.tail(display_days).copy()
    
    if df_chart.empty:
        print("ERROR: Chart-ready DataFrame is empty. Cannot create JSON output.")
        return
        
    output_df = pd.DataFrame({
        'Date': df_chart.index.strftime('%Y-%m-%d'),
        'VIX_Spread': df_chart['VIX_Spread'].round(2),
        'VIX_Close': df_chart['VIX_Close'].round(2),
        'P_Panic': df_chart['P_Panic'].round(4) * 100, # Convert to % for the chart
    })

    # Calculate the VIX Spread value and status for the summary header
    last_spread = df_chart.iloc[-1]['VIX_Spread']
    last_p_panic = df_chart.iloc[-1]['P_Panic']
    
    # Determine the statistical regime
    regime = "LOW-VOLATILITY" if last_p_panic < 0.5 else "HIGH-VOLATILITY"

    spread_status = "Backwardation (Acute Stress)" if last_spread < 0 else "Contango (Healthy)"
    
    # Create the final JSON structure
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

    # Ensure the docs directory exists
    os.makedirs(os.path.dirname(CHART_DATA_FILE), exist_ok=True)
    
    try:
        with open(CHART_DATA_FILE, 'w') as f:
            json.dump(chart_json, f, indent=4)
            
        print(f"SUCCESS: Updated {CHART_DATA_FILE} with {len(output_df)} data points.")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not write JSON file to disk: {e}")

if __name__ == '__main__':
    update_historical_data()
