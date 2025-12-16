# run_hmm.py

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
# Note: Changing this to 'chart_data.json' might fix the 404 error if your
# website is configured to read from the repository root.
CHART_DATA_FILE = 'docs/chart_data.json' # File for JS
CHART_PERIOD_DAYS = 365 * 1 # Last 1 year for display
SYMBOL = "^GSPC" # S&P 500
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
    
    # X now includes Log_Return and the VIX_Spread (Term Structure)
    X = df[['Log_Return', 'VIX_Spread']].values

    # Initialize the HMM
    model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type="diag", n_iter=500, tol=0.0001)
    
    # 1. Initialize means for two features: [Log_Return, VIX_Spread]
    model.means_ = np.array([[0.0003, 1.5], # Calm State: Positive return, Contango
                             [-0.0005, -2.0]]) # Panic State: Negative return, Backwardation
    
    # 2. Initialize diagonal covariances for two features: [Returns Variance, VIX Spread Variance]
    model.covars_ = np.array([[0.00005, 0.5], # Calm State: Low variance
                              [0.0005, 5.0]]) # Panic State: High variance
    
    # 3. Train the model
    try:
        # Use only the feature matrix X for training
        model.fit(X)
    except Exception as e:
        print(f"ERROR: HMM fit failed: {e}. Skipping update.")
        return None
    
    # --- REVISED PREDICTION AND INDEXING ---

    # Get the probabilities of being in each state
    probabilities = model.predict_proba(X)
    
    # DETERMINE THE PANIC STATE INDEX:
    # The Panic State is the one with the lowest VIX Spread mean (second feature)
    vix_spread_means = model.means_[:, 1] # Extract the VIX Spread means (second column)
    panic_state_index = np.argmin(vix_spread_means) # Index 0 or 1 that has the lowest Spread
    
    # 2. Assign the correct probability columns
    df['P_Panic'] = probabilities[:, panic_state_index]
    
    # The remaining state is P_Calm
    calm_state_index = 1 - panic_state_index
    df['P_Calm'] = probabilities[:, calm_state_index]

    return df

def update_historical_data():
    """Main function to run the process."""
    
    print("Starting HMM data update process...")
    
    # Fetch data
    TICKERS = ["^GSPC", "^VIX", "^VXST"] # S&P, VIX, Short-Term VIX
    data = fetch_historical_data(TICKERS, TRAINING_DAYS)
    
    if data.empty:
        print("FATAL ERROR: Data fetching resulted in an empty DataFrame. Cannot proceed.")
        return

    # Separate dataframes for clarity
    sp_df = data['^GSPC'].to_frame(name='Close')
    vix_df = data[['^VIX', '^VXST']].rename(columns={'^VIX': 'VIX_Close', '^VXST': 'VXST_Close'})
    
    # 1. Calculate Log Returns
    sp_df['Log_Return'] = np.log(sp_df['Close'] / sp_df['Close'].shift(1))
    
    # 2. Feature Engineering: Calculate VIX Term Structure Proxy (Spread)
    vix_df['VIX_Spread'] = vix_df['VIX_Close'] - vix_df['VXST_Close']
    
    # 3. Merge S&P 500 features (Log_Return) with VIX features (VIX_Spread)
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

    # Run HMM
    df_with_hmm = train_and_predict_hmm(df)
    
    if df_with_hmm is None or df_with_hmm.empty:
        print("FATAL ERROR: HMM training failed or returned an empty DataFrame. JSON not generated.")
        return

    # --- Prepare Data for Charting (JSON) ---
    
    # Filter for the last ~1 year (approx 252 trading days)
    display_days = int(CHART_PERIOD_DAYS * 0.7) 
    
    df_chart = df_with_hmm.tail(display_days).copy()
    
    if df_chart.empty:
        print("ERROR: Chart-ready DataFrame is empty. Cannot create JSON output.")
        return
        
    # Prepare the output DataFrame
    output_df = pd.DataFrame({
        'Date': df_chart.index.strftime('%Y-%m-%d'),
        'VIX_Spread': df_chart['VIX_Spread'].round(2),
        'VIX_Close': df_chart['VIX_Close'].round(2),
        'P_Panic': df_chart['P_Panic'].round(4),
    })

    # Calculate the VIX Spread value and status for the summary header
    last_spread = df_chart.iloc[-1]['VIX_Spread']
    spread_status = "Backwardation (Acute Stress)" if last_spread < 0 else "Contango (Healthy)"
    
    # Create the JSON structure for Chart.js
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
        }
    }

    # Ensure the docs directory exists
    os.makedirs(os.path.dirname(CHART_DATA_FILE), exist_ok=True)
    
    try:
        with open(CHART_DATA_FILE, 'w') as f:
            json.dump(chart_json, f, indent=4) # Use indent for readability when debugging
            
        print(f"SUCCESS: Updated {CHART_DATA_FILE} with {len(output_df)} data points.")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not write JSON file to disk: {e}")

if __name__ == '__main__':
    update_historical_data()
