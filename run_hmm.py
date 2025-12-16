# run_hmm.py

import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn import hmm
from datetime import date, timedelta
import os
import warnings
import json

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
DATA_FILE = 'hmm_historical_data.csv'
CHART_DATA_FILE = 'docs/chart_data.json' # File for JS
CHART_PERIOD_DAYS = 365 * 1 # Last 1 year for display
SYMBOL = "^GSPC" # S&P 500
# VIX_SYMBOL is now unused as we fetch multiple VIX-related indices
N_COMPONENTS = 2 # 2 Regimes: Calm and Panic
TRAINING_DAYS = 365 * 3 # Use 3 years of data for training

def fetch_historical_data(tickers, period="3y"):
    """Fetches historical data for multiple tickers using yfinance."""
    end_date = date.today()
    start_date = end_date - timedelta(days=period)
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    return data.dropna()


def train_and_predict_hmm(df):
    """Trains the HMM and calculates the probability of the Panic Regime."""
    
    # X now includes Log_Return and the VIX_Spread (Term Structure)
    X = df[['Log_Return', 'VIX_Spread']].values

    # Initialize the HMM
    model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type="diag", n_iter=500)
    
    
    # 1. Initialize means for two features: [Log_Return, VIX_Spread]
    # Calm State (0): Positive return, Positive Spread (Contango, e.g., 1.5)
    # Panic State (1): Negative return, Negative Spread (Backwardation, e.g., -2.0)
    model.means_ = np.array([[0.0003, 1.5], 
                             [-0.0005, -2.0]])
    
    # 2. Initialize diagonal covariances for two features: [Returns Variance, VIX Spread Variance]
    # Calm State (0): Low return variance, Low VIX Spread variance
    # Panic State (1): High return variance, High VIX Spread variance
    model.covars_ = np.array([[0.00005, 0.5], 
                              [0.0005, 5.0]])
    
    
    # 2. Train the model
    try:
        model.fit(X)
    except Exception as e:
        print(f"HMM fit failed: {e}. Skipping update.")
        return None
    
    # --- REVISED PREDICTION AND INDEXING ---

    # Get the probabilities of being in each state
    probabilities = model.predict_proba(X)
    
    # DETERMINE THE PANIC STATE INDEX:
    # The Panic State is the one with the lowest VIX Spread mean (second feature) 
    # as backwardation (negative spread) indicates panic.
    
    # 1. Find the index (0 or 1) where the VIX Spread mean is lowest.
    vix_spread_means = model.means_[:, 1] # Extract the VIX Spread means (second column)
    panic_state_index = np.argmin(vix_spread_means) # Index 0 or 1 that has the lowest Spread (most negative)
    
    # 2. Use the correct index for the output
    df['P_Panic'] = probabilities[:, panic_state_index]
    
    # The remaining state is P_Calm
    calm_state_index = 1 - panic_state_index
    df['P_Calm'] = probabilities[:, calm_state_index]

    return df

def update_historical_data():
    """Main function to run the process."""
    
    print("Fetching financial data...")
    # Fetch 3 years of data for robust training
    TICKERS = ["^GSPC", "^VIX", "^VXST"] # S&P, VIX, Short-Term VIX
    data = fetch_historical_data(TICKERS, period=TRAINING_DAYS)
    
    # Separate dataframes for clarity
    sp_df = data['^GSPC'].to_frame(name='Close')
    vix_df = data[['^VIX', '^VXST']].rename(columns={'^VIX': 'VIX_Close', '^VXST': 'VXST_Close'})
    
    # 1. Calculate Log Returns
    sp_df['Log_Return'] = np.log(sp_df['Close'] / sp_df['Close'].shift(1))
    
    # 2. Feature Engineering: Calculate VIX Term Structure Proxy (Spread)
    # Spread = VIX_Close - VXST_Close. Backwardation (Panic) means this spread is negative.
    vix_df['VIX_Spread'] = vix_df['VIX_Close'] - vix_df['VXST_Close']
    
    # 3. Merge S&P 500 features (Log_Return) with VIX features (VIX_Spread)
    df = pd.merge(
        sp_df[['Log_Return', 'Close']],
        vix_df[['VIX_Close', 'VIX_Spread']],
        left_index=True, 
        right_index=True, 
        how='inner'
    ).dropna()
    
    # Run HMM
    df_with_hmm = train_and_predict_hmm(df)
    
    if df_with_hmm is None:
        return

    # --- Prepare Data for Charting (JSON) ---
    
    # Filter for the last ~1 year for display
    display_days = int(CHART_PERIOD_DAYS * 0.7) 
    
    df_chart = df_with_hmm.tail(display_days)
    
    # Prepare the output DataFrame (Simplified for this check)
    output_df = pd.DataFrame({
        'Date': df_chart.index.strftime('%Y-%m-%d'),
        'VIX_Spread': df_chart['VIX_Spread'].round(2),
        'VIX_Close': df_chart['VIX_Close'].round(2),
        'P_Panic': df_chart['P_Panic'].round(4),
    })

    # Calculate the VIX Spread value for the summary header
    # Check if df_chart is not empty before accessing .iloc[-1]
    if df_chart.empty:
        print("Error: df_chart is empty. Cannot create JSON output.")
        return
        
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
            'vix_close': output_df.iloc[-1]['VIX_Close'],
            'vix_spread': float(output_df.iloc[-1]['VIX_Spread']), # Convert to float explicitly
            'spread_status': spread_status,
            'p_panic': float(output_df.iloc[-1]['P_Panic']), # Convert to float explicitly
        }
    }

    # Ensure the docs directory exists
    os.makedirs(os.path.dirname(CHART_DATA_FILE), exist_ok=True)
    
    with open(CHART_DATA_FILE, 'w') as f:
        json.dump(chart_json, f)
        
    print(f"Successfully updated {CHART_DATA_FILE} with {len(output_df)} data points.")
    
if __name__ == '__main__':
    update_historical_data()
