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
VIX_SYMBOL = "^VIX"
N_COMPONENTS = 2 # 2 Regimes: Calm and Panic
TRAINING_DAYS = 365 * 3 # Use 3 years of data for training

def get_data(symbol, start_date):
    """Fetches historical data using yfinance."""
    end_date = date.today()
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return data

def train_and_predict_hmm(df):
    """Trains the HMM and calculates the probability of the Panic Regime."""
    
    # Use Log Returns as the observable input
    X = df['Log_Return'].values.reshape(-1, 1)

    # Initialize the HMM (Gaussian for simplicity, upgrade to Student's t later)
    model = hmm.GaussianHMM(n_components=N_COMPONENTS, covariance_type="diag", n_iter=100)
    
    # 1. Initialize parameters based on volatility (variance)
    initial_covars = [[0.0002], [0.00005]] # Guess: High variance (Panic) and Low variance (Calm)
    model.covars_ = initial_covars
    
    # 2. Train the model
    try:
        model.fit(X)
    except Exception as e:
        print(f"HMM fit failed: {e}. Skipping update.")
        return None
    
    # 3. Predict the hidden states and probabilities
    probabilities = model.predict_proba(X)
    
    # 4. Identify the Panic State (The one with the higher variance)
    panic_state_index = np.argmax(model.covars_.flatten())
    
    # Get the last day's probability
    df['P_Panic'] = probabilities[:, panic_state_index]
    
    return df

def update_historical_data():
    """Main function to run the process."""
    
    print("Fetching financial data...")
    # Fetch 3 years of data for robust training
    start_date = date.today() - timedelta(days=TRAINING_DAYS)
    sp_data = get_data(SYMBOL, start_date)
    vix_data = get_data(VIX_SYMBOL, start_date)
    
    # Pre-processing
    sp_data['Log_Return'] = np.log(sp_data['Close'] / sp_data['Close'].shift(1))
    
    # --- REVISED CODE (Solution 2) ---

    # 1. Prepare VIX data: Convert Series to DataFrame and explicitly rename the column
    vix_df = vix_data[['Close']].rename(columns={'Close': 'VIX_Close'})
    
    # 2. Merge S&P 500 (df) with the renamed VIX data (vix_df) on the index (Date)
    df = pd.merge(
        sp_data[['Log_Return', 'Close']],
        vix_df,
        left_index=True, 
        right_index=True, 
        how='inner'
    ).dropna()
   
    # Run HMM
    df_with_hmm = train_and_predict_hmm(df)
    
    if df_with_hmm is None:
        return

    # --- Prepare Data for CSV ---
    # Get the last two years of data (training data minus recent date)
    df_new_history = df_with_hmm.tail(TRAINING_DAYS - 365)
    
    # Prepare the output row for the CSV structure
    output_df = pd.DataFrame({
        'Date': df_new_history.index.strftime('%Y-%m-%d'),
        'Return': df_new_history['Log_Return'].round(6),
        'VIX_Close': df_new_history['VIX_Close'].round(2),
        'P_Calm': (1 - df_new_history['P_Panic']).round(4),
        'P_Panic': df_new_history['P_Panic'].round(4),
        # Assuming Regime 2 is Panic
        'Most_Likely_Regime': np.where(df_new_history['P_Panic'] > 0.5, 2, 1)
    })
    
    # --- Update the JSON File for Charting ---
    
    # Filter for the last 1 year for display
    chart_data = output_df.tail(int(CHART_PERIOD_DAYS * 0.7)) # approx. 1 year of trading days
    
    # Create the JSON structure for Chart.js
    chart_json = {
        'dates': chart_data['Date'].tolist(),
        'p_panic': chart_data['P_Panic'].tolist(),
        'vix_close': chart_data['VIX_Close'].tolist(),
        'last_reading': chart_data.iloc[-1].to_dict()
    }

    # Ensure the docs directory exists
    os.makedirs(os.path.dirname(CHART_DATA_FILE), exist_ok=True)
    
    with open(CHART_DATA_FILE, 'w') as f:
        json.dump(chart_json, f)
        
    print(f"Successfully updated {CHART_DATA_FILE} with {len(chart_data)} data points.")
    
if __name__ == '__main__':
    update_historical_data()
