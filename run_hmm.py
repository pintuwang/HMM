# run_hmm.py - TEMPORARY DUMMY TEST SCRIPT

import json
import os
from datetime import date

CHART_DATA_FILE = 'docs/chart_data.json'

def create_dummy_json():
    """Generates a simple, guaranteed-to-work JSON structure."""
    
    print("--- Running DUMMY JSON TEST ---")
    
    # Create a simple, known-good data structure
    dummy_data = {
        'dates': [date.today().strftime('%Y-%m-%d')],
        'p_panic': [10.0],
        'vix_close': [15.0],
        'vix_spread': [2.5],
        'last_reading': {
            'date': date.today().strftime('%Y-%m-%d'),
            'vix_close': 15.0,
            'vix_spread': 2.5,
            'spread_status': 'Contango (Healthy)',
            'p_panic': 10.0,
        }
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(CHART_DATA_FILE), exist_ok=True)
    
    # Write the file
    try:
        with open(CHART_DATA_FILE, 'w') as f:
            json.dump(dummy_data, f, indent=4)
            
        print(f"SUCCESS: Successfully wrote DUMMY JSON to {CHART_DATA_FILE}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not write JSON file to disk: {e}")

if __name__ == '__main__':
    create_dummy_json()
