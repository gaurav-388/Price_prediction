import pandas as pd
import joblib
from pathlib import Path
import os
from app import prepare_district_df, train_and_forecast # reusing functions from app

def train_all_models():
    """
    Trains a model for each district in the CSV and saves it to the models directory.
    """
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / 'merged.csv'
    MODELS_DIR = BASE_DIR / 'models'
    MODELS_DIR.mkdir(exist_ok=True)

    if not CSV_PATH.exists():
        print(f"CSV file not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Find district column
    district_col = None
    for c in df.columns:
        if c.lower() in ('district','district name','district_name','market','market name','market_name'):
            district_col = c
            break
    
    if not district_col:
        print("No district/market column found in CSV")
        return

    districts = df[district_col].dropna().astype(str).unique()
    print(f"Found {len(districts)} districts. Training models...")

    for district in districts:
        print(f"Training model for {district}...")
        try:
            gdf = prepare_district_df(df, district_col, district)
            if gdf.empty:
                print(f"  No data for {district}, skipping.")
                continue
            
            model, scaler = train_and_forecast(gdf, return_model_scaler=True)
            
            model_path = MODELS_DIR / f'model_{district.lower()}.joblib'
            scaler_path = MODELS_DIR / f'scaler_{district.lower()}.joblib'
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"  Saved model and scaler for {district}")

        except Exception as e:
            print(f"  Failed to train model for {district}: {e}")

if __name__ == '__main__':
    train_all_models()