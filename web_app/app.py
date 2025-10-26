from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'merged.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def prepare_district_df(df, district_col, district_name):
    # normalize column names used in notebook
    df = df.rename(columns={
        'Price Date': 'date',
        'Modal Price (Rs./Quintal)': 'price',
        'Min Price (Rs./Quintal)': 'price_min',
        'Max Price (Rs./Quintal)': 'price_max'
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    # detect district/market column
    if district_col not in df.columns:
        # search common names
        for c in df.columns:
            if c.lower() in ('district','district name','district_name','market','market name','market_name'):
                district_col = c
                break
    if district_col not in df.columns:
        raise RuntimeError('No district/market column found in CSV')

    gdf = df[df[district_col].astype(str).str.lower() == district_name.lower()].sort_values('date').reset_index(drop=True)
    if gdf.empty:
        raise RuntimeError(f"No data found for district '{district_name}'")

    # create lag features (same as notebook)
    gdf['price_lag_1'] = gdf['price'].shift(1)
    gdf['price_roll7_mean'] = gdf['price'].rolling(7, min_periods=1).mean().shift(1)
    gdf = gdf.dropna(subset=['price','price_lag_1','price_roll7_mean']).reset_index(drop=True)
    return gdf


def train_and_forecast(gdf, horizon=7, return_model_scaler=False):
    # simple training on the district history and iterative forecast
    from sklearn.preprocessing import StandardScaler
    try:
        import xgboost as xgb
        ModelClass = xgb.XGBRegressor
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        ModelClass = RandomForestRegressor

    features = ['price_lag_1', 'price_roll7_mean']
    X = gdf[features].values
    y = gdf['price'].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = ModelClass(n_estimators=200, random_state=42) if hasattr(ModelClass, 'n_estimators') else ModelClass(random_state=42)
    # fit model
    try:
        model.fit(X_s, y)
    except Exception:
        # some models (xgboost) accept pandas arrays; try with raw X
        model.fit(X, y)

    if return_model_scaler:
        return model, scaler

    # iterative forecast
    hist_prices = list(gdf['price'].astype(float))
    preds = []
    for i in range(horizon):
        last_price = hist_prices[-1]
        roll_mean = float(np.mean(hist_prices[-7:])) if len(hist_prices) > 0 else last_price
        feat = [last_price, roll_mean]
        Xf = np.array(feat).reshape(1, -1)
        try:
            Xf_s = scaler.transform(Xf)
        except Exception:
            Xf_s = Xf
        p = model.predict(Xf_s)
        pval = p.item() if isinstance(p, np.ndarray) else float(p)
        preds.append(pval)
        hist_prices.append(pval)

    return preds


@app.route('/')
def index():
    # try to read CSV and enumerate available districts
    districts = []
    sample_cols = []
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            # find district-like column
            district_col = None
            for c in df.columns:
                if c.lower() in ('district','district name','district_name','market','market name','market_name'):
                    district_col = c
                    break
            if district_col:
                districts = sorted(df[district_col].dropna().astype(str).unique().tolist())
            sample_cols = df.columns.tolist()
        except Exception:
            pass
    return render_template('index.html', districts=districts, sample_cols=sample_cols)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    district = data.get('district')
    date_str = data.get('date')
    horizon = int(data.get('horizon', 7))

    if not district or not date_str:
        return jsonify({'error': 'Please provide district and date'}), 400

    try:
        prediction_date = pd.to_datetime(date_str)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400

    model_path = MODELS_DIR / f'model_{district.lower()}.joblib'
    scaler_path = MODELS_DIR / f'scaler_{district.lower()}.joblib'

    if not model_path.exists() or not scaler_path.exists():
        return jsonify({'error': f'Model for district {district} not found. Please train the models first.'}), 404

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # We need some historical data to start the forecast
    df = pd.read_csv(CSV_PATH)
    try:
        gdf = prepare_district_df(df, 'district', district) # Assuming 'district' is the column name
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    gdf = gdf[gdf['date'] < prediction_date]
    if gdf.empty:
        return jsonify({'error': f'No historical data available for {district} before {date_str}'}), 400

    # iterative forecast
    hist_prices = list(gdf['price'].astype(float))
    preds = []
    for i in range(horizon):
        last_price = hist_prices[-1]
        roll_mean = float(np.mean(hist_prices[-7:])) if len(hist_prices) > 0 else last_price
        feat = [last_price, roll_mean]
        Xf = np.array(feat).reshape(1, -1)
        try:
            Xf_s = scaler.transform(Xf)
        except Exception:
            Xf_s = Xf
        p = model.predict(Xf_s)
        pval = p.item() if isinstance(p, np.ndarray) else float(p)
        preds.append(pval)
        hist_prices.append(pval)

    # build dates starting from the selected date
    dates = [(prediction_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(horizon)]
    result = [{'date': d, 'predicted_price': float(p)} for d, p in zip(dates, preds)]
    return jsonify({'district': district, 'horizon': horizon, 'forecast': result})


if __name__ == '__main__':
    # run locally for quick testing
    app.run(host='0.0.0.0', port=5000, debug=True)
