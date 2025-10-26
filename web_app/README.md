# AgPrice Web App

This is a minimal Flask web app to run the district-level forecasting model from the notebook.

Quick start (Windows cmd.exe):

```
cd c:\Users\acer\Desktop\agprice\web_app
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser. The app expects your `merged.csv` at `C:\Users\acer\Downloads\merged.csv`.

Notes:
- This is a simple demo — it trains per-request on the district history. For production, pre-train models and cache them.
- If xgboost isn't available, the app will fall back to a sklearn RandomForest for prediction.
