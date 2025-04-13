# app.py
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()
import os
PORT = os.environ.get("PORT")
print("PORT:", PORT)

# ——— Setup Flask —————————————————————————————————————————————————————————————
app = Flask(__name__)
CORS(app)  # allow React frontend to call this API

BASE = Path(__file__).parent  # your project directory

# ——— Load & preprocess dataset —————————————————————————————————————————————
csv_path = BASE / "yield_df_complete_2023.csv"
df = pd.read_csv(csv_path)

# Rename yield column if needed
if 'hg/ha_yield' in df.columns:
    df = df.rename(columns={'hg/ha_yield': 'yield'})

# Drop the unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Filter out countries with fewer than 100 records
counts = df['Area'].value_counts()
valid = counts[counts >= 100].index
df = df[df['Area'].isin(valid)].reset_index(drop=True)

# Encode categorical features
area_encoder = LabelEncoder()
item_encoder = LabelEncoder()
df['Area_enc'] = area_encoder.fit_transform(df['Area'])
df['Item_enc'] = item_encoder.fit_transform(df['Item'])

# Prepare feature matrix & target (no Year column)
FEATURE_COLS = [
    'Area_enc',
    'Item_enc',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp'
]
X = df[FEATURE_COLS]
y = df['yield']

# Train/test split & model training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ——— Map UI labels → dataset labels ————————————————————————————————————————
# Your React UI uses "Rice (paddy)" but the CSV has "Rice, paddy"
UI_ITEM_MAP = {
    "Rice (paddy)": "Rice, paddy"
}

# ——— Prediction endpoint ———————————————————————————————————————————————————
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Ensure all required fields are present
    for f in ('area','item','rainfall','pesticide','temperature'):
        if data.get(f) is None:
            return jsonify({'error': f'Missing field: {f}'}), 400

    try:
        # Extract & cast inputs
        area      = data['area']
        item      = data['item']
        rainfall  = float(data['rainfall'])
        pesticide = float(data['pesticide'])
        temp      = float(data['temperature'])

        # Map UI label to actual dataset label
        item = UI_ITEM_MAP.get(item, item)

        # Encode
        a_enc = area_encoder.transform([area])[0]
        i_enc = item_encoder.transform([item])[0]

        # Build feature vector
        features = np.array([[a_enc, i_enc, rainfall, pesticide, temp]])

        # Predict per‑year yield
        per_year_yield = model.predict(features)[0]

        return jsonify({'yield': round(per_year_yield, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return "✅ Crop Yield Prediction API (no‑year) is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
