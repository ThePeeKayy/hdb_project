import json
import requests
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

def fetch_hdb_data_all(town=None, flat_type=None, max_records=10000):
    dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    base_url = "https://data.gov.sg/api/action/datastore_search"

    filters = {}
    if town:
        filters["town"] = town
    if flat_type:
        filters["flat_type"] = flat_type

    all_records = []
    limit = 1000
    offset = 0

    while offset < max_records:
        params = {
            "resource_id": dataset_id,
            "filters": json.dumps(filters),
            "limit": limit,
            "offset": offset
        }
        r = requests.get(base_url, params=params)
        r.raise_for_status()
        result = r.json()["result"]
        records = result.get("records", [])
        if not records:
            break

        all_records.extend(records)
        if len(records) < limit:
            break
        offset += limit

    return all_records

def filter_by_remaining_lease_bucket(records, bucket_str, current_year=None):
    if current_year is None:
        current_year = datetime.now().year

    min_years, max_years = map(int, bucket_str.split('-'))
    filtered = []
    for r in records:
        lease_year = int(r["lease_commence_date"])
        remaining_lease = 99 - (current_year - lease_year)
        if min_years <= remaining_lease < max_years:
            r["remaining_lease_estimate"] = remaining_lease
            filtered.append(r)

    return filtered

with open("arima_unified_model.pkl", 'rb') as f:
    model_data = pickle.load(f)

@app.route('/api/predict', methods=['POST'])
def predict_prices():
    try:
        data = request.get_json()
        town = data.get('town')
        flat_type = data.get('flat_type')
        remaining_lease_bucket = data.get('remaining_lease_bucket')

        all_records = fetch_hdb_data_all(town=town, flat_type=flat_type, max_records=10000)
        filtered_records = filter_by_remaining_lease_bucket(all_records, remaining_lease_bucket)
        if len(filtered_records) == 0:
            return jsonify({
            "current_avg_price": 0,
            "predicted_6m_price": 0,
            "predicted_12m_price": 0,
            "trend": 'No sale'
        })
        current_avg_price = sum(int(r['resale_price']) for r in filtered_records) / len(filtered_records)
        town_encoded = model_data['le_town'].transform([town])[0]
        flat_type_encoded = model_data['le_flat'].transform([flat_type])[0]
        lease_bucket_encoded = model_data['le_lease'].transform([remaining_lease_bucket])[0]
        exog_pred = np.column_stack([
            [town_encoded] * 12,
            [flat_type_encoded] * 12,
            [lease_bucket_encoded] * 12
        ])
        forecast = model_data['model_fit'].forecast(steps=12, exog=exog_pred)
        predicted_6m_price = float(forecast[5])
        predicted_12m_price = float(forecast[11])
        
        change_percent = ((predicted_12m_price - current_avg_price) / current_avg_price) * 100
        if change_percent > 2:
            trend = 'increasing'
        elif change_percent < -2:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return jsonify({
            "current_avg_price": current_avg_price,
            "predicted_6m_price": predicted_6m_price,
            "predicted_12m_price": predicted_12m_price,
            "trend": trend
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
