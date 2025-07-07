import os
import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

DATA_FOLDER = "../hdbData"      
OUTPUT_MODEL = "arima_unified_model.pkl" 
TEST_STEPS = 30                       

def bucketize(lease_commence_date):
    current_year = datetime.now().year
    building_age = current_year - lease_commence_date
    remaining_lease = 99 - building_age
    
    if remaining_lease <= 20:
        return '0-20'
    elif remaining_lease <= 40:
        return '20-40'
    elif remaining_lease <= 60:
        return '40-60'
    elif remaining_lease <= 80:
        return '60-80'
    else:
        return '80-100'

def load_and_preprocess():
    all_dfs = []
    for fname in os.listdir(DATA_FOLDER):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_FOLDER, fname))
            all_dfs.append(df)

    data = pd.concat(all_dfs, ignore_index=True)
    data = data.drop(columns=['block', 'street_name', 'remaining_lease'])

    data['month'] = pd.to_datetime(data['month'], format='%Y-%m')
    data['remaining_lease_bucket'] = data['lease_commence_date'].apply(bucketize)
    data = data.drop(columns=['lease_commence_date'])

    return data

def train_unified_arima(data):
    le_town = LabelEncoder()
    le_flat = LabelEncoder()
    le_lease = LabelEncoder()
    
    data['town_encoded'] = le_town.fit_transform(data['town'])
    data['flat_type_encoded'] = le_flat.fit_transform(data['flat_type'])
    data['lease_bucket_encoded'] = le_lease.fit_transform(data['remaining_lease_bucket'])

    monthly_data = data.groupby('month').agg({
        'resale_price': 'mean',
        'town_encoded': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'flat_type_encoded': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'lease_bucket_encoded': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).sort_index()
    
    train_data = monthly_data.iloc[:-TEST_STEPS]
    test_data = monthly_data.iloc[-TEST_STEPS:]
    
    exog_train = train_data[['town_encoded', 'flat_type_encoded', 'lease_bucket_encoded']]
    exog_test = test_data[['town_encoded', 'flat_type_encoded', 'lease_bucket_encoded']]
    
    model = ARIMA(train_data['resale_price'], exog=exog_train, order=(2, 2, 2))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=TEST_STEPS, exog=exog_test)
    mae = mean_absolute_error(test_data['resale_price'], forecast)
    print(f"MAE on holdout: {mae:.2f}")
    
    model_data = {
        'model_fit': model_fit,
        'mae': mae,
        'le_town': le_town,
        'le_flat': le_flat,
        'le_lease': le_lease,
        'start_date': train_data.index.min(),
        'freq': train_data.index.inferred_freq
    }
    
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"final pth file: {OUTPUT_MODEL}")
    
    return model_data

if __name__ == "__main__":
    data = load_and_preprocess()
    train_unified_arima(data)