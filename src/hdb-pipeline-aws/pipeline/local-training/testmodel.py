"""
SARIMAX MODEL TRAINING - Memory Efficient for t4g.nano
Predict average prices by flat type and region using SARIMAX
Uses exogenous variables for better predictions
Much lighter on memory than deep learning models

USAGE: Place in same directory as your 5 CSV files and run:
    python testmodel_sarimax.py

OUTPUTS (for manual S3 upload):
    - sarimax-models-dict.pkl  → Upload to s3://hdb-prediction-pipeline/gold/
    - model-metrics.json       → Upload to s3://hdb-prediction-pipeline/gold/
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from glob import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths - same directory as CSVs
MODEL_PATH = Path('sarimax-models-dict.pkl')
METRICS_PATH = Path('model-metrics.json')


def map_town_to_region(town):
    """Map town to Singapore region"""
    central = ['BISHAN', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA', 'GEYLANG', 
               'KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'TOA PAYOH']
    east = ['BEDOK', 'PASIR RIS', 'TAMPINES']
    north = ['ANG MO KIO', 'HOUGANG', 'PUNGGOL', 'SENGKANG', 'SERANGOON', 
             'SEMBAWANG', 'WOODLANDS', 'YISHUN']
    west = ['BUKIT BATOK', 'BUKIT PANJANG', 'CHOA CHU KANG', 'CLEMENTI', 
            'JURONG EAST', 'JURONG WEST', 'LIM CHU KANG']
    
    if town in central:
        return 'CENTRAL'
    elif town in east:
        return 'EAST'
    elif town in north:
        return 'NORTH'
    elif town in west:
        return 'WEST'
    else:
        return 'OTHERS'


def extract_storey_mid(storey_range):
    """Extract midpoint of storey range"""
    if pd.isna(storey_range):
        return np.nan
    try:
        parts = str(storey_range).split(' TO ')
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
    except:
        pass
    return np.nan


def calculate_remaining_lease(row):
    """Calculate remaining lease in years"""
    try:
        lease_commence = int(row['lease_commence_date'])
        transaction_year = row['month'].year
        remaining = 99 - (transaction_year - lease_commence)
        return max(0, remaining)
    except:
        return np.nan


def load_data():
    """Load all CSV files"""
    logger.info("Loading CSV files...")
    csv_files = glob('*.csv')
    
    if len(csv_files) == 0:
        raise FileNotFoundError("No CSV files found!")
    
    dfs = []
    for csv_file in csv_files:
        logger.info(f"  Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"✓ Loaded {len(combined):,} rows")
    
    return combined


def prepare_data(df):
    """Clean and prepare data with enhanced features"""
    logger.info("\nPreparing data...")
    
    # Basic cleaning
    df['month'] = pd.to_datetime(df['month'])
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
    
    # Extract storey midpoint
    df['storey_mid'] = df['storey_range'].apply(extract_storey_mid)
    
    # Calculate remaining lease
    df['remaining_lease'] = df.apply(calculate_remaining_lease, axis=1)
    
    # Remove outliers
    df = df.dropna(subset=['month', 'town', 'flat_type', 'resale_price', 'floor_area_sqm'])
    df = df[(df['resale_price'] > 50000) & (df['resale_price'] < 2000000)]
    df = df[(df['floor_area_sqm'] > 20) & (df['floor_area_sqm'] < 300)]
    
    logger.info(f"✓ Clean data: {len(df):,} rows")
    
    # Add region
    df['region'] = df['town'].apply(map_town_to_region)
    
    # Enhanced aggregation with more features
    agg = df.groupby(['month', 'region', 'flat_type']).agg({
        'resale_price': 'mean',
        'floor_area_sqm': 'mean',
        'storey_mid': 'mean',
        'remaining_lease': 'mean',
        'lease_commence_date': 'mean'
    }).reset_index()
    
    agg.columns = ['month', 'region', 'flat_type', 'avg_price', 'avg_floor_area', 
                   'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']
    
    # Fill any remaining NaNs with group means
    for col in ['avg_storey', 'avg_remaining_lease', 'avg_lease_commence']:
        agg[col] = agg.groupby(['region', 'flat_type'])[col].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Create group key
    agg['group_key'] = agg['region'] + '_' + agg['flat_type']
    
    logger.info(f"✓ Created {agg['group_key'].nunique()} groups")
    logger.info(f"✓ Aggregated to {len(agg):,} monthly observations")
    logger.info(f"✓ Features: price, floor_area, storey, remaining_lease, lease_commence")
    
    return agg


def prepare_group_data(group_df):
    """Prepare data for one group - fill missing months"""
    group_df = group_df.copy()
    group_df = group_df.sort_values('month')
    
    # Fill missing months
    group_df = group_df.set_index('month')
    group_df = group_df.asfreq('MS')
    
    # Forward fill then backward fill
    group_df = group_df.ffill().bfill()
    
    # Fill any remaining NaNs with median
    for col in group_df.columns:
        if group_df[col].isna().any():
            median = group_df[col].median()
            group_df[col] = group_df[col].fillna(0.0 if pd.isna(median) else median)
    
    return group_df.reset_index()


def scale_data(y, exog):
    """Scale data to prevent numerical instability"""
    # Scale y to be in range [0, 1]
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min + 1e-8)
    
    # Scale each exog column
    exog_scaled = np.zeros_like(exog)
    exog_params = []
    for i in range(exog.shape[1]):
        col_min, col_max = exog[:, i].min(), exog[:, i].max()
        exog_scaled[:, i] = (exog[:, i] - col_min) / (col_max - col_min + 1e-8)
        exog_params.append((col_min, col_max))
    
    return y_scaled, exog_scaled, (y_min, y_max), exog_params


def unscale_predictions(y_pred_scaled, y_params):
    """Unscale predictions back to original range"""
    y_min, y_max = y_params
    return y_pred_scaled * (y_max - y_min) + y_min


def train_sarimax_model(y, exog, order=(1,0,1), seasonal_order=(0,0,0,0)):
    """
    Train SARIMAX model for one group with scaling for stability
    
    Uses simpler ARIMAX (no differencing, no seasonality) to avoid instability
    Scaling prevents numerical explosion
    """
    try:
        # Scale data
        y_scaled, exog_scaled, y_params, exog_params = scale_data(y, exog)
        
        # Simple ARIMAX model (more stable than full SARIMAX)
        model = SARIMAX(
            y_scaled,
            exog=exog_scaled,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c'  # Add constant trend
        )
        
        fitted_model = model.fit(disp=False, maxiter=200, method='lbfgs')
        
        # Return model with scaling parameters
        return {
            'model': fitted_model,
            'y_params': y_params,
            'exog_params': exog_params
        }
        
    except Exception as e:
        logger.warning(f"SARIMAX fitting failed: {e}")
        # Fallback to even simpler AR model
        try:
            y_scaled, exog_scaled, y_params, exog_params = scale_data(y, exog)
            
            model = SARIMAX(
                y_scaled,
                exog=exog_scaled,
                order=(1,0,0),  # Just AR(1)
                seasonal_order=(0,0,0,0),
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend='c'
            )
            fitted_model = model.fit(disp=False, maxiter=100)
            
            return {
                'model': fitted_model,
                'y_params': y_params,
                'exog_params': exog_params
            }
        except:
            return None


def train_models(df):
    """Train SARIMAX model for each group"""
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SARIMAX MODELS (with scaling)")
    logger.info("=" * 60)
    
    groups = sorted(df['group_key'].unique())
    logger.info(f"Processing {len(groups)} groups...")
    
    models_dict = {}
    train_metrics = []
    
    for i, group in enumerate(groups):
        logger.info(f"\n[{i+1}/{len(groups)}] Training {group}...")
        
        group_df = df[df['group_key'] == group].copy()
        
        # Need at least 36 months
        if len(group_df) < 36:
            logger.warning(f"  Skipping: only {len(group_df)} months")
            continue
        
        # Prepare data
        group_df = prepare_group_data(group_df)
        
        # Use last 36 months for training
        group_df = group_df.tail(36)
        
        # Prepare target and exogenous variables
        y = group_df['avg_price'].values
        exog = group_df[['avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']].values
        
        # Split: use last 6 months as test
        train_size = len(y) - 6
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        
        # Train model (returns dict with model and scaling params)
        model_dict = train_sarimax_model(y_train, exog_train)
        
        if model_dict is None:
            logger.warning(f"  Failed to fit model")
            continue
        
        # Validate on test set
        try:
            # Scale test exog using same parameters
            exog_test_scaled = np.zeros_like(exog_test)
            for j, (col_min, col_max) in enumerate(model_dict['exog_params']):
                exog_test_scaled[:, j] = (exog_test[:, j] - col_min) / (col_max - col_min + 1e-8)
            
            # Forecast in scaled space
            forecast_scaled = model_dict['model'].forecast(steps=6, exog=exog_test_scaled)
            
            # Unscale predictions
            forecast = unscale_predictions(forecast_scaled, model_dict['y_params'])
            
            # Sanity check: predictions should be in reasonable range
            if forecast.min() < 0 or forecast.max() > 5000000:  # $5M is way too high
                logger.warning(f"  Predictions out of range: {forecast.min():.0f} to {forecast.max():.0f}")
                continue
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, forecast))
            mae = mean_absolute_error(y_test, forecast)
            mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
            
            # Sanity check metrics
            if rmse > 500000 or mape > 50:  # Something is very wrong
                logger.warning(f"  Metrics too high - RMSE: ${rmse:,.0f}, MAPE: {mape:.1f}%")
                continue
            
            logger.info(f"  ✓ Test RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, MAPE: {mape:.2f}%")
            
            # Store model and last known data for future predictions
            models_dict[group] = {
                'model': model_dict['model'],
                'y_params': model_dict['y_params'],
                'exog_params': model_dict['exog_params'],
                'last_y': y,  # Store full history
                'last_exog': exog,  # Store full exog history
                'last_date': group_df['month'].iloc[-1]
            }
            
            train_metrics.append({
                'group': group,
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape)
            })
            
        except Exception as e:
            logger.warning(f"  Validation failed: {e}")
    
    logger.info(f"\n✓ Successfully trained {len(models_dict)} models")
    
    return models_dict, train_metrics


def save_models(models_dict, metrics):
    """Save models and metrics"""
    logger.info("\nSaving models...")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(models_dict, f)
    logger.info(f"✓ Models saved: {MODEL_PATH}")
    
    # Calculate average metrics
    avg_rmse = np.mean([m['rmse'] for m in metrics])
    avg_mae = np.mean([m['mae'] for m in metrics])
    avg_mape = np.mean([m['mape'] for m in metrics])
    
    metrics_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'SARIMAX-scaled',
        'num_models': len(models_dict),
        'groups': list(models_dict.keys()),
        'features': ['avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence'],
        'architecture': {
            'base_order': '(1,0,1)',
            'seasonal_order': '(0,0,0,0)',
            'exogenous_variables': 4,
            'scaling': 'min-max normalization',
            'note': 'Simple ARIMAX without differencing for stability'
        },
        'test_performance': {
            'avg_rmse': float(avg_rmse),
            'avg_mae': float(avg_mae),
            'avg_mape': float(avg_mape)
        },
        'individual_metrics': metrics
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"✓ Metrics saved: {METRICS_PATH}")
    
    logger.info(f"\nAverage Test Performance:")
    logger.info(f"  RMSE: ${avg_rmse:,.2f}")
    logger.info(f"  MAE:  ${avg_mae:,.2f}")
    logger.info(f"  MAPE: {avg_mape:.2f}%")
    
    logger.info("\n" + "=" * 60)
    logger.info("NEXT STEPS - UPLOAD TO S3:")
    logger.info("=" * 60)
    logger.info(f"Run these commands to upload to your S3 bucket:")
    logger.info("")
    logger.info(f"  aws s3 cp {MODEL_PATH} s3://hdb-prediction-pipeline/gold/")
    logger.info(f"  aws s3 cp {METRICS_PATH} s3://hdb-prediction-pipeline/gold/")
    logger.info("")
    logger.info("Verify upload:")
    logger.info("  aws s3 ls s3://hdb-prediction-pipeline/gold/")
    logger.info("=" * 60)


def main():
    try:
        df = load_data()
        df = prepare_data(df)
        
        models_dict, metrics = train_models(df)
        
        if len(models_dict) == 0:
            logger.error("No models trained successfully!")
            return False
        
        save_models(models_dict, metrics)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ TRAINING COMPLETE!")
        logger.info(f"✓ Trained {len(models_dict)} SARIMAX models")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)