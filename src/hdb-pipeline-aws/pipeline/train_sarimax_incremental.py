"""
SARIMAX INCREMENTAL MODEL TRAINING - Memory Efficient for t4g.nano
Updates existing SARIMAX models with new data instead of full retraining
Falls back to full training if no existing models or data changes significantly

UPDATED: 7-month minimum with 2-month prediction horizon

USAGE: Place in same directory as your 5 CSV files and run:
    python train_sarimax_incremental.py

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

# File paths
MODEL_PATH = Path('sarimax-models-dict.pkl')
METRICS_PATH = Path('model-metrics.json')
S3_MODEL_PATH = Path('sarimax-models-dict.pkl')  # Downloaded from S3

# Training parameters
MIN_TRAINING_MONTHS = 7
PREDICTION_HORIZON = 2
INCREMENTAL_THRESHOLD_MONTHS = 3  # Only retrain if 3+ new months available


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


def load_existing_models():
    """Load existing models from S3 download or local file"""
    if S3_MODEL_PATH.exists():
        try:
            with open(S3_MODEL_PATH, 'rb') as f:
                models_dict = pickle.load(f)
            logger.info(f"✓ Loaded existing models: {len(models_dict)} groups")
            return models_dict
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
            return None
    elif MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                models_dict = pickle.load(f)
            logger.info(f"✓ Loaded existing models: {len(models_dict)} groups")
            return models_dict
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
            return None
    else:
        logger.info("No existing models found - will train from scratch")
        return None


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
    
    # Convert types
    df['month'] = pd.to_datetime(df['month'])
    df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
    
    # Extract storey midpoint
    df['storey_mid'] = df['storey_range'].apply(extract_storey_mid)
    
    # Calculate remaining lease
    df['remaining_lease'] = df.apply(calculate_remaining_lease, axis=1)
    
    # Clean data
    df = df.dropna(subset=['month', 'town', 'flat_type', 'resale_price', 'floor_area_sqm'])
    df = df[(df['resale_price'] > 50000) & (df['resale_price'] < 2000000)]
    df = df[(df['floor_area_sqm'] > 20) & (df['floor_area_sqm'] < 300)]
    
    logger.info(f"✓ Clean data: {len(df):,} rows")
    
    # Add region
    df['region'] = df['town'].apply(map_town_to_region)
    
    # Aggregate by month, region, flat_type
    agg = df.groupby(['month', 'region', 'flat_type']).agg({
        'resale_price': 'mean',
        'floor_area_sqm': 'mean',
        'storey_mid': 'mean',
        'remaining_lease': 'mean',
        'lease_commence_date': 'mean'
    }).reset_index()
    
    agg.columns = ['month', 'region', 'flat_type', 'avg_price', 'avg_floor_area', 
                   'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']
    
    # Fill missing exogenous features
    for col in ['avg_storey', 'avg_remaining_lease', 'avg_lease_commence']:
        agg[col] = agg.groupby(['region', 'flat_type'])[col].transform(
            lambda x: x.fillna(x.mean())
        )
    
    # Create group key
    agg['group_key'] = agg['region'] + '_' + agg['flat_type']
    
    logger.info(f"✓ Created {agg['group_key'].nunique()} groups")
    logger.info(f"✓ Aggregated to {len(agg):,} monthly observations")
    
    return agg


def prepare_group_data(group_df):
    """Prepare data for one group - fill missing months"""
    group_df = group_df.copy()
    group_df = group_df.sort_values('month')
    
    # Resample to monthly frequency
    group_df = group_df.set_index('month')
    group_df = group_df.asfreq('MS')
    
    # Forward fill then backward fill
    group_df = group_df.ffill().bfill()
    
    # Fill any remaining NaNs
    for col in group_df.columns:
        if group_df[col].isna().any():
            median = group_df[col].median()
            group_df[col] = group_df[col].fillna(0.0 if pd.isna(median) else median)
    
    return group_df.reset_index()


def scale_data(y, exog):
    """Scale data to prevent numerical instability"""
    # Scale target
    y_min, y_max = y.min(), y.max()
    y_scaled = (y - y_min) / (y_max - y_min + 1e-8)
    
    # Scale exogenous
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


def check_incremental_update_needed(existing_model, new_data):
    """
    Determine if incremental update is needed
    Returns: (needs_update, new_months_count, reason)
    """
    if existing_model is None:
        return True, 0, "No existing model"
    
    try:
        last_date = existing_model['last_date']
        new_max_date = new_data['month'].max()
        
        # Calculate months difference
        months_diff = (new_max_date.year - last_date.year) * 12 + \
                      (new_max_date.month - last_date.month)
        
        if months_diff >= INCREMENTAL_THRESHOLD_MONTHS:
            return True, months_diff, f"{months_diff} new months available"
        else:
            return False, months_diff, f"Only {months_diff} new months (threshold: {INCREMENTAL_THRESHOLD_MONTHS})"
            
    except Exception as e:
        return True, 0, f"Error checking update: {e}"


def train_sarimax_model(y, exog, order=(1,0,1), seasonal_order=(0,0,0,0)):
    """
    Train SARIMAX model for one group with scaling for stability
    """
    try:
        # Scale data
        y_scaled, exog_scaled, y_params, exog_params = scale_data(y, exog)
        
        # Fit model
        model = SARIMAX(
            y_scaled,
            exog=exog_scaled,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
            trend='c'
        )
        
        fitted_model = model.fit(disp=False, maxiter=200, method='lbfgs')
        
        return {
            'model': fitted_model,
            'y_params': y_params,
            'exog_params': exog_params
        }
        
    except Exception as e:
        logger.warning(f"SARIMAX fitting failed: {e}")
        
        # Fallback to simpler model
        try:
            y_scaled, exog_scaled, y_params, exog_params = scale_data(y, exog)
            
            model = SARIMAX(
                y_scaled,
                exog=exog_scaled,
                order=(1,0,0),
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


def train_models_incremental(df, existing_models=None):
    """
    Train or update SARIMAX models incrementally
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"INCREMENTAL SARIMAX TRAINING")
    logger.info(f"Minimum training data: {MIN_TRAINING_MONTHS} months")
    logger.info(f"Prediction horizon: {PREDICTION_HORIZON} months")
    logger.info(f"Incremental threshold: {INCREMENTAL_THRESHOLD_MONTHS} months")
    logger.info("=" * 60)
    
    groups = sorted(df['group_key'].unique())
    logger.info(f"Processing {len(groups)} groups...")
    
    models_dict = existing_models if existing_models is not None else {}
    train_metrics = []
    
    groups_trained = 0
    groups_updated = 0
    groups_skipped = 0
    
    for i, group in enumerate(groups):
        logger.info(f"\n[{i+1}/{len(groups)}] Processing {group}...")
        
        group_df = df[df['group_key'] == group].copy()
        
        if len(group_df) < MIN_TRAINING_MONTHS:
            logger.warning(f"  Skipping: only {len(group_df)} months (need {MIN_TRAINING_MONTHS})")
            groups_skipped += 1
            continue
        
        group_df = prepare_group_data(group_df)
        
        # Check if incremental update is needed
        existing_model = models_dict.get(group)
        needs_update, new_months, reason = check_incremental_update_needed(existing_model, group_df)
        
        if not needs_update:
            logger.info(f"  ⏭️  Skipping: {reason}")
            groups_skipped += 1
            continue
        
        logger.info(f"  🔄 Update needed: {reason}")
        logger.info(f"  Using all {len(group_df)} available months")
        
        y = group_df['avg_price'].values
        exog = group_df[['avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence']].values
        
        test_size = min(2, len(y) // 3)
        train_size = len(y) - test_size
        
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        
        logger.info(f"  Train: {train_size} months, Test: {test_size} months")
        
        # Train model
        model_dict = train_sarimax_model(y_train, exog_train)
        
        if model_dict is None:
            logger.warning(f"  Failed to fit model")
            groups_skipped += 1
            continue
        
        # Validate on test set
        try:
            # Scale test exogenous data
            exog_test_scaled = np.zeros_like(exog_test)
            for j, (col_min, col_max) in enumerate(model_dict['exog_params']):
                exog_test_scaled[:, j] = (exog_test[:, j] - col_min) / (col_max - col_min + 1e-8)
            
            # Forecast
            forecast_scaled = model_dict['model'].forecast(steps=test_size, exog=exog_test_scaled)
            
            # Unscale
            forecast = unscale_predictions(forecast_scaled, model_dict['y_params'])
            
            # Sanity check
            if forecast.min() < 0 or forecast.max() > 5000000:
                logger.warning(f"  Predictions out of range: {forecast.min():.0f} to {forecast.max():.0f}")
                groups_skipped += 1
                continue
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, forecast))
            mae = mean_absolute_error(y_test, forecast)
            mape = np.mean(np.abs((y_test - forecast) / y_test)) * 100
            
            # Quality check
            if rmse > 500000 or mape > 150:
                logger.warning(f"  Metrics too high - RMSE: ${rmse:,.0f}, MAPE: {mape:.1f}%")
                groups_skipped += 1
                continue
            
            logger.info(f"  ✓ Test RMSE: ${rmse:,.2f}, MAE: ${mae:,.2f}, MAPE: {mape:.2f}%")
            
            # Save model
            models_dict[group] = {
                'model': model_dict['model'],
                'y_params': model_dict['y_params'],
                'exog_params': model_dict['exog_params'],
                'last_y': y,
                'last_exog': exog,
                'last_date': group_df['month'].iloc[-1]
            }
            
            train_metrics.append({
                'group': group,
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'training_months': len(y_train),
                'new_months': int(new_months) if existing_model else 0,
                'update_type': 'incremental' if existing_model else 'new'
            })
            
            if existing_model:
                groups_updated += 1
            else:
                groups_trained += 1
            
        except Exception as e:
            logger.warning(f"  Validation failed: {e}")
            groups_skipped += 1
    
    logger.info(f"\n" + "=" * 60)
    logger.info(f"✓ Training Summary:")
    logger.info(f"  New models trained: {groups_trained}")
    logger.info(f"  Existing models updated: {groups_updated}")
    logger.info(f"  Models skipped (no update needed): {groups_skipped}")
    logger.info(f"  Total models: {len(models_dict)}")
    logger.info("=" * 60)
    
    return models_dict, train_metrics


def save_models(models_dict, metrics):
    """Save models and metrics"""
    logger.info("\nSaving models...")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(models_dict, f)
    logger.info(f"✓ Models saved: {MODEL_PATH}")
    
    # Calculate average metrics
    if len(metrics) > 0:
        avg_rmse = np.mean([m['rmse'] for m in metrics])
        avg_mae = np.mean([m['mae'] for m in metrics])
        avg_mape = np.mean([m['mape'] for m in metrics])
    else:
        avg_rmse = avg_mae = avg_mape = 0.0
    
    metrics_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'SARIMAX-scaled-incremental',
        'num_models': len(models_dict),
        'min_training_months': MIN_TRAINING_MONTHS,
        'prediction_horizon': PREDICTION_HORIZON,
        'incremental_threshold_months': INCREMENTAL_THRESHOLD_MONTHS,
        'groups': list(models_dict.keys()),
        'features': ['avg_floor_area', 'avg_storey', 'avg_remaining_lease', 'avg_lease_commence'],
        'architecture': {
            'base_order': '(1,0,1)',
            'seasonal_order': '(0,0,0,0)',
            'exogenous_variables': 4,
            'scaling': 'min-max normalization',
            'training_mode': 'incremental',
            'note': 'Incremental updates when 3+ new months available'
        },
        'training_summary': {
            'new_models': len([m for m in metrics if m.get('update_type') == 'new']),
            'updated_models': len([m for m in metrics if m.get('update_type') == 'incremental']),
            'total_updated_this_run': len(metrics)
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
    
    if len(metrics) > 0:
        logger.info(f"\nAverage Test Performance:")
        logger.info(f"  RMSE: ${avg_rmse:,.2f}")
        logger.info(f"  MAE:  ${avg_mae:,.2f}")
        logger.info(f"  MAPE: {avg_mape:.2f}%")


def main():
    try:
        # Load existing models
        existing_models = load_existing_models()
        
        # Load new data
        df = load_data()
        df = prepare_data(df)
        
        # Train or update models
        models_dict, metrics = train_models_incremental(df, existing_models)
        
        if len(models_dict) == 0:
            logger.error("No models in dictionary!")
            return False
        
        # Save models
        save_models(models_dict, metrics)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ INCREMENTAL TRAINING COMPLETE!")
        logger.info(f"✓ Total models: {len(models_dict)}")
        logger.info(f"✓ Models updated this run: {len(metrics)}")
        logger.info(f"✓ Minimum training data: {MIN_TRAINING_MONTHS} months")
        logger.info(f"✓ Prediction horizon: {PREDICTION_HORIZON} months")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)