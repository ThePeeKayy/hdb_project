

set -e

PIPELINE_DIR="/home/ubuntu/pipeline"
LOG_DIR="/home/ubuntu/logs"
LOG_FILE="$LOG_DIR/weekly_retrain_$(date +%Y%m%d_%H%M%S).log"
S3_BUCKET="hdb-prediction-pipeline"

mkdir -p "$LOG_DIR"

source /home/ubuntu/miniconda3/bin/activate hdb-env

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "WEEKLY MODEL RETRAIN (SARIMAX 7-MONTH)"
log "=========================================="

cd "$PIPELINE_DIR"


log "→ Running daily pipeline first..."
bash daily_pipeline.sh >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    log "✗ Daily pipeline failed"
    exit 1
fi

log "✓ Daily pipeline completed"


log "→ Downloading data from S3 for training..."
aws s3 cp s3://$S3_BUCKET/resale-flat-prices-based-on-approval-date-1990-1999.csv . >> "$LOG_FILE" 2>&1
aws s3 cp s3://$S3_BUCKET/resale-flat-prices-based-on-approval-date-2000-feb-2012.csv . >> "$LOG_FILE" 2>&1
aws s3 cp s3://$S3_BUCKET/resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv . >> "$LOG_FILE" 2>&1
aws s3 cp s3://$S3_BUCKET/resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv . >> "$LOG_FILE" 2>&1
aws s3 cp s3://$S3_BUCKET/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv . >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    log "✗ Failed to download data from S3"
    exit 1
fi

log "✓ Data downloaded successfully"


log "→ Training SARIMAX models (7-month minimum)..."
python3 train_sarimax_incremental.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✓ Models trained successfully"
    
    
    log "→ Uploading models to S3..."
    aws s3 cp sarimax-models-dict.pkl s3://$S3_BUCKET/gold/ >> "$LOG_FILE" 2>&1
    aws s3 cp model-metrics.json s3://$S3_BUCKET/gold/ >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log "✗ Failed to upload models to S3"
        exit 1
    fi
    
    log "✓ Models uploaded to S3"
    
    
    log "→ Generating predictions with updated models..."
    python3 gold_predictions.py >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "✓ Predictions generated successfully"
    else
        log "✗ Prediction generation failed"
        exit 1
    fi
    
    
    log "→ Cleaning up downloaded CSV files..."
    rm -f resale-flat-prices-*.csv
    log "✓ Cleanup completed"
else
    log "✗ Model training failed"
    exit 1
fi


find "$LOG_DIR" -name "weekly_retrain_*.log" -type f -mtime +90 -delete

log "=========================================="
log "✓ WEEKLY RETRAIN COMPLETED SUCCESSFULLY"
log "Models: 7-month minimum, 2-month predictions"
log "=========================================="

exit 0