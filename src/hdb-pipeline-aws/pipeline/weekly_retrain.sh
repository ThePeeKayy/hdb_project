set -e  

PIPELINE_DIR="/home/ubuntu/pipeline"
LOG_DIR="/home/ubuntu/logs"
LOG_FILE="$LOG_DIR/weekly_retrain_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting Weekly Model Retraining"
log "=========================================="

source /home/ubuntu/miniconda3/bin/activate hdb-env

cd "$PIPELINE_DIR"


bash daily_pipeline.sh >> "$LOG_FILE" 2>&1


log "Training new TiDE model..."
python3 train_tide_model.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    
    log "Generating predictions with new model..."
    python3 gold_predictions.py >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "Predictions generated successfully"
    else
        log "Prediction generation failed"
        exit 1
    fi
else
    log "✗ Model training failed"
    exit 1
fi


find "$LOG_DIR" -name "weekly_retrain_*.log" -type f -mtime +90 -delete

exit 0