
set -e

PIPELINE_DIR="/home/ubuntu/pipeline"
LOG_DIR="/home/ubuntu/logs"
LOG_FILE="$LOG_DIR/weekly_retrain_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

source /home/ubuntu/miniconda3/bin/activate hdb-env

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "WEEKLY MODEL RETRAIN"
log "=========================================="

cd "$PIPELINE_DIR"


log "→ Running daily pipeline first..."
bash daily_pipeline.sh >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    log "✗ Daily pipeline failed"
    exit 1
fi

log "✓ Daily pipeline completed"


log "→ Running incremental model update..."
python3 train_tide_model.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✓ Model updated successfully"
    
    
    log "→ Generating predictions with updated model..."
    python3 gold_predictions.py >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        log "✓ Predictions generated successfully"
    else
        log "✗ Prediction generation failed"
        exit 1
    fi
else
    log "✗ Model update failed"
    exit 1
fi


find "$LOG_DIR" -name "weekly_retrain_*.log" -type f -mtime +90 -delete

log "=========================================="
log "✓ WEEKLY RETRAIN COMPLETED SUCCESSFULLY"
log "=========================================="

exit 0