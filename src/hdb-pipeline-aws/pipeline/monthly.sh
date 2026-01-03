
set -e

PIPELINE_DIR="/home/ubuntu/pipeline"
LOG_DIR="/home/ubuntu/logs"
LOG_FILE="$LOG_DIR/daily_pipeline_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

source /home/ubuntu/miniconda3/bin/activate hdb-env

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "MONTHLY PREDICTION PIPELINE"
log "=========================================="


log "→ Running bronze layer (data ingestion)..."
cd "$PIPELINE_DIR"
python3 bronze_ingestion.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✓ Bronze layer completed"
else
    log "✗ Bronze layer failed"
    exit 1
fi


log "→ Running silver layer (aggregation)..."
python3 silver_features.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✓ Silver layer completed"
else
    log "✗ Silver layer failed"
    exit 1
fi


log "→ Running gold layer (predictions)..."
python3 gold_predictions.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✓ Gold layer completed"
else
    log "✗ Gold layer failed"
    exit 1
fi


find "$LOG_DIR" -name "daily_pipeline_*.log" -type f -mtime +30 -delete

log "=========================================="
log "✓ DAILY PIPELINE COMPLETED SUCCESSFULLY"
log "=========================================="

exit 0