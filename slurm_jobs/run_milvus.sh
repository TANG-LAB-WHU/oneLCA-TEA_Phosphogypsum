#!/bin/bash
#SBATCH --job-name=milvus_service
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/milvus_%j.log




# Change to project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

# Ensure logs and coordination directories exist
mkdir -p logs/slurm datahub/processed

# Set directory variables for persistence (allow override to scratch for massive graphs)
DATA_DIR="${MILVUS_DATA_DIR:-$(pwd)/datahub/processed/milvus/data}"
CONF_DIR="$(pwd)/datahub/processed/milvus/conf"
ENDPOINT_FILE="$(pwd)/datahub/processed/.milvus_endpoint"

mkdir -p "$DATA_DIR" "$CONF_DIR"

# Generate milvus.yaml if it doesn't exist (to run standalone inside container)
MILVUS_YAML="$CONF_DIR/milvus.yaml"
if [ ! -f "$MILVUS_YAML" ]; then
    echo "Creating default milvus.yaml configuration..."
    cat <<EOF > "$MILVUS_YAML"
# Standalone Milvus Configuration
etcd:
  use: embed
  data.dir: /var/lib/milvus/etcd
metastore:
  type: sqlite
storage:
  path: /var/lib/milvus/data
queryNode:
  gracefulTimeOut: 0
dataNode:
  gracefulTimeOut: 0
EOF
fi

HOST_NAME=$(hostname)
MILVUS_PORT=19530
echo "============================================================"
echo " Starting Milvus Standalone Service via Singularity/Apptainer"
echo " Job ID:   ${SLURM_JOB_ID:-interactive}"
echo " Host:     $HOST_NAME"
echo " Ports:    $MILVUS_PORT (gRPC), 9091 (REST API)"
echo " Data Dir: $DATA_DIR"
echo "============================================================"

# Publish active endpoint for worker auto-discovery seam
echo "http://${HOST_NAME}:${MILVUS_PORT}" > "$ENDPOINT_FILE"
echo "Published endpoint: http://${HOST_NAME}:${MILVUS_PORT} -> $ENDPOINT_FILE"

# Clean up published endpoint on job exit
cleanup() {
    echo "Cleaning up Milvus endpoint registration..."
    rm -f "$ENDPOINT_FILE"
}
trap cleanup EXIT SIGINT SIGTERM

# Ensure container module is loaded
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

# Run Milvus container using Apptainer (or fallback to Singularity)
if command -v apptainer &> /dev/null; then
    CONTAINER_RUNNER="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_RUNNER="singularity"
else
    echo "[ERROR] Neither Apptainer nor Singularity found in path."
    exit 1
fi

echo "Using container runner: $CONTAINER_RUNNER"

# Use pre-built static SIF image from scratch to avoid network pull & quota issues
MILVUS_SIF="/scratch/tangsiqi/containers/images/milvus_v2.3.10.sif"
if [ -f "$MILVUS_SIF" ]; then
    IMAGE_TARGET="$MILVUS_SIF"
else
    IMAGE_TARGET="docker://milvusdb/milvus:v2.3.10"
fi
# =============================================================================
# Background Idle-Timeout Watchdog (Autonomous Quota Guard)
# =============================================================================
IDLE_TIMEOUT_MINUTES="${AUTO_SHUTDOWN_TIMEOUT:-30}"
if [ "$IDLE_TIMEOUT_MINUTES" -gt 0 ] && [ -n "$SLURM_JOB_ID" ]; then
    (
        GRACE_SECONDS=1800  # 30-min initial grace period on startup
        echo "[Watchdog] Auto-shutdown watchdog active (Grace: ${GRACE_SECONDS}s, Idle timeout: ${IDLE_TIMEOUT_MINUTES}m)"
        sleep "$GRACE_SECONDS"

        IDLE_SECS=0
        CHECK_INTERVAL=60
        MAX_IDLE_SECS=$((IDLE_TIMEOUT_MINUTES * 60))

        while true; do
            sleep "$CHECK_INTERVAL"

            # Check 1: Ingestion or Agent Slurm jobs running by this user
            ACTIVE_JOBS=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -iE "pg_parallel_ingest|pg_auto_pipeline|Phosphogypsum|kg_test" | wc -l)

            # Check 2: Established TCP connections to Milvus gRPC port
            ACTIVE_CONNS=$(ss -nt "sport = :$MILVUS_PORT" 2>/dev/null | grep -v "State" | wc -l)
            if [ "$ACTIVE_CONNS" -eq 0 ]; then
                ACTIVE_CONNS=$(netstat -tn 2>/dev/null | grep ":$MILVUS_PORT " | grep -c "ESTABLISHED" || true)
            fi

            # Check 3: Keepalive override file
            KEEPALIVE_FILE="$(pwd)/datahub/processed/.keepalive"

            if [ "$ACTIVE_JOBS" -gt 0 ] || [ "$ACTIVE_CONNS" -gt 0 ] || [ -f "$KEEPALIVE_FILE" ]; then
                IDLE_SECS=0
            else
                IDLE_SECS=$((IDLE_SECS + CHECK_INTERVAL))
                if [ $((IDLE_SECS % 300)) -eq 0 ]; then
                    echo "[Watchdog] Milvus has been idle for $((IDLE_SECS / 60)) / ${IDLE_TIMEOUT_MINUTES} minutes..."
                fi

                if [ "$IDLE_SECS" -ge "$MAX_IDLE_SECS" ]; then
                    echo "============================================================"
                    echo "[AUTO-SHUTDOWN] No active client connections or ingestion jobs"
                    echo "                detected for ${IDLE_TIMEOUT_MINUTES} minutes."
                    echo "                Terminating Slurm Job $SLURM_JOB_ID to save cluster quota."
                    echo "============================================================"
                    scancel "$SLURM_JOB_ID"
                    exit 0
                fi
            fi
        done
    ) &
    WATCHDOG_PID=$!
    echo "Watchdog background process launched (PID: $WATCHDOG_PID)"
fi

# Run Milvus standalone in container using exec for clean signal propagation (SIGTERM)

exec $CONTAINER_RUNNER run \
    --writable-tmpfs \
    --bind "$DATA_DIR":/var/lib/milvus \
    --bind "$MILVUS_YAML":/milvus/configs/milvus.yaml \
    "$IMAGE_TARGET" \
    milvus run standalone



