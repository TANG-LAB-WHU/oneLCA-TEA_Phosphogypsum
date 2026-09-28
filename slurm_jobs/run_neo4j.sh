#!/bin/bash
#SBATCH --job-name=neo4j_service
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --output=logs/slurm/neo4j_%j.log


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
DATA_DIR="${NEO4J_DATA_DIR:-$(pwd)/datahub/processed/neo4j/data}"
LOGS_DIR="$(pwd)/datahub/processed/neo4j/logs"
CONF_DIR="$(pwd)/datahub/processed/neo4j/conf"
IMPORT_DIR="$(pwd)/datahub/processed/neo4j/import"
ENDPOINT_FILE="$(pwd)/datahub/processed/.neo4j_endpoint"

mkdir -p "$DATA_DIR" "$LOGS_DIR" "$CONF_DIR" "$IMPORT_DIR"

HOST_NAME=$(hostname)
BOLT_PORT=7687
HTTP_PORT=7474
echo "============================================================"
echo " Starting Neo4j Database Service via Singularity/Apptainer"
echo " Job ID:   ${SLURM_JOB_ID:-interactive}"
echo " Host:     $HOST_NAME"
echo " Ports:    $HTTP_PORT (HTTP), $BOLT_PORT (Bolt)"
echo " Data Dir: $DATA_DIR"
echo "============================================================"

# Publish active endpoint for worker auto-discovery seam
echo "bolt://${HOST_NAME}:${BOLT_PORT}" > "$ENDPOINT_FILE"
echo "Published endpoint: bolt://${HOST_NAME}:${BOLT_PORT} -> $ENDPOINT_FILE"

# Clean up published endpoint on job exit
cleanup() {
    echo "Cleaning up Neo4j endpoint registration..."
    rm -f "$ENDPOINT_FILE"
}
trap cleanup EXIT SIGINT SIGTERM

# Set up default credentials, network binding, and Tini subreaper
export NEO4J_AUTH="${NEO4J_AUTH:-neo4j/password123}"
export NEO4J_server_default__listen__address="0.0.0.0"
export TINI_SUBREAPER=1

# JVM Heap and Pagecache Tuning (safe bounds within 64G allocation)
export NEO4J_server_memory_heap_initial__size="${NEO4J_HEAP_INIT:-4G}"
export NEO4J_server_memory_heap_max__size="${NEO4J_HEAP_MAX:-16G}"
export NEO4J_server_memory_pagecache_size="${NEO4J_PAGECACHE:-8G}"

# Ensure container module is loaded
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

# Run Neo4j container using Apptainer (or fallback to Singularity)
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
NEO4J_SIF="/scratch/tangsiqi/containers/images/neo4j_5.12.0.sif"
if [ -f "$NEO4J_SIF" ]; then
    IMAGE_TARGET="$NEO4J_SIF"
else
    IMAGE_TARGET="docker://neo4j:5.12.0"
fi
echo "Using Neo4j image: $IMAGE_TARGET"

# Seed default configurations from SIF to host $CONF_DIR if not present
if [ ! -f "$CONF_DIR/neo4j.conf" ]; then
    echo "Seeding default configuration into $CONF_DIR from container..."
    $CONTAINER_RUNNER exec "$IMAGE_TARGET" cp -r /var/lib/neo4j/conf/. "$CONF_DIR/" 2>/dev/null || true
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

            # Check 2: Established TCP connections to Neo4j Bolt port
            ACTIVE_CONNS=$(ss -nt "sport = :$BOLT_PORT" 2>/dev/null | grep -v "State" | wc -l)
            if [ "$ACTIVE_CONNS" -eq 0 ]; then
                ACTIVE_CONNS=$(netstat -tn 2>/dev/null | grep ":$BOLT_PORT " | grep -c "ESTABLISHED" || true)
            fi

            # Check 3: Keepalive override file
            KEEPALIVE_FILE="$(pwd)/datahub/processed/.keepalive"

            if [ "$ACTIVE_JOBS" -gt 0 ] || [ "$ACTIVE_CONNS" -gt 0 ] || [ -f "$KEEPALIVE_FILE" ]; then
                IDLE_SECS=0
            else
                IDLE_SECS=$((IDLE_SECS + CHECK_INTERVAL))
                if [ $((IDLE_SECS % 300)) -eq 0 ]; then
                    echo "[Watchdog] Neo4j has been idle for $((IDLE_SECS / 60)) / ${IDLE_TIMEOUT_MINUTES} minutes..."
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


# Run container in foreground with exec for direct SIGTERM signal forwarding.
# We mount $CONF_DIR directly to /var/lib/neo4j/conf (avoiding /conf which triggers destructive rm)
# and use --writable-tmpfs to allow entrypoint scripts to make runtime modifications safely.
exec $CONTAINER_RUNNER run \
    --writable-tmpfs \
    --bind "$DATA_DIR":/data \
    --bind "$LOGS_DIR":/logs \
    --bind "$CONF_DIR":/var/lib/neo4j/conf \
    --bind "$IMPORT_DIR":/import \
    "$IMAGE_TARGET"



