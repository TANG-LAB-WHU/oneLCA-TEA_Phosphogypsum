#!/bin/bash

#=============================================================================#
# Phosphogypsum Bot: Smart Ingestion Dispatcher & Service Manager
# A unified CLI facade to minimize HPC core-hour waste and manage workloads.
#=============================================================================#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$PROJECT_ROOT"

NEO4J_ENDPOINT_FILE="$PROJECT_ROOT/datahub/processed/.neo4j_endpoint"
MILVUS_ENDPOINT_FILE="$PROJECT_ROOT/datahub/processed/.milvus_endpoint"
UNPARSED_DIR="$PROJECT_ROOT/datahub/raw/papers/unparsed"
REGISTRY_FILE="$PROJECT_ROOT/datahub/processed/ingestion_registry.json"

show_help() {
    cat << EOF
Phosphogypsum Knowledge Graph Ingestion Dispatcher

Usage:
  ./scripts/submit_ingestion.sh [COMMAND] [OPTIONS]

Commands:
  (default)          Smart analyze pending papers and submit optimal pipeline
  --auto, -a         Submit cost-effective single-node self-contained pipeline (Zero wasted hours)
  --distributed, -d  Submit high-throughput multi-node array pipeline (for 50+ papers)
  --status, -s       Inspect active databases, endpoints, and running Slurm jobs
  --stop             Cancel all active database services and ingestion tasks
  --help, -h         Show this help message

Examples:
  ./scripts/submit_ingestion.sh             # Smart recommendation based on workload
  ./scripts/submit_ingestion.sh --status    # Check who is running
  ./scripts/submit_ingestion.sh --stop      # Clean shutdown

EOF
}

check_status() {
    echo "============================================================"
    echo " Phosphogypsum Knowledge Graph: Service & Job Status"
    echo "============================================================"
    
    echo "--- 1. Slurm Job Queue ---"
    squeue -u "$USER" -o "%.10i %.15j %.8u %.8T %.10M %.8l %.6D %R" | grep -E "JOBID|neo4j|milvus|pg_|Phosphogypsum" || echo "  No active Phosphogypsum jobs found."
    
    echo ""
    echo "--- 2. Registered Service Endpoints ---"
    if [ -f "$NEO4J_ENDPOINT_FILE" ]; then
        echo "  [Neo4j] Active at: $(cat "$NEO4J_ENDPOINT_FILE")"
    else
        echo "  [Neo4j] Offline (No active endpoint registered)"
    fi

    if [ -f "$MILVUS_ENDPOINT_FILE" ]; then
        echo "  [Milvus] Active at: $(cat "$MILVUS_ENDPOINT_FILE")"
    else
        echo "  [Milvus] Offline (No active endpoint registered)"
    fi

    echo ""
    echo "--- 3. Literature Ingestion State ---"
    TOTAL_PDFS=$(find "$UNPARSED_DIR" -name "*.pdf" 2>/dev/null | wc -l)
    echo "  Total PDFs in unparsed directory: $TOTAL_PDFS"
    if [ -f "$REGISTRY_FILE" ]; then
        PROCESSED_COUNT=$(grep -o '"status": "success"' "$REGISTRY_FILE" 2>/dev/null | wc -l)
        echo "  Successfully registered papers:   $PROCESSED_COUNT"
    fi
}

stop_all() {
    echo "============================================================"
    echo " Stopping all active Phosphogypsum services and jobs..."
    echo "============================================================"
    JOB_IDS=$(squeue -u "$USER" -h -o "%i %j" | grep -E "neo4j_service|milvus_service|pg_parallel_ingest|pg_auto_pipeline" | awk '{print $1}')
    if [ -n "$JOB_IDS" ]; then
        echo "Canceling Slurm Job IDs: $JOB_IDS"
        scancel $JOB_IDS
    else
        echo "No active jobs found in queue."
    fi
    rm -f "$NEO4J_ENDPOINT_FILE" "$MILVUS_ENDPOINT_FILE"
    echo "Endpoints cleaned up."
}

submit_auto() {
    echo "Submitting self-contained ephemeral pipeline (slurm_jobs/run_auto_pipeline.sh)..."
    sbatch slurm_jobs/run_auto_pipeline.sh
}

submit_distributed() {
    echo "Submitting distributed pipeline with autonomous idle watchdog..."
    
    # 1. Start Neo4j if not running
    if ! squeue -u "$USER" -h -o "%j" | grep -q "neo4j_service"; then
        echo "Launching Neo4j service..."
        sbatch slurm_jobs/run_neo4j.sh
    else
        echo "Neo4j is already running."
    fi

    # 2. Start Milvus if not running
    if ! squeue -u "$USER" -h -o "%j" | grep -q "milvus_service"; then
        echo "Launching Milvus service..."
        sbatch slurm_jobs/run_milvus.sh
    else
        echo "Milvus is already running."
    fi

    # 3. Wait briefly for endpoint files to be published
    echo "Waiting for endpoints to register..."
    for i in {1..30}; do
        if [ -f "$NEO4J_ENDPOINT_FILE" ] && [ -f "$MILVUS_ENDPOINT_FILE" ]; then
            break
        fi
        sleep 2
    done

    # 4. Submit ingestion array
    echo "Launching parallel ingestion worker array..."
    sbatch slurm_jobs/run_parallel_ingestion.sh
}

smart_dispatch() {
    TOTAL_PDFS=$(find "$UNPARSED_DIR" -name "*.pdf" 2>/dev/null | wc -l)
    
    echo "============================================================"
    echo " Analyzing Workload & Compute Cost Optimization"
    echo " Total raw PDFs found: $TOTAL_PDFS"
    echo "============================================================"

    # If both DBs are ALREADY active, directly run ingestion array!
    if squeue -u "$USER" -h -o "%j" | grep -q "neo4j_service" && squeue -u "$USER" -h -o "%j" | grep -q "milvus_service"; then
        echo "[Mode: Incremental Worker] Active database services detected."
        echo "Submitting parallel worker array directly..."
        sbatch slurm_jobs/run_parallel_ingestion.sh
        return 0
    fi

    # If small to medium workload, recommend single-node all-in-one pipeline
    if [ "$TOTAL_PDFS" -le 35 ]; then
        echo "[Recommendation] Workload is light ($TOTAL_PDFS papers)."
        echo "Using single-node ephemeral pipeline: zero queue overhead & zero idle waste."
        submit_auto
    else
        echo "[Recommendation] Workload is large ($TOTAL_PDFS papers)."
        echo "Using multi-node distributed array with autonomous idle-timeout watchdogs."
        submit_distributed
    fi
}

# Command dispatching
case "$1" in
    --status|-s)
        check_status
        ;;
    --stop)
        stop_all
        ;;
    --auto|-a)
        submit_auto
        ;;
    --distributed|-d)
        submit_distributed
        ;;
    --help|-h)
        show_help
        ;;
    "")
        smart_dispatch
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
