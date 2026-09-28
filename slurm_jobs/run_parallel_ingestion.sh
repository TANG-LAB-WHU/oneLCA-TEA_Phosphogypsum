#!/bin/bash
#SBATCH --job-name=pg_parallel_ingest
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-49%8             # Launch 50 tasks with max 8 concurrent to avoid DB connection exhaustion
#SBATCH --output=logs/slurm/ingest_array_%A_%a.log


# Note: %A is main Job ID, %a is Slurm Array Task ID

# Load CUDA environment
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

# Change to project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

echo "============================================================"
echo " Phosphogypsum Bot Parallel Ingestion Worker"
echo " Array Job ID: ${SLURM_ARRAY_JOB_ID:-manual} | Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo " Host: $(hostname)"
echo " Partition: 9a14a"
echo " Working Dir: $(pwd)"
echo "============================================================"

# Ensure logs dir exists
mkdir -p logs/slurm

# Set shared tiktoken cache directory
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"

# Silence ONNX Runtime thread affinity warnings under Slurm cgroups
export ORT_DISABLE_THREAD_AFFINITY=1

# Ingestion configuration
export MINERU_MODEL_SOURCE=local

# =============================================================================
# Database Coordinates & Auto-Discovery Seam
# =============================================================================
# If URI is default/localhost, attempt auto-discovery from active service registry
if [ -z "$NEO4J_URI" ] || [ "$NEO4J_URI" = "bolt://localhost:7687" ]; then
    if [ -f "datahub/processed/.neo4j_endpoint" ]; then
        export NEO4J_URI=$(cat datahub/processed/.neo4j_endpoint)
        echo "[Discovery] Auto-detected active Neo4j endpoint: $NEO4J_URI"
    fi
fi

if [ -z "$MILVUS_URI" ] || [ "$MILVUS_URI" = "http://localhost:19530" ]; then
    if [ -f "datahub/processed/.milvus_endpoint" ]; then
        export MILVUS_URI=$(cat datahub/processed/.milvus_endpoint)
        echo "[Discovery] Auto-detected active Milvus endpoint: $MILVUS_URI"
    fi
fi

export LIGHTRAG_GRAPH_STORAGE="Neo4JStorage"
export LIGHTRAG_VECTOR_STORAGE="MilvusVectorDBStorage"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"
export MILVUS_URI="${MILVUS_URI:-http://localhost:19530}"
export MILVUS_DB_NAME="${MILVUS_DB_NAME:-lightrag}"

# =============================================================================
# Pre-flight Database Reachability Check
# =============================================================================
echo "Verifying database service reachability..."
python - << 'EOF'
import os, sys, urllib.parse, socket

def check_conn(uri_str, name):
    try:
        parsed = urllib.parse.urlparse(uri_str)
        host = parsed.hostname or "localhost"
        port = parsed.port or (7687 if "bolt" in parsed.scheme else 19530)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        result = sock.connect_ex((host, port))
        sock.close()
        if result == 0:
            print(f"  [OK] {name} reachable at {host}:{port}")
            return True
        else:
            print(f"  [FAIL] Cannot connect to {name} at {host}:{port} (socket code {result})")
            return False
    except Exception as e:
        print(f"  [FAIL] Error probing {name} ({uri_str}): {e}")
        return False

neo_ok = check_conn(os.getenv("NEO4J_URI", "bolt://localhost:7687"), "Neo4j")
mil_ok = check_conn(os.getenv("MILVUS_URI", "http://localhost:19530"), "Milvus")

if not (neo_ok and mil_ok):
    print("\n[CRITICAL] Database services unreachable.")
    print("Ensure 'sbatch slurm_jobs/run_neo4j.sh' and 'sbatch slurm_jobs/run_milvus.sh' are RUNNING.")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "[ABORT] Aborting task ${SLURM_ARRAY_TASK_ID:-0} due to unreachable database backend."
    exit 1
fi

# =============================================================================
# Run Sharded Parsing and Indexing Task
# =============================================================================
# Step 1: Parse PDFs assigned to this array shard
python scripts/build_knowledge_graph.py \
  --step parse \
  --parser mineru

# Wait briefly proportional to task ID to avoid thundering herd on DB startup
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 2 ))

# Step 2: Index parsed documents into LightRAG (Milvus + Neo4j)
python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag

