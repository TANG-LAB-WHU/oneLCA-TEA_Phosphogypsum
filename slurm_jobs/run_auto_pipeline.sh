#!/bin/bash
#SBATCH --job-name=pg_auto_pipeline
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/auto_pipeline_%j.log


#=============================================================================#
# Phosphogypsum Knowledge Graph: Self-Contained Ephemeral Pipeline
# Lifecycle: Boots Neo4j & Milvus -> Probes Readiness -> Parses & Indexes
#            -> Flushes DBs -> Shuts down cleanly (Zero wasted CPU hours)
#=============================================================================#

# Establish robust workspace anchoring
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

PROJECT_ROOT="$(pwd)"
LOG_DIR="$PROJECT_ROOT/logs/slurm"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/datahub/processed"

echo "============================================================"
echo " Phosphogypsum Bot: Autonomous Ephemeral Ingestion Pipeline"
echo " Job ID:       ${SLURM_JOB_ID:-interactive}"
echo " Host Node:    $(hostname)"
echo " Working Dir:  $PROJECT_ROOT"
echo " Logs Dir:     $LOG_DIR"
echo " Start Time:   $(date)"
echo "============================================================"

# 1. Environment Activation
module load apptainer 2>/dev/null || module load singularity 2>/dev/null
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

# 2. Performance & Cache Configuration
export TIKTOKEN_CACHE_DIR="${HOME}/.cache/tiktoken"
export ORT_DISABLE_THREAD_AFFINITY=1
export MINERU_MODEL_SOURCE=local
export LIGHTRAG_GRAPH_STORAGE="Neo4JStorage"
export LIGHTRAG_VECTOR_STORAGE="MilvusVectorDBStorage"

# Dynamically resolve available ports on localhost to prevent "bind: address already in use"
eval $(python - << 'EOF'
import socket

def find_free_port(preferred):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            if s.connect_ex(('127.0.0.1', preferred)) != 0:
                return preferred
    except Exception:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

print(f"export NEO4J_PORT={find_free_port(7687)}")
print(f"export MILVUS_PORT={find_free_port(19530)}")
EOF
)

echo "Assigned Ports: Neo4j Bolt -> $NEO4J_PORT, Milvus gRPC -> $MILVUS_PORT"

# 3. Locate Container Images
CONTAINER_RUNNER="apptainer"
if ! command -v apptainer &> /dev/null; then
    CONTAINER_RUNNER="singularity"
fi

NEO4J_SIF="/scratch/tangsiqi/containers/images/neo4j_5.12.0.sif"
MILVUS_SIF="/scratch/tangsiqi/containers/images/milvus_v2.3.10.sif"

if [ ! -f "$NEO4J_SIF" ] || [ ! -f "$MILVUS_SIF" ]; then
    echo "[ERROR] Pre-built SIF images not found in /scratch/tangsiqi/containers/images/."
    exit 1
fi

# 4. Service Persistence Directories
NEO4J_DATA="$PROJECT_ROOT/datahub/processed/neo4j/data"
NEO4J_LOGS="$PROJECT_ROOT/datahub/processed/neo4j/logs"
NEO4J_CONF="$PROJECT_ROOT/datahub/processed/neo4j/conf"
NEO4J_IMPORT="$PROJECT_ROOT/datahub/processed/neo4j/import"
MILVUS_DATA="$PROJECT_ROOT/datahub/processed/milvus/data"
MILVUS_CONF="$PROJECT_ROOT/datahub/processed/milvus/conf"
MILVUS_YAML="$MILVUS_CONF/milvus.yaml"

mkdir -p "$NEO4J_DATA" "$NEO4J_LOGS" "$NEO4J_CONF" "$NEO4J_IMPORT" "$MILVUS_DATA" "$MILVUS_CONF"

# Seed Neo4j configuration if not present
if [ ! -f "$NEO4J_CONF/neo4j.conf" ]; then
    $CONTAINER_RUNNER exec "$NEO4J_SIF" cp -r /var/lib/neo4j/conf/. "$NEO4J_CONF/" 2>/dev/null || true
fi

# Clean up any stale or invalid settings from previous runs
if [ -f "$NEO4J_CONF/neo4j.conf" ]; then
    sed -i '/^USERNAME=/d' "$NEO4J_CONF/neo4j.conf" 2>/dev/null || true
    sed -i '/^URI=/d' "$NEO4J_CONF/neo4j.conf" 2>/dev/null || true
    sed -i '/^PASSWORD=/d' "$NEO4J_CONF/neo4j.conf" 2>/dev/null || true
fi

# Generate Milvus configuration with dynamic port binding
cat <<EOF > "$MILVUS_YAML"
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
proxy:
  port: ${MILVUS_PORT}
EOF

# Clean up stale locks or pid files if any
rm -f /tmp/milvus/standalone.pid 2>/dev/null || true

# 5. Lifecycle Management: Trap Cleanup
PID_NEO4J=""
PID_MILVUS=""

cleanup() {
    echo ""
    echo "============================================================"
    echo " [TEARDOWN] Stopping ephemeral databases and flushing state..."
    echo "============================================================"
    if [ -n "$PID_NEO4J" ]; then
        kill "$PID_NEO4J" 2>/dev/null
    fi
    if [ -n "$PID_MILVUS" ]; then
        kill "$PID_MILVUS" 2>/dev/null
    fi
    wait "$PID_NEO4J" 2>/dev/null
    wait "$PID_MILVUS" 2>/dev/null
    echo " Databases cleanly shutdown. Data safely preserved on disk."
    echo " Finished at: $(date)"
}
trap cleanup EXIT SIGINT SIGTERM

# 6. Launch Databases in Background (Localhost Binding)
echo "[1/4] Starting Neo4j background container..."
export NEO4J_AUTH="neo4j/password123"
export NEO4J_server_default__listen__address="0.0.0.0"
export NEO4J_server_bolt_listen__address="0.0.0.0:${NEO4J_PORT}"
export NEO4J_server_config_strict__validation_enabled="false"
export TINI_SUBREAPER=1
export NEO4J_server_memory_heap_initial__size="2G"
export NEO4J_server_memory_heap_max__size="8G"
export NEO4J_server_memory_pagecache_size="4G"

# Launch container with clean environment (prevent host NEO4J_USERNAME from polluting neo4j.conf)
env -u NEO4J_USERNAME -u NEO4J_URI -u NEO4J_PASSWORD \
$CONTAINER_RUNNER run \
    --writable-tmpfs \
    --bind "$NEO4J_DATA":/data \
    --bind "$NEO4J_LOGS":/logs \
    --bind "$NEO4J_CONF":/var/lib/neo4j/conf \
    --bind "$NEO4J_IMPORT":/import \
    "$NEO4J_SIF" > "$LOG_DIR/auto_neo4j_${SLURM_JOB_ID}.log" 2>&1 &
PID_NEO4J=$!

echo "[2/4] Starting Milvus Standalone background container..."
$CONTAINER_RUNNER run \
    --writable-tmpfs \
    --bind "$MILVUS_DATA":/var/lib/milvus \
    --bind "$MILVUS_YAML":/milvus/configs/milvus.yaml \
    "$MILVUS_SIF" milvus run standalone > "$LOG_DIR/auto_milvus_${SLURM_JOB_ID}.log" 2>&1 &
PID_MILVUS=$!

# 7. Pre-flight Readiness Check (Poll up to 240 seconds)
echo "[3/4] Probing database readiness on localhost..."
python - << 'EOF'
import socket, time, sys, os

def wait_port(port, name, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        res = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if res == 0:
            print(f"  [OK] {name} is ready on 127.0.0.1:{port} (elapsed: {int(time.time()-start)}s)")
            return True
        time.sleep(2)
    print(f"  [TIMEOUT] {name} failed to become ready on port {port} within {timeout}s.")
    return False

neo_port = int(os.environ.get("NEO4J_PORT", 7687))
mil_port = int(os.environ.get("MILVUS_PORT", 19530))

if not (wait_port(neo_port, "Neo4j") and wait_port(mil_port, "Milvus")):
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "[ABORT] Database services failed to initialize in time."
    exit 1
fi

# 8. Execute Ingestion Pipeline
echo "[4/4] Executing parsing and indexing pipeline..."

echo "--- Phase A: PDF Parsing (MinerU) ---"
python scripts/build_knowledge_graph.py \
  --step parse \
  --parser mineru

echo "--- Phase B: LightRAG Indexing (Graph + Vector) ---"
export NEO4J_URI="bolt://127.0.0.1:${NEO4J_PORT}"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password123"
export MILVUS_URI="http://127.0.0.1:${MILVUS_PORT}"
export MILVUS_DB_NAME="lightrag"

python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag

echo ""
echo "============================================================"
echo " All ingestion steps completed successfully!"
echo "============================================================"
# Script exits normally, triggering trap cleanup to shutdown DBs
exit 0
