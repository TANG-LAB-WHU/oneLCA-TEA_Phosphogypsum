#!/bin/bash
#SBATCH --job-name=pg_build_kg_9a14a
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=21-00:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/slurm/build_kg_9a14a_%j.log
#SBATCH --error=logs/slurm/build_kg_9a14a_%j.err

# =============================================================================
# Phosphogypsum Bot: Full Knowledge Graph & Vector Index Pipeline (9a14a CPU)
# Hardware: 192 AMD EPYC CPU Cores with Dual-Socket NUMA Hardware Isolation
#
# Process Sequence:
#   1. Start Qwen3.6-35B Reasoner on NUMA Socket 0 (CPU 0-95, 80 threads, Port 11434)
#   2. Start Qwen3-Embedding-8B on NUMA Socket 1 (CPU 96-191, 80 threads, Port 11436)
#   3. Wait for healthcheck readiness
#   4. Step 2 (index): Full 72-paper LightRAG graph & vector index construction
#   5. Step 3 (extract): High-fidelity LLM engineering parameter JSON extraction
#   6. Step 4 (ranges): Global empirical parameter bounds & uncertainty synthesis
#   7. Step 5 (build): Assembly of queryable Phosphogypsum Knowledge Graph
#   8. Clean shutdown of background model servers
# =============================================================================

# 1. Environment Initialization
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

# 2. Performance & Cache Configuration
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"
export ORT_DISABLE_THREAD_AFFINITY=1
export CUDA_VISIBLE_DEVICES=""          # Strict CPU mode for Python workers

# Ensure working directory is project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

PROJECT_ROOT="$(pwd)"
echo "============================================================"
echo "  Phosphogypsum Knowledge Graph Full Pipeline (9a14a CPU)"
echo "  Job ID:       ${SLURM_JOB_ID:-manual}"
echo "  Compute Node: $(hostname)"
echo "  Partition:    9a14a (192 Cores, NUMA Dual-Socket)"
echo "  Project Root: $PROJECT_ROOT"
echo "  Start Time:   $(date)"
echo "============================================================"

# Ensure directories exist
mkdir -p logs/slurm logs/services \
         datahub/processed/lightrag_db \
         datahub/processed/extracted_data \
         datahub/processed/parameter_ranges \
         datahub/processed/knowledge_graph

# 3. Model Server Binaries & Model Weights
LLAMA_SERVER_BIN="/project/tangsiqi/software/llama.cpp/build_cpu/bin/llama-server"
REASONER_MODEL="/home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
REASONER_MMPROJ="/home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf"
EMBED_MODEL="/home/tangsiqi/scratch/ai_models/qwen/Qwen3-Embedding-8B-Q8_0.gguf"

if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    echo "[FATAL] llama-server CPU binary not found at: $LLAMA_SERVER_BIN"
    exit 1
fi

PID_REASONER=""
PID_EMBED=""

# 4. Lifecycle Management: Clean Signal Propagation & Shutdown Trap
cleanup() {
    echo ""
    echo "============================================================"
    echo " [TEARDOWN] Stopping background llama-server instances..."
    echo "============================================================"
    if [ -n "$PID_REASONER" ] && kill -0 "$PID_REASONER" 2>/dev/null; then
        echo "  - Terminating Reasoner Server (PID: $PID_REASONER)..."
        kill "$PID_REASONER" 2>/dev/null
    fi
    if [ -n "$PID_EMBED" ] && kill -0 "$PID_EMBED" 2>/dev/null; then
        echo "  - Terminating Embedding Server (PID: $PID_EMBED)..."
        kill "$PID_EMBED" 2>/dev/null
    fi
    wait "$PID_REASONER" 2>/dev/null
    wait "$PID_EMBED" 2>/dev/null
    echo " [TEARDOWN] Background services stopped cleanly."
}
trap cleanup EXIT SIGINT SIGTERM

# 5. Launch Dual Model Servers via NUMA Hardware Isolation
echo ""
echo "[1/4] Launching Dual NUMA Model Servers..."

# Server A: Reasoner LLM on Socket 0 (CPU Cores 0-95, Mem Node 0)
echo "  - Starting Qwen3.6-35B Reasoner on Socket 0 (Port 11434, 80 threads)..."
numactl --cpunodebind=0 --membind=0 \
  "$LLAMA_SERVER_BIN" \
  --model "$REASONER_MODEL" \
  --mmproj "$REASONER_MMPROJ" \
  --host 127.0.0.1 \
  --port 11434 \
  --threads 80 \
  --ctx-size 65536 \
  --parallel 4 \
  --n-gpu-layers 0 > "logs/services/reasoner_server_${SLURM_JOB_ID}.log" 2>&1 &
PID_REASONER=$!

# Server B: Embedding Model on Socket 1 (CPU Cores 96-191, Mem Node 1)
echo "  - Starting Qwen3-Embedding-8B on Socket 1 (Port 11436, 80 threads)..."
numactl --cpunodebind=1 --membind=1 \
  "$LLAMA_SERVER_BIN" \
  --model "$EMBED_MODEL" \
  --embedding \
  --host 127.0.0.1 \
  --port 11436 \
  --threads 80 \
  --ctx-size 32768 \
  --parallel 8 \
  --n-gpu-layers 0 > "logs/services/embed_server_${SLURM_JOB_ID}.log" 2>&1 &
PID_EMBED=$!

# 6. Pre-flight Readiness Check (Poll every 5s, max 10 minutes)
echo ""
echo "[2/4] Awaiting model server health status..."
TIMEOUT=600
INTERVAL=5
ELAPSED=0

while true; do
    # Verify process vitality
    if ! kill -0 "$PID_REASONER" 2>/dev/null; then
        echo "[FATAL] Reasoner server process (PID $PID_REASONER) died unexpectedly!"
        echo "===== Reasoner Server Log ====="
        tail -n 30 "logs/services/reasoner_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    if ! kill -0 "$PID_EMBED" 2>/dev/null; then
        echo "[FATAL] Embedding server process (PID $PID_EMBED) died unexpectedly!"
        echo "===== Embedding Server Log ====="
        tail -n 30 "logs/services/embed_server_${SLURM_JOB_ID}.log"
        exit 1
    fi

    REASONER_READY=0
    EMBED_READY=0

    if curl -s -f http://127.0.0.1:11434/health > /dev/null 2>&1; then
        REASONER_READY=1
    fi
    if curl -s -f http://127.0.0.1:11436/health > /dev/null 2>&1; then
        EMBED_READY=1
    fi

    if [ "$REASONER_READY" -eq 1 ] && [ "$EMBED_READY" -eq 1 ]; then
        echo "  [OK] Both Reasoner (11434) and Embedding (11436) servers active and healthy! (Elapsed: ${ELAPSED}s)"
        break
    fi

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[TIMEOUT] Model servers failed to reach healthy state within ${TIMEOUT}s."
        exit 1
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  ... Initializing weights: ${ELAPSED}/${TIMEOUT}s (Reasoner: $REASONER_READY, Embed: $EMBED_READY)"
    fi
done

# 7. Configure Environment for Knowledge Graph Engine
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export EMBEDDING_BASE_URL="http://127.0.0.1:11436/v1"
export LLM_TIMEOUT=3600
export EMBEDDING_TIMEOUT=1200

# 8. Execute Full Knowledge Pipeline (All 72 Papers, No Limit)
echo ""
echo "[3/4] Executing Full-Scale Knowledge Graph & Parameter Synthesis..."

# Step 2: Index all parsed papers into LightRAG (Vector + Graph)
echo ""
echo ">>> [Phase 1/4] Step 2: Full LightRAG Index Construction (72 Papers) <<<"
python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag

INDEX_STATUS=$?
if [ $INDEX_STATUS -ne 0 ]; then
    echo "[WARN] Step 2 (Index) finished with non-zero exit code: $INDEX_STATUS"
fi

# Step 3: Extract structured parameters JSON via Reasoner LLM
echo ""
echo ">>> [Phase 2/4] Step 3: Structured Engineering Parameter Extraction <<<"
python scripts/build_knowledge_graph.py \
  --step extract \
  --engine lightrag

EXTRACT_STATUS=$?
if [ $EXTRACT_STATUS -ne 0 ]; then
    echo "[WARN] Step 3 (Extract) finished with non-zero exit code: $EXTRACT_STATUS"
fi

# Step 4: Synthesize global parameter distributions for MCMC / Reverse Design
echo ""
echo ">>> [Phase 3/4] Step 4: Parameter Range & Uncertainty Synthesis <<<"
python scripts/build_knowledge_graph.py \
  --step ranges

RANGES_STATUS=$?

# Step 5: Assemble unified queryable Phosphogypsum Knowledge Graph
echo ""
echo ">>> [Phase 4/4] Step 5: Knowledge Graph Assembly <<<"
python scripts/build_knowledge_graph.py \
  --step build

BUILD_STATUS=$?

# 9. Pipeline Completion Summary
echo ""
echo "============================================================"
echo "  [4/4] Full Pipeline Execution Summary"
echo "  - LightRAG Index (Step 2):  Exit $INDEX_STATUS"
echo "  - Data Extract (Step 3):    Exit $EXTRACT_STATUS"
echo "  - Parameter Ranges (Step 4): Exit $RANGES_STATUS"
echo "  - Knowledge Graph (Step 5):  Exit $BUILD_STATUS"
echo "============================================================"

# Verification of output files
PARSED_COUNT=$(find datahub/interim/papers/parsed -name "*.md" 2>/dev/null | wc -l)
JSON_COUNT=$(find datahub/processed/extracted_data -name "*_extracted.json" 2>/dev/null | wc -l)
RANGE_COUNT=$(find datahub/processed/parameter_ranges -name "*.json" 2>/dev/null | wc -l)

echo "  Output Metrics on Disk:"
echo "    * Input Parsed Papers:      $PARSED_COUNT / 72"
echo "    * Extracted Parameter Sets: $JSON_COUNT"
echo "    * Synthesized Range Files:  $RANGE_COUNT"
echo "  Finished at: $(date)"
echo "============================================================"

exit 0
