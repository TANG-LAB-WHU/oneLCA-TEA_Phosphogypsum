#!/bin/bash
#SBATCH --job-name=pg_extract_ranges
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/slurm/extract_ranges_%j.log
#SBATCH --error=logs/slurm/extract_ranges_%j.err

# =============================================================================
# Phosphogypsum Bot: Fast Parameter Extraction & Range Synthesis (9a14a CPU)
# Hardware: 192 AMD EPYC CPU Cores with NUMA Socket 0 (80 Threads)
#
# Key Features:
#   1. Sets --parallel 1 to allocate FULL 65,536 tokens to single slot
#      (fixes 16k context overflow on 47 large review papers)
#   2. Cleans up 2-byte empty JSON files so only missing papers are extracted
#   3. Automatically preserves already-extracted valid JSONs (instant skip)
#   4. Re-synthesizes global parameter ranges (Step 4) & unified KG (Step 5)
#   5. Expected runtime: ~4-6 hours (much faster than full 18-day LightRAG pipeline)
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
echo "  Phosphogypsum Parameter Extraction & Range Synthesis"
echo "  Job ID:       ${SLURM_JOB_ID:-manual}"
echo "  Compute Node: $(hostname)"
echo "  Partition:    9a14a (192 Cores, NUMA Socket 0)"
echo "  Project Root: $PROJECT_ROOT"
echo "  Start Time:   $(date)"
echo "============================================================"

# Ensure directories exist
mkdir -p logs/slurm logs/services \
         datahub/processed/extracted_data \
         datahub/processed/parameter_ranges \
         datahub/processed/knowledge_graph

# 3. Model Server Binaries & Model Weights
if [ -f "/project/tangsiqi/software/llama.cpp/build_cpu/bin/llama-server" ]; then
    LLAMA_SERVER_BIN="/project/tangsiqi/software/llama.cpp/build_cpu/bin/llama-server"
elif [ -f "/home/tangsiqi/project/software/llama.cpp/build_cpu/bin/llama-server" ]; then
    LLAMA_SERVER_BIN="/home/tangsiqi/project/software/llama.cpp/build_cpu/bin/llama-server"
else
    LLAMA_SERVER_BIN="/project/tangsiqi/software/llama.cpp/build_cpu/bin/llama-server"
fi
REASONER_MODEL="/home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
REASONER_MMPROJ="/home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf"

if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    echo "[FATAL] llama-server CPU binary not found at: $LLAMA_SERVER_BIN"
    exit 1
fi

PID_REASONER=""

# 4. Lifecycle Management: Clean Signal Propagation & Shutdown Trap
cleanup() {
    echo ""
    echo "============================================================"
    echo " [TEARDOWN] Stopping background llama-server instance..."
    echo "============================================================"
    if [ -n "$PID_REASONER" ] && kill -0 "$PID_REASONER" 2>/dev/null; then
        echo "  - Terminating Reasoner Server (PID: $PID_REASONER)..."
        kill "$PID_REASONER" 2>/dev/null
    fi
    wait "$PID_REASONER" 2>/dev/null
    echo " [TEARDOWN] Background services stopped cleanly."
}
trap cleanup EXIT SIGINT SIGTERM

# 5. Launch Reasoner LLM with FULL 65,536 Context (--parallel 1)
echo ""
echo "[1/3] Launching Qwen3.6-35B Reasoner with Full 65k Context..."
echo "  - Socket 0 (Port 11434, 80 threads, --ctx-size 81920, --parallel 1)..."

numactl --cpunodebind=0 --membind=0 \
  "$LLAMA_SERVER_BIN" \
  --model "$REASONER_MODEL" \
  --mmproj "$REASONER_MMPROJ" \
  --host 127.0.0.1 \
  --port 11434 \
  --threads 80 \
  --ctx-size 81920 \
  --parallel 1 \
  --n-gpu-layers 0 > "logs/services/reasoner_server_${SLURM_JOB_ID}.log" 2>&1 &
PID_REASONER=$!

# Pre-flight Readiness Check
echo "  - Waiting for Reasoner server readiness..."
TIMEOUT=600
INTERVAL=5
ELAPSED=0

while true; do
    if ! kill -0 "$PID_REASONER" 2>/dev/null; then
        echo "[FATAL] Reasoner server process (PID $PID_REASONER) died unexpectedly!"
        tail -n 30 "logs/services/reasoner_server_${SLURM_JOB_ID}.log"
        exit 1
    fi

    if curl -s -f http://127.0.0.1:11434/health > /dev/null 2>&1; then
        echo "  [OK] Reasoner server (11434) active with full 65k context! (Elapsed: ${ELAPSED}s)"
        break
    fi

    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "[TIMEOUT] Reasoner server failed to reach healthy state within ${TIMEOUT}s."
        exit 1
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

# Configure Environment for Knowledge Graph Engine
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_TIMEOUT=3600
export LLM_CONTEXT_LENGTH=81920

# 6. Clean up 2-byte empty JSON files so they are re-extracted
echo ""
echo "[2/3] Cleaning up 2-byte empty extraction JSONs..."
EMPTY_COUNT=$(find datahub/processed/extracted_data -name "*_extracted.json" -size -10c 2>/dev/null | wc -l)
echo "  - Found $EMPTY_COUNT empty/placeholder files to re-extract."
if [ "$EMPTY_COUNT" -gt 0 ]; then
    find datahub/processed/extracted_data -name "*_extracted.json" -size -10c -delete
    echo "  - Cleaned up $EMPTY_COUNT files. Existing valid files are preserved."
fi

# 7. Execute Parameter Extraction, Range Synthesis & KG Assembly
echo ""
echo "[3/3] Executing Parameter Extraction, Range Synthesis & Assembly..."

# Step 3: Extract structured parameters JSON via Reasoner LLM
echo ""
echo ">>> [Phase 1/3] Step 3: Structured Engineering Parameter Extraction <<<"
python scripts/build_knowledge_graph.py \
  --step extract \
  --engine lightrag

EXTRACT_STATUS=$?
if [ $EXTRACT_STATUS -ne 0 ]; then
    echo "[WARN] Step 3 (Extract) finished with exit code: $EXTRACT_STATUS"
fi

# Step 4: Synthesize global parameter distributions for MCMC / Reverse Design
echo ""
echo ">>> [Phase 2/3] Step 4: Parameter Range & Uncertainty Synthesis <<<"
python scripts/build_knowledge_graph.py \
  --step ranges

RANGES_STATUS=$?

# Step 5: Assemble unified queryable Phosphogypsum Knowledge Graph
echo ""
echo ">>> [Phase 3/3] Step 5: Knowledge Graph Assembly <<<"
python scripts/build_knowledge_graph.py \
  --step build

BUILD_STATUS=$?

# Summary
echo ""
echo "============================================================"
echo "  Execution Summary"
echo "  - Data Extract (Step 3):     Exit $EXTRACT_STATUS"
echo "  - Parameter Ranges (Step 4):  Exit $RANGES_STATUS"
echo "  - Knowledge Graph (Step 5):   Exit $BUILD_STATUS"
echo "============================================================"

VALID_COUNT=$(find datahub/processed/extracted_data -name "*_extracted.json" -size +10c 2>/dev/null | wc -l)
TOTAL_JSON=$(find datahub/processed/extracted_data -name "*_extracted.json" 2>/dev/null | wc -l)
echo "  Output Metrics on Disk:"
echo "    * Total Extracted JSONs:    $TOTAL_JSON / 72"
echo "    * Non-empty Valid Sets:     $VALID_COUNT / 72"
echo "  Finished at: $(date)"
echo "============================================================"

exit 0