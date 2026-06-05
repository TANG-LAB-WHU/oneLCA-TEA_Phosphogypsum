#!/bin/bash
#SBATCH --job-name=kg_test_a100
#SBATCH --partition=a100x4
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2              # Use 2 A100s (40GB each)
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00           # 2 hours should be enough for 1 paper
#SBATCH --output=logs/test_kg_pipeline_a100/kg_test_a100_%j.log

# Load CUDA environment
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/12.2 2>/dev/null || module load cuda/12.0 2>/dev/null || module load cuda 2>/dev/null

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

# Set shared tiktoken cache directory for compute nodes (no internet access)
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"

# Fallback: search for libcudart.so.12 in conda and system paths if not loaded by module
for path in \
    "$CONDA_PREFIX/lib" \
    "$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib" \
    "$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib" \
    "/usr/local/cuda/lib64" \
    "/usr/local/cuda-12.1/lib64" \
    "/usr/local/cuda-12.2/lib64" \
    "/usr/local/cuda-12.0/lib64"; do
    if [ -f "$path/libcudart.so.12" ]; then
        export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
        break
    fi
done

# Change to project root so relative paths in Python scripts work correctly
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

echo "============================================================"
echo "  Phosphogypsum Knowledge Graph Pipeline Quick Test (1 Paper)"
echo "  Node: $(hostname)  |  Partition: A100"
echo "  Working Dir: $(pwd)"
echo "============================================================"

# =============================================================================
# PHASE 1: PDF Parsing with MinerU (GPU-accelerated)
# =============================================================================
# MinerU needs ~6GB VRAM. A100 has 40GB, so GPU 0 is more than enough.
# MinerU auto-exits after parsing, releasing all VRAM.
# =============================================================================

echo ""
echo ">>> PHASE 1: PDF Parsing with MinerU (GPU 0) <<<"
echo ""

export CUDA_VISIBLE_DEVICES=0
export MINERU_MODEL_SOURCE=local

python scripts/build_knowledge_graph.py \
  --step parse \
  --parser mineru \
  --limit 1

PARSE_EXIT_CODE=$?

if [ $PARSE_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] MinerU PDF parsing failed with exit code $PARSE_EXIT_CODE"
    exit $PARSE_EXIT_CODE
fi

echo ""
echo ">>> PHASE 1 complete. MinerU exited, GPU 0 VRAM released. <<<"
echo ""

unset CUDA_VISIBLE_DEVICES

# =============================================================================
# PHASE 2: Knowledge Graph Construction (LLM + Embedding servers)
# =============================================================================
# A100 has plenty of VRAM (40GB per card).
# GPU 0: Reasoner LLM (27GB) + Embedding model (7.5GB) = ~34.5GB (fits in 40GB)
# GPU 1: Available for future VLM tasks
# =============================================================================

echo ">>> PHASE 2: Knowledge Graph Construction (LLM servers) <<<"
echo ""

# 1. Start Reasoner LLM (Qwen3.6-27B) on GPU 0
export CUDA_VISIBLE_DEVICES=0
/project/tangsiqi/software/llama.cpp/build_a100/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --mmproj /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf \
  --host 127.0.0.1 \
  --port 11434 \
  --ctx-size 65536 \
  --parallel 4 \
  --n-gpu-layers 99 > slurm_jobs/logs/test_kg_pipeline_a100/reasoner_server_${SLURM_JOB_ID}.log 2>&1 &
PID_REASONER=$!

# 2. Start Embedding Model (Qwen3-Embedding-8B-Q8_0, full precision) on GPU 1
export CUDA_VISIBLE_DEVICES=1
/project/tangsiqi/software/llama.cpp/build_a100/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3-Embedding-8B-Q8_0.gguf \
  --embedding \
  --host 127.0.0.1 \
  --port 11436 \
  --ctx-size 32768 \
  --parallel 8 \
  --n-gpu-layers 99 > slurm_jobs/logs/test_kg_pipeline_a100/embed_server_${SLURM_JOB_ID}.log 2>&1 &
PID_EMBED=$!

echo "Waiting for llama-servers to initialize (polling every 5 seconds, max 10 minutes)..."
TIMEOUT=600
INTERVAL=5
ELAPSED=0

while true; do
    REASONER_UP=0
    EMBED_UP=0
    
    if curl -s -f http://127.0.0.1:11434/health > /dev/null; then
        REASONER_UP=1
    fi
    
    if curl -s -f http://127.0.0.1:11436/health > /dev/null; then
        EMBED_UP=1
    fi
    
    if [ $REASONER_UP -eq 1 ] && [ $EMBED_UP -eq 1 ]; then
        echo "Both llama-servers are up and healthy after ${ELAPSED} seconds!"
        break
    fi
    
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "[ERROR] llama-servers failed to initialize within ${TIMEOUT} seconds."
        echo "===== Reasoner Server Log (last 20 lines) ====="
        tail -n 20 slurm_jobs/logs/test_kg_pipeline_a100/reasoner_server_${SLURM_JOB_ID}.log
        echo "===== Embedding Server Log (last 20 lines) ====="
        tail -n 20 slurm_jobs/logs/test_kg_pipeline_a100/embed_server_${SLURM_JOB_ID}.log
        kill $PID_REASONER 2>/dev/null
        kill $PID_EMBED 2>/dev/null
        exit 1
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "Still waiting... (${ELAPSED}/${TIMEOUT}s) - Reasoner ready: ${REASONER_UP}, Embed ready: ${EMBED_UP}"
done

echo "Both llama-servers started successfully."

# Configure environment for Python pipeline
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export EMBEDDING_BASE_URL="http://127.0.0.1:11436/v1"
export LLM_TIMEOUT=1800
export EMBEDDING_TIMEOUT=600

# Run remaining pipeline steps: index -> extract
python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag \
  --limit 1

INDEX_EXIT_CODE=$?

if [ $INDEX_EXIT_CODE -ne 0 ]; then
    echo "[WARN] LightRAG index step failed with exit code $INDEX_EXIT_CODE"
fi

python scripts/build_knowledge_graph.py \
  --step extract \
  --engine lightrag \
  --limit 1

EXTRACT_EXIT_CODE=$?

if [ $EXTRACT_EXIT_CODE -ne 0 ]; then
    echo "[WARN] Extraction step failed with exit code $EXTRACT_EXIT_CODE"
fi

# =============================================================================
# PHASE 3: Cleanup
# =============================================================================
echo ""
echo ">>> Shutting down llama-server instances... <<<"
kill $PID_REASONER 2>/dev/null
kill $PID_EMBED 2>/dev/null
wait $PID_REASONER 2>/dev/null
wait $PID_EMBED 2>/dev/null

# Determine overall exit code
FINAL_EXIT_CODE=0
if [ $PARSE_EXIT_CODE -ne 0 ]; then FINAL_EXIT_CODE=1; fi
if [ $INDEX_EXIT_CODE -ne 0 ]; then FINAL_EXIT_CODE=1; fi
if [ $EXTRACT_EXIT_CODE -ne 0 ]; then FINAL_EXIT_CODE=1; fi

echo ""
echo "============================================================"
echo "  Pipeline Summary"
echo "  Parse (MinerU):        exit $PARSE_EXIT_CODE"
echo "  Index (LightRAG):      exit $INDEX_EXIT_CODE"
echo "  Extract (LLM):         exit $EXTRACT_EXIT_CODE"
echo "  Overall:               exit $FINAL_EXIT_CODE"
echo "============================================================"

exit $FINAL_EXIT_CODE
