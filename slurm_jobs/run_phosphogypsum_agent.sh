#!/bin/bash
#SBATCH --job-name=PhosphogypsumBot_Agent
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=pgbot_agent_%j.log
#SBATCH --error=pgbot_agent_%j.err

#=============================================================================#
# [Submission Guidelines on WHU-SCC Cluster]
# 1. A100 GPU (Recommended): sbatch -p a100x4 --gres=gpu:1 --cpus-per-task=16 slurm_jobs/run_phosphogypsum_agent.sh
# 2. V100 GPU:               sbatch -p gpu --gres=gpu:2 --cpus-per-task=10 slurm_jobs/run_phosphogypsum_agent.sh
# 3. 9a14a CPU (192 Cores):  sbatch -p 9a14a --nodes=1 --cpus-per-task=192 slurm_jobs/run_phosphogypsum_agent.sh
# 4. 9a14a CPU (64 Cores):   sbatch -p 9a14a --nodes=1 --cpus-per-task=64 slurm_jobs/run_phosphogypsum_agent.sh
#=============================================================================#

echo "======================================================================="
echo "Starting Slurm Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Node assigned:      $SLURM_JOB_NODELIST"
echo "Submission dir:     $SLURM_SUBMIT_DIR"
echo "Start time:         $(date)"
echo "======================================================================="

# Establish robust workspace anchoring
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    case "$SLURM_SUBMIT_DIR" in
        */slurm_jobs) PROJECT_ROOT="$( dirname "$SLURM_SUBMIT_DIR" )" ;;
        *)            PROJECT_ROOT="$SLURM_SUBMIT_DIR" ;;
    esac
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    case "$SCRIPT_DIR" in
        */slurm_jobs) PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )" ;;
        *)            PROJECT_ROOT="$SCRIPT_DIR" ;;
    esac
fi
cd "$PROJECT_ROOT"
echo "Active workspace root: $(pwd)"

# -----------------------------------------------------------------------------
# 1. Hugging Face Global Cache Redirection
# -----------------------------------------------------------------------------
export HF_HOME="/scratch/$USER/huggingface_cache"
mkdir -p "$HF_HOME"
echo "Redirected HF_HOME caching registry to: $HF_HOME"

# -----------------------------------------------------------------------------
# 2. Conda Environment Activation
# -----------------------------------------------------------------------------
echo "Initializing Anaconda..."
if [ -f "$HOME/project/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/project/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/project/miniconda3/bin:$HOME/anaconda3/bin:$HOME/miniconda3/bin:$PATH"
    source conda activate 2>/dev/null
fi

echo "Activating virtual environment..."
conda activate pgbot 2>/dev/null || conda activate pyrolysis_model_dnn 2>/dev/null || conda activate base

# -----------------------------------------------------------------------------
# 3. Model Weight & Backend Path Resolution
# -----------------------------------------------------------------------------
PORT=11434
HOST="127.0.0.1"
export LLM_BASE_URL="http://${HOST}:${PORT}/v1"
export LLM_API_KEY="sk-no-key-required"
export LLM_MODEL="Qwen/Qwen3.6-35B-A3B-Instruct"

LLAMA_DIR="/home/$USER/project/software/llama.cpp"

# Locate pre-cached model weights in /scratch
if [ -f "/scratch/$USER/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf" ]; then
    GGUF_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    MMPROJ_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf"
elif [ -f "/scratch/$USER/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf" ]; then
    GGUF_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf"
    MMPROJ_MODEL=""
else
    GGUF_MODEL="/home/$USER/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
    MMPROJ_MODEL="/home/$USER/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf"
fi

echo "Using Model Weight: $GGUF_MODEL"

# Process cleanup trap
SERVER_PID=0
trap 'if [ "$SERVER_PID" -gt 0 ]; then echo "[Trap] Terminating llama-server (PID: $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; fi' EXIT INT TERM

# Load compilation modules
module load scl/gcc13 2>/dev/null || true

# Smart hardware routing
if [ "$SLURM_JOB_PARTITION" == "a100x4" ]; then
    echo "[Info] A100 GPU partition detected..."
    module load nvidia/cuda/12.9 2>/dev/null || module load cuda/12.1 2>/dev/null || true
    LLAMA_BIN="${LLAMA_DIR}/build_a100/bin/llama-server"
    SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "32768" "--n-gpu-layers" "99" "-sm" "row" "-fa" "on")
    if [ -n "$MMPROJ_MODEL" ] && [ -f "$MMPROJ_MODEL" ]; then
        SERVER_CMD+=("--mmproj" "$MMPROJ_MODEL")
    fi
elif [ "$SLURM_JOB_PARTITION" == "gpu" ]; then
    echo "[Info] V100 GPU partition detected..."
    module load nvidia/cuda/12.9 2>/dev/null || module load cuda/12.1 2>/dev/null || true
    LLAMA_BIN="${LLAMA_DIR}/build_v100/bin/llama-server"
    SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "16384" "--n-gpu-layers" "99" "-sm" "row" "-fa" "on")
elif [ "$SLURM_JOB_PARTITION" == "9a14a" ]; then
    echo "[Info] Pure CPU partition (AMD EPYC with NUMA interleaving) detected..."
    LLAMA_BIN="${LLAMA_DIR}/build_cpu/bin/llama-server"
    NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
    SERVER_CMD=("numactl" "--interleave=all" "$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "16384" "--threads" "$NUM_THREADS" "--n-gpu-layers" "0")
else
    echo "[Info] Standard partition fallback..."
    LLAMA_BIN="${LLAMA_DIR}/build_cpu/bin/llama-server"
    SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "8192" "--n-gpu-layers" "0")
fi

echo "Deploying backend server..."
mkdir -p logs
"${SERVER_CMD[@]}" > logs/llama_server_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

echo "Waiting for local model server to initialize and load weights (PID: $SERVER_PID)..."
MAX_RETRIES=60
RETRY_COUNT=0
SERVER_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://${HOST}:${PORT}/health" > /dev/null 2>&1 || curl -s "http://${HOST}:${PORT}/v1/models" > /dev/null 2>&1; then
        SERVER_READY=true
        break
    fi
    sleep 3
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
done
echo ""

if [ "$SERVER_READY" = false ]; then
    echo "[Error] Server failed to start within $((MAX_RETRIES * 3)) seconds."
    cat logs/llama_server_${SLURM_JOB_ID}.log | tail -n 20
    exit 1
fi

echo "======================================================================="
echo "Local Qwen Model Server is ONLINE and READY!"
echo "Server Endpoint: $LLM_BASE_URL"
echo "======================================================================="

# -----------------------------------------------------------------------------
# 4. Run Phosphogypsum Agent Orchestrator
# -----------------------------------------------------------------------------
echo "Launching PhosphogypsumBot Agent..."

python - << 'EOF'
import os
from chat_agent.agent import PhosphogypsumAgent

agent = PhosphogypsumAgent(
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    model=os.getenv("LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-Instruct")
)

queries = [
    "请列出当前支持的所有磷石膏综合利用技术路径代码与名称。",
    "针对 PG-CementProd（磷石膏制水泥）和 PG-REEextract（磷石膏提取稀土）路径，分别计算 1 吨磷石膏处理下的 LCA 环境影响和 TEA 经济指标，并给出对比分析。",
    "在以渣定产政策下，综合考虑 TRL 熟化度、碳减排与经济回报，对所有路径进行多准则优选打分，推荐最具实施价值的方案。"
]

for i, q in enumerate(queries, 1):
    print("\n" + "="*80)
    print(f"[Query {i}] 👨‍🔬 研究员提问: {q}")
    print("="*80)
    response = agent.chat(q)
    print(f"\n[PhosphogypsumBot 回复]:\n{response}\n")

EOF

echo "======================================================================="
echo "All autonomous queries completed successfully! (Job ID: $SLURM_JOB_ID)"
echo "Finished at: $(date)"
echo "======================================================================="
