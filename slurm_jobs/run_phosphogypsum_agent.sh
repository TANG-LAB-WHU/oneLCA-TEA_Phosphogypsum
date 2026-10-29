#!/bin/bash
#SBATCH --job-name=PhosphogypsumBot_Agent
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/pgbot_agent_%j.log
#SBATCH --error=logs/slurm/pgbot_agent_%j.err

#=============================================================================#
# [Submission Guidelines on WHU-SCC Cluster]
# (Note: Always submit jobs from the repository root: sbatch slurm_jobs/run_phosphogypsum_agent.sh)
#
# 1. Hardware Default Routing (Auto-Selection / Default: Qwen3.8-Flash-Next):
#    - Full VRAM Offload (Recommended):  sbatch -p a100x4 --gres=gpu:4 --cpus-per-task=32 slurm_jobs/run_phosphogypsum_agent.sh --flash
#      * 4x 40GB A100 = 160GB VRAM (Fits 100% of ~110GB weights, ngl 99 via NVLink tensor parallel -sm row)
#    - Single A100 GPU (Hybrid Offload): sbatch -p a100x4 --gres=gpu:1 --cpus-per-task=16 slurm_jobs/run_phosphogypsum_agent.sh
#      * 1x 40GB A100 = 40GB VRAM (32 layers offloaded to GPU to prevent CUDA OOM, remaining on CPU RAM)
#    - V100 GPU (Hybrid Offload):        sbatch -p gpu --gres=gpu:2 --cpus-per-task=10 slurm_jobs/run_phosphogypsum_agent.sh
#    - 9a14a 192-Core CPU (Pure NUMA):   sbatch -p 9a14a --nodes=1 --cpus-per-task=192 slurm_jobs/run_phosphogypsum_agent.sh --flash
#
# 2. Explicit Model Selection & Cross-Partition Usage:
#    - Force Qwen3.8-Flash-Next (Default): sbatch -p a100x4 --gres=gpu:4 slurm_jobs/run_phosphogypsum_agent.sh --flash
#    - Force Qwen3.8-27B-Instruct:         sbatch -p a100x4 --gres=gpu:1 slurm_jobs/run_phosphogypsum_agent.sh --27b
#
# 3. Custom Query Execution:
#    - Run batch optimization:             sbatch -p a100x4 --gres=gpu:4 slurm_jobs/run_phosphogypsum_agent.sh "评估磷石膏制硫酸联产水泥生产线的碳减排潜力和经济净现值"
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

# =============================================================================
# 0. Dynamic Parameter Parsing & Model Target Resolution (Default: flash)
# =============================================================================
TARGET_MODEL="${PGAGENT_MODEL:-flash}"
EXEC_MODE="batch"
CUSTOM_QUERY=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            if [[ $# -ge 2 && ! "$2" =~ ^-- ]]; then
                TARGET_MODEL="$2"; shift 2
            else
                echo "[Error] Flag -m/--model requires a value (e.g., flash, 27b)."; exit 1
            fi
            ;;
        --model=*)
            TARGET_MODEL="${1#*=}"; shift
            ;;
        --flash|--next|--flash-next)
            TARGET_MODEL="flash"; shift
            ;;
        --27b|--qwen27b)
            TARGET_MODEL="27b"; shift
            ;;
        --chatbot|--cli)
            EXEC_MODE="chatbot"; shift
            ;;
        --query)
            if [[ $# -ge 2 ]]; then
                CUSTOM_QUERY="$2"; shift 2
            else
                shift
            fi
            ;;
        -h|--help)
            echo "Usage: sbatch [sbatch_options] slurm_jobs/run_phosphogypsum_agent.sh [options] [query]"
            echo "Options:"
            echo "  -m, --model <flash|27b|auto>  Select Qwen model architecture (default: flash)"
            echo "  --flash, --next               Shortcut for --model flash (Qwen3.8-Flash-Next 125B MoE)"
            echo "  --27b                         Shortcut for --model 27b (Qwen3.8-27B-Instruct Dense)"
            echo "  --chatbot                     Launch interactive conversational shell"
            echo "  --query <prompt>              Run specific natural language domain query"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1"); shift
            ;;
    esac
done

if [ -z "$CUSTOM_QUERY" ] && [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    CUSTOM_QUERY="${POSITIONAL_ARGS[*]}"
fi

# Normalize model identifier
TARGET_MODEL=$(echo "$TARGET_MODEL" | tr '[:upper:]' '[:lower:]')

# -----------------------------------------------------------------------------
# 1. Hugging Face Global Cache Redirection
# -----------------------------------------------------------------------------
export HF_HOME="/scratch/$USER/huggingface_cache"
mkdir -p "$HF_HOME"
echo "Redirected HF_HOME caching registry to: $HF_HOME"

# Set offline tiktoken cache directory for compute nodes (no internet access)
export TIKTOKEN_CACHE_DIR="/home/$USER/.cache/tiktoken"
if [ ! -d "$TIKTOKEN_CACHE_DIR" ] && [ -d "/home/tangsiqi/.cache/tiktoken" ]; then
    export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"
fi
echo "Configured offline TIKTOKEN_CACHE_DIR to: $TIKTOKEN_CACHE_DIR"

# Enforce UTF-8 standard encoding for Python console & log redirection
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

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
# 3. Model Weight & Backend Path Resolution (Qwen3.8-Flash-Next Default)
# -----------------------------------------------------------------------------
PORT=11434
HOST="127.0.0.1"
if [ -d "/project/$USER/software/llama.cpp" ]; then
    LLAMA_DIR="/project/$USER/software/llama.cpp"
elif [ -d "/home/$USER/project/software/llama.cpp" ]; then
    LLAMA_DIR="/home/$USER/project/software/llama.cpp"
else
    LLAMA_DIR="/project/$USER/software/llama.cpp"
fi

case "$TARGET_MODEL" in
    flash|next|flash-next|moe|auto|"")
        CHOSEN_MODEL="flash"
        ;;
    27b|qwen27b|dense)
        CHOSEN_MODEL="27b"
        ;;
    *)
        echo "[Warning] Unrecognized model target '${TARGET_MODEL}'. Falling back to flash."
        CHOSEN_MODEL="flash"
        ;;
esac

# Locate pre-cached model weights in /scratch
if [ "$CHOSEN_MODEL" == "flash" ]; then
    MODEL_ID="Qwen/Qwen3.8-Flash-Next"
    FLASH_DIR_1="/scratch/$USER/ai_models/qwen/Qwen3.8-Flash-Next-Q8_0"
    FLASH_DIR_2="/home/$USER/scratch/ai_models/qwen/Qwen3.8-Flash-Next-Q8_0"

    if [ -f "${FLASH_DIR_1}/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf" ]; then
        GGUF_MODEL="${FLASH_DIR_1}/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
        MMPROJ_MODEL="${FLASH_DIR_1}/mmproj-BF16.gguf"
    elif [ -f "${FLASH_DIR_2}/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf" ]; then
        GGUF_MODEL="${FLASH_DIR_2}/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf"
        MMPROJ_MODEL="${FLASH_DIR_2}/mmproj-BF16.gguf"
    else
        echo "[Warning] Flash-Next weights missing on scratch. Gracefully falling back to 27B / 35B..."
        CHOSEN_MODEL="27b"
    fi
fi

if [ "$CHOSEN_MODEL" == "27b" ]; then
    MODEL_ID="Qwen/Qwen3.8-27B-Instruct"
    if [ -f "/scratch/$USER/ai_models/qwen/Qwen3.8-27B-Unsloth/Qwen3.8-27B-UD-Q8_K_XL.gguf" ]; then
        GGUF_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.8-27B-Unsloth/Qwen3.8-27B-UD-Q8_K_XL.gguf"
        MMPROJ_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.8-27B-Unsloth/mmproj-BF16.gguf"
    elif [ -f "/scratch/$USER/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf" ]; then
        MODEL_ID="Qwen/Qwen3.6-35B-A3B-Instruct"
        GGUF_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
        MMPROJ_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf"
    elif [ -f "/scratch/$USER/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf" ]; then
        MODEL_ID="Qwen/Qwen3.6-27B-Instruct"
        GGUF_MODEL="/scratch/$USER/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf"
        MMPROJ_MODEL=""
    else
        MODEL_ID="Qwen/Qwen3.6-35B-A3B-Instruct"
        GGUF_MODEL="/home/$USER/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
        MMPROJ_MODEL="/home/$USER/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf"
    fi
fi

export LLM_BASE_URL="http://${HOST}:${PORT}/v1"
export LLM_API_KEY="sk-no-key-required"
export LLM_MODEL="$MODEL_ID"

echo "[Model Selection] Active Model ID: $LLM_MODEL"
echo "[Model Selection] GGUF Weights:    $GGUF_MODEL"

# Process cleanup trap
SERVER_PID=0
trap 'if [ "$SERVER_PID" -gt 0 ]; then echo "[Trap] Terminating llama-server (PID: $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; fi' EXIT INT TERM

# Load compilation modules
module load scl/gcc13 2>/dev/null || true

# Smart hardware routing logic with model-aware VRAM protection
if [ "$SLURM_JOB_PARTITION" == "a100x4" ]; then
    echo "[Info] A100 GPU partition detected..."
    module load nvidia/cuda/12.9 2>/dev/null || module load cuda/12.1 2>/dev/null || true
    LLAMA_BIN="${LLAMA_DIR}/build_a100/bin/llama-server"

    if [ "$CHOSEN_MODEL" == "flash" ]; then
        # WHU-SCC a100x4 partition: Each NVIDIA A100 has 40GB VRAM.
        # Qwen3.8-Flash-Next total sharded weights: ~110GB.
        # Dynamic VRAM scaling:
        #   - >= 4 GPUs (160GB VRAM): 100% full VRAM offload (ngl 99) with NVLink row-split parallelism (-sm row).
        #   - 2-3 GPUs (80-120GB VRAM): Hybrid offload (ngl 64).
        #   - 1 GPU (40GB VRAM): Hybrid offload (ngl 32) to prevent CUDA OOM.
        GPU_COUNT=1
        if [ -n "$SLURM_GPUS_ON_NODE" ]; then
            GPU_COUNT="$SLURM_GPUS_ON_NODE"
        elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v "^$" | wc -l)
        elif command -v nvidia-smi &>/dev/null; then
            GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
        fi
        [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 1 ] && GPU_COUNT=1

        if [ "$GPU_COUNT" -ge 4 ]; then
            echo "[Info] Flash-Next on $GPU_COUNT x A100 (160GB VRAM total). Offloading all layers to VRAM (ngl 99, -sm row)."
            GPU_ARGS=("--n-gpu-layers" "99" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
        elif [ "$GPU_COUNT" -ge 2 ]; then
            echo "[Info] Flash-Next on $GPU_COUNT x A100 (80GB VRAM total). Offloading 64 layers to VRAM (ngl 64, -sm row)."
            GPU_ARGS=("--n-gpu-layers" "64" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
        else
            echo "[Info] Flash-Next on 1x A100 (40GB VRAM). Offloading 32 layers to VRAM to prevent OOM (hybrid CPU-GPU)."
            GPU_ARGS=("--n-gpu-layers" "32" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
        fi
    else
        GPU_ARGS=("--n-gpu-layers" "99" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
    fi
    SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "16384" "${GPU_ARGS[@]}")

elif [ "$SLURM_JOB_PARTITION" == "gpu" ]; then
    echo "[Info] V100 GPU partition detected..."
    module load nvidia/cuda/12.9 2>/dev/null || module load cuda/12.1 2>/dev/null || true
    LLAMA_BIN="${LLAMA_DIR}/build_v100/bin/llama-server"

    if [ "$CHOSEN_MODEL" == "flash" ]; then
        GPU_COUNT=1
        if [ -n "$SLURM_GPUS_ON_NODE" ]; then
            GPU_COUNT="$SLURM_GPUS_ON_NODE"
        elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
            GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v "^$" | wc -l)
        fi
        [ -z "$GPU_COUNT" ] || [ "$GPU_COUNT" -lt 1 ] && GPU_COUNT=1

        if [ "$GPU_COUNT" -ge 4 ]; then
            echo "[Info] Flash-Next on $GPU_COUNT x V100. Allocating 64 layers to VRAM (hybrid offloading)."
            GPU_ARGS=("--n-gpu-layers" "64" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
        else
            echo "[Info] Flash-Next (125B MoE) on Volta GPU ($GPU_COUNT card(s)). Allocating 24 layers to VRAM to prevent OOM."
            GPU_ARGS=("--n-gpu-layers" "24" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
        fi
    else
        GPU_ARGS=("--n-gpu-layers" "99" "-sm" "row" "-fa" "on" "--ctk" "q8_0" "--ctv" "q8_0")
    fi
    SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "16384" "${GPU_ARGS[@]}")

elif [ "$SLURM_JOB_PARTITION" == "9a14a" ]; then
    echo "[Info] Pure CPU partition (AMD EPYC with NUMA interleaving) detected..."
    LLAMA_BIN="${LLAMA_DIR}/build_cpu/bin/llama-server"
    
    # Optimize CPU threads for AMD EPYC NUMA architecture:
    # 96+ threads cross dual sockets causes severe OpenMP barrier latency.
    # 48 threads for token generation and 64 threads for batch prompt prefill achieves optimal throughput.
    NUM_THREADS=48
    NUM_BATCH_THREADS=64
    if [ -n "$SLURM_CPUS_PER_TASK" ] && [ "$SLURM_CPUS_PER_TASK" -lt 48 ]; then
        NUM_THREADS="$SLURM_CPUS_PER_TASK"
        NUM_BATCH_THREADS="$SLURM_CPUS_PER_TASK"
    fi
    echo "[Info] Configured CPU threads: ${NUM_THREADS} (generation), ${NUM_BATCH_THREADS} (batch prompt prefill)."

    # Pin OpenMP threads cleanly to avoid thrashing across CCX
    export OMP_NUM_THREADS="$NUM_THREADS"
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores

    # Context window set to 16384 to support complex multi-hop tool-calling sequences (e.g. 7+ tool calls producing ~8.6k tokens)
    if command -v numactl &>/dev/null; then
        SERVER_CMD=("numactl" "--interleave=all" "$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "16384" "-b" "512" "-ub" "256" "--threads" "$NUM_THREADS" "--threads-batch" "$NUM_BATCH_THREADS" "--n-gpu-layers" "0" "--numa" "distribute")
    else
        SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "16384" "-b" "512" "-ub" "256" "--threads" "$NUM_THREADS" "--threads-batch" "$NUM_BATCH_THREADS" "--n-gpu-layers" "0")
    fi
else
    echo "[Info] Standard partition fallback..."
    LLAMA_BIN="${LLAMA_DIR}/build_cpu/bin/llama-server"
    SERVER_CMD=("$LLAMA_BIN" "-m" "$GGUF_MODEL" "--host" "$HOST" "--port" "$PORT" "-c" "8192" "--n-gpu-layers" "0")
fi

if [ -n "$MMPROJ_MODEL" ] && [ -f "$MMPROJ_MODEL" ]; then
    echo "[Info] Multimodal vision projector detected: $MMPROJ_MODEL"
    SERVER_CMD+=("--mmproj" "$MMPROJ_MODEL")
fi

echo "Deploying backend server..."
mkdir -p logs/services logs/slurm reports
"${SERVER_CMD[@]}" > logs/services/llama_server_${SLURM_JOB_ID:-local}.log 2>&1 &
SERVER_PID=$!

echo "Waiting for local model server to initialize and load weights (PID: $SERVER_PID)..."
MAX_RETRIES=360 # 1800 seconds max timeout for large sharded weights loading
RETRY_COUNT=0
SERVER_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "[Error] Server process died unexpectedly. Inspecting logs/services/llama_server_${SLURM_JOB_ID:-local}.log:"
        tail -n 35 "logs/services/llama_server_${SLURM_JOB_ID:-local}.log"
        exit 1
    fi

    # Strictly check for HTTP 200 (HTTP 503 means weights are still loading into memory)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        SERVER_READY=true
        echo " [Ready! Server responded HTTP 200]"
        break
    fi

    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    if [ $((RETRY_COUNT % 12)) -eq 0 ]; then
        echo " [Loading weights: $((RETRY_COUNT * 5))s, HTTP status: ${HTTP_CODE:-pending}]"
    fi
done
echo ""

if [ "$SERVER_READY" = false ]; then
    echo "[Error] Server failed to start within $((MAX_RETRIES * 5)) seconds."
    cat logs/services/llama_server_${SLURM_JOB_ID:-local}.log | tail -n 25
    exit 1
fi

echo "======================================================================="
echo "Local Qwen Model Server ($LLM_MODEL) is ONLINE and READY!"
echo "Server Endpoint: $LLM_BASE_URL"
echo "======================================================================="

# -----------------------------------------------------------------------------
# 4. Run Phosphogypsum Agent Orchestrator
# -----------------------------------------------------------------------------
echo "Launching PhosphogypsumBot Agent..."

# Set extended timeout (60 mins) for CPU inference on large 125B MoE models
export LLM_TIMEOUT=3600
export LLM_MAX_TOKENS=1500

if [ "$EXEC_MODE" == "chatbot" ]; then
    echo "[Interactive Mode] Launching Chatbot CLI..."
    python -m chat_agent.cli
elif [ -n "$CUSTOM_QUERY" ]; then
    echo "[Custom Query Mode] Running query: $CUSTOM_QUERY"
    python - << EOF
import os
from chat_agent.agent import PhosphogypsumAgent

agent = PhosphogypsumAgent(
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    model=os.getenv("LLM_MODEL", "Qwen/Qwen3.8-Flash-Next"),
    timeout=float(os.getenv("LLM_TIMEOUT", "3600.0"))
)

q = """$CUSTOM_QUERY"""
print("\n" + "="*80)
print(f"👨‍🔬 研究员提问: {q}")
print("="*80)
response = agent.chat(q)
print(f"\n[PhosphogypsumBot 回复]:\n{response}\n")
EOF
else
    echo "[Default Benchmark Mode] Running phosphogypsum domain verification queries..."
    python - << 'EOF'
import os
from chat_agent.agent import PhosphogypsumAgent

timeout_val = float(os.getenv("LLM_TIMEOUT", "3600.0"))
agent = PhosphogypsumAgent(
    base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
    model=os.getenv("LLM_MODEL", "Qwen/Qwen3.8-Flash-Next"),
    timeout=timeout_val
)

queries = [
    "请列出当前支持的所有磷石膏综合利用技术路径代码与名称。",
    "针对 PG-CementProd（磷石膏制水泥）和 PG-REEextract（磷石膏提取稀土）路径，分别计算 1 吨磷石膏处理下的 LCA 环境影响与 TEA 经济指标，并给出对比分析。",
    "在以渣定产政策下，综合考虑 TRL 熟化度、碳减排与经济回报，对所有路径进行多准则优选打分，推荐最具实施价值的方案。"
]

job_id = os.getenv("SLURM_JOB_ID", "local")
report_file = f"reports/agent_decision_report_{job_id}.md"

with open(report_file, "w", encoding="utf-8") as rf:
    rf.write(f"# PhosphogypsumBot 自主决策与综合评估报告 (Job ID: {job_id})\n\n")
    rf.write("> [!NOTE]\n")
    rf.write("> **计算环境与元数据**\n")
    rf.write(f"> - **作业编号**：Job ID {job_id}\n")
    rf.write(f"> - **模型底座**：{os.getenv('LLM_MODEL', 'Qwen/Qwen3.8-Flash-Next')}\n")
    rf.write(f"> - **执行节点**：{os.getenv('SLURMD_NODENAME', 'n0517')} ({os.getenv('SLURM_JOB_PARTITION', '9a14a')})\n")
    rf.write(f"> - **推理框架**：llama-server (Qwen3.8-Flash-Next 125B MoE)\n\n")
    rf.write("---\n\n")

for i, q in enumerate(queries, 1):
    print("\n" + "="*80)
    print(f"[Query {i}] 👨‍🔬 研究员提问: {q}")
    print("="*80)
    # Reset conversation context between queries to avoid context explosion on pure CPU
    agent.reset_conversation()
    response = agent.chat(q)
    print(f"\n[PhosphogypsumBot 回复]:\n{response}\n")

    with open(report_file, "a", encoding="utf-8") as rf:
        rf.write(f"## 查询 {i}：{q}\n\n")
        rf.write(f"{response}\n\n---\n\n")

print(f"\n[Success] Complete decision report saved to: {report_file}")
EOF
fi

echo "======================================================================="
echo "All autonomous queries completed successfully! (Job ID: ${SLURM_JOB_ID:-local})"
echo "Finished at: $(date)"
echo "======================================================================="
