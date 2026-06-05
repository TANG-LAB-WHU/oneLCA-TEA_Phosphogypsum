#!/bin/bash
#SBATCH --job-name=pgbot_v100
#SBATCH --partition=gpu
#SBATCH --account=tangsiqi      # Replace with your supervisor's account name
#SBATCH --nodes=2                 # Default to 2 nodes for dedicated deployment, set to 1 for Single-Node NUMA Isolation, or 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4              # Request all 4 V100 GPU cards on each node (16GB each)
#SBATCH --cpus-per-task=20        # Allocate 20 CPU cores per node (full node: 5 cores per GPU max)
#SBATCH --mem=118G                # Max safe RAM per node (~118GB out of 128GB physical)
#SBATCH --time=72:00:00           # Run time limit (up to 7 days)
#SBATCH --output=pgbot_v100_%j.log

# Load CUDA environment to resolve "libcudart.so.12 => not found"
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda/12.2 2>/dev/null || module load cuda/12.0 2>/dev/null || module load cuda 2>/dev/null

# Activate conda environment to access dynamic library fallbacks
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

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

# Get the hostnames of the allocated compute nodes
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_NODES=${#nodes[@]}

echo "============================================================"
echo "  PhosphogypsumBot Dual-LLM Server"
echo "  Allocated Nodes: ${nodes[*]}"
echo "  Total Nodes: $NUM_NODES"
echo "============================================================"

# -----------------------------------------------------------------------------
# WHU V100 node physical topology (per node):
#   Socket 0 (NUMA node0): CPU 0-9,  GPU 0 & GPU 1
#   Socket 1 (NUMA node1): CPU 10-19, GPU 2 & GPU 3
# Each GPU: Nvidia Tesla V100 16GB, interconnected via NVLink within node
# Node interconnect: Intel OPA 100Gbps
# -----------------------------------------------------------------------------

if [ "$NUM_NODES" -eq 1 ]; then
    echo "=== Single-Node Mode: NUMA Hardware Isolation ==="
    # Pin each model to its own CPU socket and GPU pair to eliminate
    # cross-socket QPI latency and memory bus contention.

    # 1. Reasoner LLM (Qwen3.6-27B) -> Socket 0 (NUMA node0: CPU 0-9, GPU 0 & 1)
    #    Model: 27GB Q8_0 -> fits in 2x 16GB = 32GB VRAM with 5GB KV cache headroom
    CUDA_VISIBLE_DEVICES=0,1 \
    numactl --cpunodebind=0 --membind=0 \
      /project/tangsiqi/software/llama.cpp/build_v100/bin/llama-server \
      --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf \
      --host 0.0.0.0 \
      --port 11434 \
      --threads 8 \
      --ctx-size 32768 \
      --split-mode layer \
      --tensor-split 0.5,0.5 \
      --n-gpu-layers 99 &
    PID_REASONER=$!

    # 2. Parser VLM (Qwen3.6-35B-A3B-Vision) -> Socket 1 (NUMA node1: CPU 10-19, GPU 2 & 3)
    #    Model: 36GB + 861MB projector -> 32GB VRAM + ~5GB spill to Socket 1 local RAM
    CUDA_VISIBLE_DEVICES=2,3 \
    numactl --cpunodebind=1 --membind=1 \
      /project/tangsiqi/software/llama.cpp/build_v100/bin/llama-server \
      --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
      --mmproj /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf \
      --host 0.0.0.0 \
      --port 11435 \
      --threads 8 \
      --ctx-size 16384 \
      --split-mode layer \
      --tensor-split 0.5,0.5 \
      --n-gpu-layers 99 &
    PID_PARSER=$!

    # 3. Embedding Model (Qwen3-Embedding-8B-Q4_K_M) -> Socket 0, GPU 0 (with CPU spillover)
    #    Model: 4.4GB Q4_K_M
    CUDA_VISIBLE_DEVICES=0 \
    numactl --cpunodebind=0 --membind=0 \
      /project/tangsiqi/software/llama.cpp/build_v100/bin/llama-server \
      --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3-Embedding-8B-Q4_K_M.gguf \
      --embedding \
      --host 0.0.0.0 \
      --port 11436 \
      --threads 4 \
      --ctx-size 8192 \
      --n-gpu-layers 99 &
    PID_EMBED=$!

    echo ""
    echo ">>> Reasoner LLM (PID $PID_REASONER) listening on ${nodes[0]}:11434"
    echo ">>> Parser VLM   (PID $PID_PARSER)   listening on ${nodes[0]}:11435"
    echo ">>> Embed Model  (PID $PID_EMBED)    listening on ${nodes[0]}:11436"
    echo ""
    echo "SSH tunnel command (run on your local Mac):"
    echo "  ssh -N -L 11434:${nodes[0]}:11434 -L 11435:${nodes[0]}:11435 -L 11436:${nodes[0]}:11436 tangsiqi@<login-node-ip>"

else
    echo "=== Multi-Node Mode: Dedicated Full-Node Deployment ($NUM_NODES nodes) ==="
    # Each model gets its own physical server with all 4 GPUs (64GB VRAM total),
    # eliminating any VRAM spillover to CPU RAM entirely.

    node1=${nodes[0]}
    node2=${nodes[1]}

    # 1. Reasoner LLM on Node 1 (all 4 GPUs, 64GB VRAM)
    #    Model: 27GB Q8_0 -> 100% in VRAM, 37GB free for massive KV cache
    echo "Launching Reasoner LLM on $node1 (all 4 GPUs)..."
    srun --nodelist=$node1 -N 1 -n 1 --gres=gpu:4 --export=ALL \
      numactl --interleave=all \
      /project/tangsiqi/software/llama.cpp/build_v100/bin/llama-server \
      --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf \
      --host 0.0.0.0 \
      --port 11434 \
      --threads 16 \
      --ctx-size 32768 \
      --split-mode layer \
      --n-gpu-layers 99 &
    PID_REASONER=$!

    # 2. Parser VLM on Node 2 (all 4 GPUs, 64GB VRAM)
    #    Model: 36GB + 861MB projector -> 100% in VRAM, no CPU RAM spill
    echo "Launching Parser VLM on $node2 (all 4 GPUs)..."
    srun --nodelist=$node2 -N 1 -n 1 --gres=gpu:4 --export=ALL \
      numactl --interleave=all \
      /project/tangsiqi/software/llama.cpp/build_v100/bin/llama-server \
      --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
      --mmproj /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf \
      --host 0.0.0.0 \
      --port 11435 \
      --threads 16 \
      --ctx-size 16384 \
      --split-mode layer \
      --n-gpu-layers 99 &
    PID_PARSER=$!

    # 3. Embedding Model on Node 1 (using Q8_0 since node1 has 4x GPUs and 64GB VRAM, plenty of space)
    echo "Launching Embedding Model on $node1..."
    srun --nodelist=$node1 -N 1 -n 1 --gres=gpu:1 --overlap --export=ALL \
      /project/tangsiqi/software/llama.cpp/build_v100/bin/llama-server \
      --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3-Embedding-8B-Q8_0.gguf \
      --embedding \
      --host 0.0.0.0 \
      --port 11436 \
      --threads 8 \
      --ctx-size 8192 \
      --n-gpu-layers 99 &
    PID_EMBED=$!

    echo ""
    echo ">>> Reasoner LLM (PID $PID_REASONER) listening on $node1:11434"
    echo ">>> Parser VLM   (PID $PID_PARSER)   listening on $node2:11435"
    echo ">>> Embed Model  (PID $PID_EMBED)    listening on $node1:11436"
    echo ""
    echo "SSH tunnel command (run on your local Mac):"
    echo "  ssh -N -L 11434:${node1}:11434 -L 11435:${node2}:11435 -L 11436:${node1}:11436 tangsiqi@<login-node-ip>"
fi

echo ""
echo "After SSH tunnel is established, set in your local .env:"
echo "  LLM_BASE_URL=http://127.0.0.1:11434/v1"
echo "  VLM_BASE_URL=http://127.0.0.1:11435/v1"
echo "  EMBEDDING_BASE_URL=http://127.0.0.1:11436/v1"
echo ""

# Wait for both server processes
wait
