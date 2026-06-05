#!/bin/bash
#SBATCH --job-name=pgbot_a100
#SBATCH --partition=a100x4
#SBATCH --account=tangsiqi         # Replace with your supervisor's account name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2              # Allocate 2 A100 GPU cards (40GB each)
#SBATCH --cpus-per-task=32        # Allocate 32 CPU cores (16 cores per GPU maximum)
#SBATCH --mem=120G                # Allocate 120GB system memory
#SBATCH --time=72:00:00           # Run time limit (up to 7 days)
#SBATCH --output=pgbot_a100_%j.log

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

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# 1. Start Reasoner LLM (Qwen3.6-27B-Instruct) on GPU 0, Port 11434
export CUDA_VISIBLE_DEVICES=0
/project/tangsiqi/software/llama.cpp/build_a100/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 11434 \
  --ctx-size 32768 \
  --n-gpu-layers 99 &
PID_REASONER=$!

# 2. Start Parser VLM (Qwen3.6-35B-A3B-Vision) on GPU 1, Port 11435
export CUDA_VISIBLE_DEVICES=1
/project/tangsiqi/software/llama.cpp/build_a100/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --mmproj /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 11435 \
  --ctx-size 16384 \
  --n-gpu-layers 99 &
PID_PARSER=$!

# 3. Start Embedding Model (Qwen3-Embedding-8B) on GPU 0, Port 11436
#    Using Q8_0 (7.5GB) since GPU 0 (40GB) running a 27GB model has ~13GB free VRAM
export CUDA_VISIBLE_DEVICES=0
/project/tangsiqi/software/llama.cpp/build_a100/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3-Embedding-8B-Q8_0.gguf \
  --embedding \
  --host 0.0.0.0 \
  --port 11436 \
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
echo ""

# Wait for background processes to complete (or until Slurm kills the job)
wait
