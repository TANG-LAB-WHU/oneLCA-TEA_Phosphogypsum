#!/bin/bash
#SBATCH --job-name=pgbot_cpu_moe
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi      # Replace with your supervisor's account name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        # Allocate 32 physical EPYC cores
#SBATCH --mem=120G                # Allocate 120GB system memory
#SBATCH --time=120:00:00          # Run time limit (up to 5 days)
#SBATCH --output=pgbot_cpu_%j.log

# 1. Start Reasoner LLM (Qwen3.6-35B-A3B-UD-Q8_K_XL) on CPU, Port 11434
/project/tangsiqi/software/llama.cpp/build_cpu/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-35B-A3B-Unsloth/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --host 0.0.0.0 \
  --port 11434 \
  --threads 32 \
  --ctx-size 16384 \
  --n-gpu-layers 0

# (Note: For CPU execution, we do not append "&" since it runs as the single primary process)
