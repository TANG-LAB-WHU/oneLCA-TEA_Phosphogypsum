# High-Performance Computing (HPC) & SLURM Deployment

This guide describes how to deploy and run the compute-heavy, dual-LLM (Reasoner LLM + Parser VLM) pipelines of **PhosphogypsumBot** on the **Wuhan University Supercomputing Center (WHU-SCC, 武汉大学超级计算中心)**. 

---

## 🏛️ WHU-SCC Cluster Environment Overview

The WHU-SCC scheduling environment relies on **Slurm** for resource allocation. The model pipelines are optimized and benchmarked across three key hardware partitions:
1.  **`a100x4` Partition**: NVIDIA A100 (40GB VRAM) nodes. Recommended for full-precision model loads.
2.  **`gpu` Partition**: NVIDIA V100 (16GB VRAM) nodes. Used for budget-friendly multi-GPU split-inference execution.
3.  **`9a14a` Partition**: CPU-only nodes hosting dual-socket AMD EPYC processors (192 physical cores, 768GB RAM). Recommended for executing massive MoE models or during cluster queue congestion.

---

## 1. Environment & Code Preparation

### 1.1 Code Upload
Avoid running the pipeline directly from your `/home` folder, as it has strict storage quotas and slow I/O. Use the high-performance scratch or project volumes:
```bash
# Execute from your local workstation terminal
scp -r /Users/siqi/GitHub/oneLCA-TEA_Phosphogypsum username@supercomputer_ip:/project/username/
```

### 1.2 Conda Isolation Setup
Log into the WHU-SCC terminal, load the Anaconda module, and create a virtual environment:
```bash
ssh username@supercomputer_ip
cd /project/username/oneLCA-TEA_Phosphogypsum

# Load cluster conda modules
module load anaconda3  # or miniconda3
conda create -n pgbot python=3.12 -y
conda activate pgbot

# Install dependencies in editable mode
pip install -r requirements.txt
pip install -e .[ai,viz,rag,dev]
```

---

## 2. Slurm Job Blueprint Scenarios

We have prepared pre-configured Slurm batch files inside the `slurm_jobs/` folder. Use the following profiles:

### Scenario A: Unified Qwen Dual-GPU (A100 Partition - Recommended)
*   **Script**: [slurm_jobs/run_dual_a100.sh](../slurm_jobs/run_dual_a100.sh)
*   **Hardware Allocation**: 2 GPU cards (A100 40GB each), 32 CPU cores, 120GB RAM.
*   **Model Configuration**:
    *   **GPU 0**: Reasoner LLM -> `Qwen3.6-27B-Q8_0.gguf` (27GB, loaded entirely into VRAM) on Port `11434`.
    *   **GPU 1**: Parser VLM -> `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (36GB) + `mmproj-BF16.gguf` (861MB) on Port `11435`.

To run, submit to the queue:
```bash
sbatch slurm_jobs/run_dual_a100.sh
```

### Scenario B: Split Multi-GPU (V100 Partition - Budget Option)
*   **Script**: [slurm_jobs/run_dual_v100.sh](../slurm_jobs/run_dual_v100.sh)
*   **Hardware Allocation**: 3 GPU cards (V100 16GB each), 14 CPU cores, 80GB RAM.
*   **Model Configuration**:
    *   **GPU 0**: Reasoner LLM -> `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` (4.6GB) on Port `11434`.
    *   **GPU 1 & 2**: Parser VLM -> `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` split equally using `--tensor-split 0.5,0.5` across both V100 cards on Port `11435`.

To run, submit to the queue:
```bash
sbatch slurm_jobs/run_dual_v100.sh
```

### Scenario C: Hybrid CPU / GPU Partition Execution
*   **Script**: [slurm_jobs/run_cpu_reasoner.sh](../slurm_jobs/run_cpu_reasoner.sh)
*   **Hardware Allocation**: 32 EPYC CPU cores, 120GB RAM (No GPU allocation).
*   **Model Configuration**: Runs the large MoE Reasoner `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` on CPU entirely on Port `11434` utilizing 32 threads.

To run, submit to the queue:
```bash
sbatch slurm_jobs/run_cpu_reasoner.sh
```

---

## ⚡ 3. NUMA-Aware Socket Isolation & Pinning (Critical)

On CPU-only partitions (like `9a14a`), nodes contain dual-socket AMD EPYC architectures. Modifying threads without NUMA node bindings will trigger severe **cross-socket memory-bus latency** and memory bandwidth saturation, degrading generation speeds.

To prevent this, the CPU script [slurm_jobs/test_kg_pipeline_cpu.sh](../slurm_jobs/test_kg_pipeline_cpu.sh) implements explicit process binding using `numactl`:

```bash
# 1. Bind the Reasoner server to NUMA Node 0 (Socket 0, Cores 0-95)
numactl --cpunodebind=0 --membind=0 \
  /project/username/software/llama.cpp/build_cpu/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --threads 80 \
  --port 11434 &

# 2. Bind the Embedding server to NUMA Node 1 (Socket 1, Cores 96-191)
numactl --cpunodebind=1 --membind=1 \
  /project/username/software/llama.cpp/build_cpu/bin/llama-server \
  --model /path/to/Qwen3-Embedding-8B-Q8_0.gguf \
  --embedding \
  --threads 80 \
  --port 11436 &
```

> **Warning**: Never remove `numactl --cpunodebind` parameters from Slurm CPU scripts. Doing so can drop tokens-per-second by over 60%.

---

## 💬 4. Running Interactive Sessions (Interactive Mode)

If you wish to test code interactively, run unit tests, or run the chat CLI directly from the terminal prompt, allocate interactive GPU resources:

```bash
# 1. Request an interactive shell (e.g., V100 GPU node)
srun --partition=gpu --account=supervisor --nodes=1 --gres=gpu:1 --cpus-per-task=10 --mem=60G --time=04:00:00 --pty /bin/bash

# 2. Once shell starts on the allocated node (e.g., node125), activate conda
conda activate pgbot

# 3. Terminal A: Start llama-server backend
/project/username/software/llama.cpp/build_v100/bin/llama-server \
  --model /home/tangsiqi/scratch/ai_models/qwen/Qwen3.6-27B/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 --port 11434 --ctx-size 16384 --n-gpu-layers 99

# 4. Terminal B (Or using tmux): Start the interactive Agent CLI
PYTHONPATH=. python -m chat_agent.cli
```
