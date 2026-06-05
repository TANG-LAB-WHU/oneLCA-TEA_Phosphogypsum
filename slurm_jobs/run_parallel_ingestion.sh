#!/bin/bash
#SBATCH --job-name=pg_parallel_ingest
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-49               # Launch 50 tasks (each processing a subset of files)
#SBATCH --output=slurm_jobs/logs/ingest_array_%A_%a.log

# Note: %A is main Job ID, %a is Slurm Array Task ID

# Load CUDA environment
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

# Change to project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

echo "============================================================"
echo " Phosphogypsum Bot Parallel Ingestion Worker"
echo " Array Job ID: ${SLURM_ARRAY_JOB_ID} | Task ID: ${SLURM_ARRAY_TASK_ID}"
echo " Host: $(hostname)"
echo " Partition: 9a14a"
echo " Working Dir: $(pwd)"
echo "============================================================"

# Ensure logs dir exists
mkdir -p slurm_jobs/logs

# Set shared tiktoken cache directory
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"

# Ingestion configuration
export MINERU_MODEL_SOURCE=local

# Set database coordinates (pointing to our active Neo4j & Milvus nodes on the cluster)
export LIGHTRAG_GRAPH_STORAGE="Neo4JStorage"
export LIGHTRAG_VECTOR_STORAGE="MilvusVectorDBStorage"
export NEO4J_URI="bolt://localhost:7687"           # Update with cluster-assigned Neo4j node hostname
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export MILVUS_URI="http://localhost:19530"          # Update with cluster-assigned Milvus node hostname
export MILVUS_DB_NAME="lightrag"

# Run sharded parsing and indexing task
# Sharding is automatically calculated inside build_knowledge_graph.py using:
# SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT
python scripts/build_knowledge_graph.py \
  --step parse \
  --parser mineru

# Wait briefly to serialize DB connection attempts across array tasks
sleep $((SLURM_ARRAY_TASK_ID * 2))

python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag
