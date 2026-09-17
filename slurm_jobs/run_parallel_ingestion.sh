#!/bin/bash
#SBATCH --job-name=pg_milvus_lite
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64          # 单节点分配64核（本地构建向量库足够，亦可按需设为192）
# #SBATCH --array=0-49              # [注意] 测试本地 Milvus-Lite 时注释此行（单任务运行）；
                                    # 本地嵌入式 .db 文件不支持多节点同时并发写入（避免文件锁冲突）
#SBATCH --output=slurm_jobs/logs/ingest_milvus_%j.log

# 如果将来并行解析新PDF需要多节点阵列，日志输出请用:
# #SBATCH --output=slurm_jobs/logs/ingest_array_%A_%a.log

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

# Ensure directories exist
mkdir -p slurm_jobs/logs
mkdir -p datahub/processed/milvus
mkdir -p datahub/processed/lightrag_db

echo "============================================================"
echo " Phosphogypsum Bot Ingestion Worker (Milvus Lite Mode)"
echo " Job ID: ${SLURM_JOB_ID:-N/A} | Task: ${SLURM_ARRAY_TASK_ID:-Single}"
echo " Host: $(hostname)"
echo " Partition: 9a14a"
echo " Working Dir: $(pwd)"
echo "============================================================"

# Prevent ONNX Runtime from attempting invalid thread affinity on Slurm cgroups
export ORT_DISABLE_THREAD_AFFINITY=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

# Set shared tiktoken cache directory
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"

# Ingestion configuration
export MINERU_MODEL_SOURCE=local

# =============================================================================
# 向量与图存储配置 (Vector & Graph Storage Configuration)
# =============================================================================
# 1. 向量存储: 保持 MilvusVectorDBStorage，使用 Milvus Lite 读写本地 .db 文件
#    【核心避坑】使用 LIGHTRAG_MILVUS_URI 并务必 unset MILVUS_URI！
#    PyMilvus 内部全局单例在 import pymilvus 时会自动读取 MILVUS_URI 并强制要求以 http[s]:// 开头。
#    若将 MILVUS_URI 设成本地路径，会导致 PyMilvus 导入时报 ConnectionConfigException。
export LIGHTRAG_VECTOR_STORAGE="MilvusVectorDBStorage"
export LIGHTRAG_MILVUS_URI="$(pwd)/datahub/processed/milvus/lightrag_milvus.db"
export MILVUS_DB_NAME="default"
unset MILVUS_URI

# 2. 图谱存储: 使用纯本地 NetworkX（保存为 GraphML 文件）
#    因当前计算节点未启动独立 Neo4j 数据库服务，清除 Neo4j 配置以防止产生网络连接拒绝报错
unset LIGHTRAG_GRAPH_STORAGE
unset NEO4J_URI
unset NEO4J_USERNAME
unset NEO4J_PASSWORD

echo "[Vector Storage] MilvusVectorDBStorage (Milvus Lite)"
echo "[Milvus DB File] ${LIGHTRAG_MILVUS_URI}"
echo "[Graph Storage]  Local NetworkX (GraphML file in lightrag_db/)"

# =============================================================================
# LLM / Embedding 服务地址 (如需覆盖 .env 中的配置，可取消注释并修改)
# =============================================================================
# export LLM_BASE_URL="http://127.0.0.1:11434/v1"
# export EMBEDDING_BASE_URL="http://127.0.0.1:11436/v1"

# =============================================================================
# 执行知识库构建任务
# =============================================================================

# Step 1: PDF 文献解析 (MinerU)
# 目前 72 篇文献已全部解析完成并保存在 datahub/interim/papers/parsed/ 中
# 测试向量库时默认跳过此步以节省时间；如放入了新的 PDF 文献，可取消下方注释
# echo ">>> Step 1: Parsing PDFs..."
# python scripts/build_knowledge_graph.py \
#   --step parse \
#   --parser mineru

# Step 2: 构建知识图谱与向量库索引 (LightRAG Indexing)
echo ""
echo ">>> Step 2: Indexing Documents with LightRAG (Milvus Lite DB) <<<"

# 防护措施：如果是以阵列作业提交，仅允许 Task 0 执行本地写入，避免多节点并发写冲突
if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
    echo "[Info] Array Task ID is $SLURM_ARRAY_TASK_ID. Local Milvus DB is updated by Task 0. Exiting."
    exit 0
fi

# 【测试配置】先试运行前 3 篇文献，验证 Milvus Lite 本地文件写入与抽取流程是否正常
# 确认无误后，将下面这一行注释掉，并解开 LIMIT_ARG="" 那一行即可全量处理全部 72 篇文献
LIMIT_ARG="--limit 3"
# LIMIT_ARG=""  # 全量处理所有文献时取消此行注释

python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag \
  ${LIMIT_ARG}

echo "============================================================"
echo " Ingestion Step 2 Complete!"
echo " Milvus Lite file status:"
ls -lh datahub/processed/milvus/
echo "============================================================"
