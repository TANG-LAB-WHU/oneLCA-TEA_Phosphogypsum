#!/bin/bash
#SBATCH --job-name=pg_local_ingest
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64          # 单节点分配64核（构建本地向量库足够，亦可按需调整为192）
# #SBATCH --array=0-49              # [注意] 测试本地向量库时注释此行（单任务运行）；
                                    # 若未来需并行解析海量新PDF，再取消注释开启50节点阵列
#SBATCH --output=slurm_jobs/logs/ingest_local_%j.log

# 如果使用阵列作业，日志输出请用:
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
mkdir -p datahub/processed/lightrag_db

echo "============================================================"
echo " Phosphogypsum Bot Ingestion Worker (Local Storage Mode)"
echo " Job ID: ${SLURM_JOB_ID:-N/A} | Array Task: ${SLURM_ARRAY_TASK_ID:-None}"
echo " Host: $(hostname)"
echo " Partition: 9a14a"
echo " Working Dir: $(pwd)"
echo "============================================================"

# Set shared tiktoken cache directory
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"

# Ingestion configuration
export MINERU_MODEL_SOURCE=local

# =============================================================================
# 向量与图存储配置 (Vector & Graph Storage Configuration)
# =============================================================================
# 【当前激活】模式 1: 纯本地存储 (Pure Local Storage)
# - 向量存储: NanoVectorDB (直接保存在本地 JSON 文件)
# - 图谱存储: NetworkX (直接保存在本地 GraphML 文件)
# - 存储路径: datahub/processed/lightrag_db/
# - 优势: 完全无需启动 Milvus 或 Neo4j 外部服务，不依赖任何网络端口！
unset LIGHTRAG_GRAPH_STORAGE
unset LIGHTRAG_VECTOR_STORAGE
unset MILVUS_URI
unset MILVUS_DB_NAME
unset NEO4J_URI

echo "[Storage Mode] Using pure local storage (NanoVectorDB + NetworkX)"
echo "[Storage Path] $(pwd)/datahub/processed/lightrag_db"

# 模式 2: Milvus Lite (如需使用 pymilvus 本地嵌入文件存储，可取消注释下面两行)
# export LIGHTRAG_VECTOR_STORAGE="MilvusVectorDBStorage"
# export MILVUS_URI="$(pwd)/datahub/processed/milvus/lightrag.db"

# 模式 3: 集群分布式服务 (需后台预先启动 Neo4j 与 Milvus 容器作业)
# export LIGHTRAG_GRAPH_STORAGE="Neo4JStorage"
# export LIGHTRAG_VECTOR_STORAGE="MilvusVectorDBStorage"
# export NEO4J_URI="bolt://<neo4j_host>:7687"
# export MILVUS_URI="http://<milvus_host>:19530"
# export MILVUS_DB_NAME="lightrag"

# =============================================================================
# LLM / Embedding 服务地址 (如需覆盖 .env 中的配置，可取消注释并修改)
# =============================================================================
# export LLM_BASE_URL="http://127.0.0.1:11434/v1"
# export EMBEDDING_BASE_URL="http://127.0.0.1:11436/v1"
# export LLM_MODEL="Qwen/Qwen3.8-Flash-Next"
# export EMBEDDING_MODEL="qwen3-embedding:4b"

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

# Step 2: 构建知识图谱与本地向量库 (LightRAG Indexing)
echo ""
echo ">>> Step 2: Indexing Documents with LightRAG (Local NanoVectorDB) <<<"

# 防护措施：如果是以阵列作业提交，仅允许 Task 0 执行本地写入，避免多节点并发写入同一文件冲突
if [ -n "$SLURM_ARRAY_TASK_ID" ] && [ "$SLURM_ARRAY_TASK_ID" -ne 0 ]; then
    echo "[Info] Array Task ID is $SLURM_ARRAY_TASK_ID. Local indexing is handled by Task 0. Exiting."
    exit 0
fi

# 【测试配置】先试运行前 3 篇文献，验证本地向量库写入与抽取流程是否正常
# 确认无误后，将下面这一行注释掉，并解开 LIMIT_ARG="" 那一行即可全量处理全部 72 篇文献
LIMIT_ARG="--limit 3"
# LIMIT_ARG=""  # 全量处理所有文献时取消此行注释

python scripts/build_knowledge_graph.py \
  --step index \
  --engine lightrag \
  ${LIMIT_ARG}

echo "============================================================"
echo " Ingestion Step 2 Complete!"
echo " Local DB contents:"
ls -la datahub/processed/lightrag_db/
echo "============================================================"
