#!/bin/bash
#SBATCH --job-name=milvus_service
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_jobs/logs/milvus_%j.log

# Ensure logs directory exists
mkdir -p slurm_jobs/logs

# Set directory variables for persistence
DATA_DIR="$(pwd)/data/processed/milvus/data"
CONF_DIR="$(pwd)/data/processed/milvus/conf"

mkdir -p "$DATA_DIR" "$CONF_DIR"

# Generate milvus.yaml if it doesn't exist (to run standalone inside container)
MILVUS_YAML="$CONF_DIR/milvus.yaml"
if [ ! -f "$MILVUS_YAML" ]; then
    echo "Creating default milvus.yaml configuration..."
    cat <<EOF > "$MILVUS_YAML"
# Standalone Milvus Configuration
etcd:
  use: embed
  data.dir: /var/lib/milvus/etcd
metastore:
  type: sqlite
storage:
  path: /var/lib/milvus/data
queryNode:
  gracefulTimeOut: 0
dataNode:
  gracefulTimeOut: 0
EOF
fi

echo "============================================================"
echo " Starting Milvus Standalone Service via Singularity/Apptainer"
echo " Host: $(hostname)"
echo " Ports: 19530 (gRPC), 9091 (REST API)"
echo " Data Dir: $DATA_DIR"
echo "============================================================"

# Run Milvus container using Apptainer (or fallback to Singularity)
if command -v apptainer &> /dev/null; then
    CONTAINER_RUNNER="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_RUNNER="singularity"
else
    echo "[ERROR] Neither Apptainer nor Singularity found in path."
    exit 1
fi

echo "Using container runner: $CONTAINER_RUNNER"

# Run Milvus standalone in container
$CONTAINER_RUNNER run \
    --bind "$DATA_DIR":/var/lib/milvus \
    --bind "$MILVUS_YAML":/milvus/configs/milvus.yaml \
    docker://milvusdb/milvus:v2.3.10 \
    milvus run standalone
