#!/bin/bash
#SBATCH --job-name=neo4j_service
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_jobs/logs/neo4j_%j.log

# Ensure logs directory exists
mkdir -p slurm_jobs/logs

# Set directory variables for persistence
DATA_DIR="$(pwd)/data/processed/neo4j/data"
LOGS_DIR="$(pwd)/data/processed/neo4j/logs"
CONF_DIR="$(pwd)/data/processed/neo4j/conf"
IMPORT_DIR="$(pwd)/data/processed/neo4j/import"

mkdir -p "$DATA_DIR" "$LOGS_DIR" "$CONF_DIR" "$IMPORT_DIR"

echo "============================================================"
echo " Starting Neo4j Database Service via Singularity/Apptainer"
echo " Host: $(hostname)"
echo " Ports: 7474 (HTTP), 7687 (Bolt)"
echo " Data Dir: $DATA_DIR"
echo "============================================================"

# Set up default credentials
export NEO4J_AUTH="neo4j/password123"

# Run Neo4j container using Apptainer (or fallback to Singularity)
if command -v apptainer &> /dev/null; then
    CONTAINER_RUNNER="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_RUNNER="singularity"
else
    echo "[ERROR] Neither Apptainer nor Singularity found in path."
    exit 1
fi

echo "Using container runner: $CONTAINER_RUNNER"

# Run container in background/foreground. Since sbatch runs as a job,
# running it in foreground will keep the job alive.
$CONTAINER_RUNNER run \
    --bind "$DATA_DIR":/data \
    --bind "$LOGS_DIR":/logs \
    --bind "$CONF_DIR":/conf \
    --bind "$IMPORT_DIR":/import \
    docker://neo4j:5.12.0
