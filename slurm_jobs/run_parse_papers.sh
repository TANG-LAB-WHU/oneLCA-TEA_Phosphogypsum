#!/bin/bash
#SBATCH --job-name=pg_parse_mineru
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/scratch/tangsiqi/liyi_folder/oneLCA-TEA_Phosphogypsum
#SBATCH --output=/scratch/tangsiqi/liyi_folder/oneLCA-TEA_Phosphogypsum/logs/slurm/parse_mineru_%j.log
#SBATCH --error=/scratch/tangsiqi/liyi_folder/oneLCA-TEA_Phosphogypsum/logs/slurm/parse_mineru_%j.log

# =============================================================================
# Phosphogypsum Bot: Dedicated Literature Parser (MinerU Multi-Modal Engine)
# Parses remaining unparsed scientific PDFs into structured Markdown with LaTeX
# formulas and HTML tables. Zero database dependencies.
# =============================================================================

# 1. Load CUDA & Environment Modules
module load nvidia/cuda/12.9 2>/dev/null || module load nvidia/cuda/12.2 2>/dev/null

# 2. Activate Conda Environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.bashrc
conda activate pgbot

# 3. Anchor Working Directory to Project Root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")"
fi

if [ "$(basename "$(pwd)")" = "slurm_jobs" ]; then
    cd ..
fi

# 4. Optimization & Model Configuration (Offline/Local)
export TIKTOKEN_CACHE_DIR="/home/tangsiqi/.cache/tiktoken"
export MINERU_MODEL_SOURCE=local
export ORT_DISABLE_THREAD_AFFINITY=1

# Ensure necessary directories exist
mkdir -p logs/slurm datahub/interim/papers/parsed datahub/processed

echo "============================================================"
echo " Phosphogypsum Bot: Dedicated Paper Parser (MinerU)"
echo " Job ID:       ${SLURM_JOB_ID:-interactive}"
echo " Host Node:    $(hostname)"
echo " Partition:    9a14a (32 Cores, 128G RAM)"
echo " Working Dir:  $(pwd)"
echo " Start Time:   $(date)"
echo "============================================================"

# Pre-flight Status: Count pending documents
TOTAL_PDFS=$(find datahub/raw/papers/unparsed -name "*.pdf" 2>/dev/null | wc -l)
if [ -f "datahub/processed/ingestion_registry.json" ]; then
    DONE_COUNT=$(grep -o '"status": "success"' datahub/processed/ingestion_registry.json 2>/dev/null | wc -l)
else
    DONE_COUNT=0
fi
echo " Total Raw PDFs:         $TOTAL_PDFS"
echo " Already Parsed & Valid: $DONE_COUNT"
echo " Pending to Parse:       $((TOTAL_PDFS - DONE_COUNT))"
echo " Output Destination:     datahub/interim/papers/parsed/<paper_stem>/auto/<paper_stem>.md"
echo "============================================================"

# 5. Execute Incremental Parsing
# Automatically skips already parsed papers via SHA256 registry check
python scripts/build_knowledge_graph.py \
    --step parse \
    --parser mineru

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " [SUCCESS] MinerU literature parsing finished cleanly!"
    NEW_DONE=$(grep -o '"status": "success"' datahub/processed/ingestion_registry.json 2>/dev/null | wc -l)
    echo " Final Registered Papers: $NEW_DONE / $TOTAL_PDFS"
else
    echo " [ERROR] Parsing process exited with code $EXIT_CODE."
fi
echo " Finished at: $(date)"
echo "============================================================"

exit $EXIT_CODE
