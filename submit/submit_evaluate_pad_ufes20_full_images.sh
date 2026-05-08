#!/bin/bash
#SBATCH --job-name=evaluate_pad_ufes20_full_images
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --account=mth250011p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR="/ocean/projects/mth250011p/troemer/skin-lesions"
cd "$REPO_DIR"

JOB_NAME="${SLURM_JOB_NAME:-$(basename "$0" .sh)}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="${REPO_DIR}/logs/${JOB_NAME}-${JOB_ID}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/stdout.log" 2> "${LOG_DIR}/stderr.log"

echo "Log directory: $LOG_DIR"
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /jet/home/troemer/.conda/envs/stat214
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
PYTHON_BIN="python"
export PYTHONUNBUFFERED=1

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

if [[ -f evaluate_pad_ufes20_full_images.py ]]; then
    SCRIPT_PATH="evaluate_pad_ufes20_full_images.py"
else
    SCRIPT_PATH="src/evaluate_pad_ufes20_full_images.py"
fi

/jet/home/troemer/.conda/envs/stat214/bin/python "$SCRIPT_PATH"

echo "Job finished on $(date)"
