#!/bin/bash
#SBATCH --job-name=train_ham10000_lesion_white_resnet
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --account=mth250011p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR="${SKIN_LESIONS_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
conda activate "${SKIN_LESIONS_CONDA_ENV:-stat214}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-${REPO_DIR}/.wandb_env}"
if [ -f "$WANDB_ENV_FILE" ]; then source "$WANDB_ENV_FILE"; fi
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"

PYTHON_MODULE="training.train_ham10000_lesion_white_resnet"

python -m "$PYTHON_MODULE"

echo "Job finished on $(date)"
