#!/bin/bash
#SBATCH --job-name=submit_all_data_creation
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1900M
#SBATCH --account=mth250011p
#SBATCH --time=00:10:00

set -euo pipefail

REPO_DIR="${SKIN_LESIONS_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

JOB_NAME="${SLURM_JOB_NAME:-$(basename "$0" .sh)}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="${REPO_DIR}/logs/${JOB_NAME}-${JOB_ID}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/stdout.log" 2> "${LOG_DIR}/stderr.log"

echo "Log directory: $LOG_DIR"
echo "Submitting all data creation jobs on $(date)"
echo "Working directory: $(pwd)"

data_job=$(sbatch --parsable submit/submit_create_data.sh)
echo "submit_create_data: $data_job"

segmentation_job=$(sbatch --parsable --dependency=afterok:${data_job} submit/submit_train_ham10000_segmentation_model.sh)
echo "submit_train_ham10000_segmentation_model after $data_job: $segmentation_job"

lesion_white_job=$(sbatch --parsable --dependency=afterok:${segmentation_job} submit/submit_create_lesion_white_data.sh)
echo "submit_create_lesion_white_data after $segmentation_job: $lesion_white_job"

echo "All data creation jobs submitted on $(date)."
