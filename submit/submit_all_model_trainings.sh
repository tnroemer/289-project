#!/bin/bash
#SBATCH --job-name=submit_all_model_trainings
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
echo "Submitting all model training jobs on $(date)"
echo "Working directory: $(pwd)"

train_scripts=(
    "submit/submit_train_ham10000_full_image_cnn.sh"
    "submit/submit_train_ham10000_full_image_vit.sh"
    "submit/submit_train_ham10000_full_image_resnet.sh"
    "submit/submit_train_ham10000_full_image_pretrained_resnet50.sh"
    "submit/submit_train_ham10000_lesion_white_cnn.sh"
    "submit/submit_train_ham10000_lesion_white_vit.sh"
    "submit/submit_train_ham10000_lesion_white_resnet.sh"
    "submit/submit_train_ham10000_lesion_white_pretrained_resnet50.sh"
    "submit/submit_train_pad_ufes20_full_image_resnet.sh"
)

for train_script in "${train_scripts[@]}"; do
    train_job=$(sbatch --parsable "$train_script")
    echo "$(basename "$train_script" .sh): $train_job"
done

echo "All model training jobs submitted on $(date)."
