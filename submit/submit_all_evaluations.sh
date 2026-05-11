#!/bin/bash
#SBATCH --job-name=submit_all_evaluations
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

REPO_DIR="/ocean/projects/mth250011p/troemer/skin-lesions"
cd "$REPO_DIR"

JOB_NAME="${SLURM_JOB_NAME:-$(basename "$0" .sh)}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="${REPO_DIR}/logs/${JOB_NAME}-${JOB_ID}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/stdout.log" 2> "${LOG_DIR}/stderr.log"

echo "Log directory: $LOG_DIR"
echo "Submitting all evaluation jobs on $(date)"
echo "Working directory: $(pwd)"

eval_scripts=(
    "submit/submit_evaluate_pad_ufes20_full_image_models.sh"
    "submit/submit_evaluate_pad_ufes20_lesion_white_models.sh"
)

eval_jobs=()
for eval_script in "${eval_scripts[@]}"; do
    eval_job=$(sbatch --parsable "$eval_script")
    eval_jobs+=("$eval_job")
    echo "$(basename "$eval_script" .sh): $eval_job"
done

eval_dependency=$(IFS=:; echo "${eval_jobs[*]}")
bootstrap_job=$(sbatch --parsable --dependency=afterok:${eval_dependency} submit/submit_bootstrap.sh)
echo "submit_bootstrap after all evaluation jobs: $bootstrap_job"

echo "All evaluation jobs submitted on $(date)."
