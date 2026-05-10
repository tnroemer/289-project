#!/bin/bash
#SBATCH --job-name=submit_full_pipeline
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
echo "Pipeline submitter started on $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

submit_job() {
    sbatch --parsable "$@"
}

echo "Submitting full skin-lesion pipeline from $REPO_DIR"

data_job=$(submit_job submit/submit_create_data.sh)
echo "create_data: $data_job"

lesion_white_job=$(submit_job --dependency=afterok:${data_job} submit/submit_create_lesion_white_data.sh)
echo "create_lesion_white_data after $data_job: $lesion_white_job"

train_jobs=()
for train_script in \
    submit/submit_train_ham10000_full_image_cnn.sh \
    submit/submit_train_ham10000_full_image_vit.sh \
    submit/submit_train_ham10000_full_image_resnet.sh \
    submit/submit_train_ham10000_full_image_pretrained_resnet50.sh \
    submit/submit_train_ham10000_lesion_white_cnn.sh \
    submit/submit_train_ham10000_lesion_white_vit.sh \
    submit/submit_train_ham10000_lesion_white_resnet.sh \
    submit/submit_train_ham10000_lesion_white_pretrained_resnet50.sh \
    submit/submit_train_pad_ufes20_full_image_resnet.sh
do
    train_job=$(submit_job --dependency=afterok:${lesion_white_job} "$train_script")
    train_jobs+=("$train_job")
    echo "$(basename "$train_script" .sh) after $lesion_white_job: $train_job"
done

train_dependency=$(IFS=:; echo "${train_jobs[*]}")

full_eval_job=$(submit_job \
    --dependency=afterok:${train_dependency} \
    submit/submit_evaluate_pad_ufes20_full_image_models.sh)
echo "evaluate_pad_ufes20_full_image_models after all training jobs: $full_eval_job"

lesion_eval_job=$(submit_job \
    --dependency=afterok:${train_dependency} \
    submit/submit_evaluate_pad_ufes20_lesion_white_models.sh)
echo "evaluate_pad_ufes20_lesion_white_models after all training jobs: $lesion_eval_job"

echo "Pipeline submitted."
echo "Slurm dependencies enforce: create data -> create lesion-white data -> train models -> evaluate PAD-UFES-20."
echo "Pipeline submitter finished on $(date)"
