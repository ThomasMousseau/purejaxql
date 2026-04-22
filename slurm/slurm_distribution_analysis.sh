#!/bin/bash
#SBATCH --job-name=distribution-analysis
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#
# Runs 5 seeds (array index) for run_distribution_analysis.py.
# Plot once runs are complete:
#
# uv run --no-sync python plot_distribution_analysis.py \
#   --experiment-tag "${WANDB_EXPERIMENT_TAG}"

# Submit: sbatch slurm/slurm_distribution_analysis.sh

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-DistAnalysis_v1}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"

echo "=========================================="
echo "Distribution analysis (offline)"
echo "=========================================="
echo "Seed:               ${SEED}"
echo "W&B experiment tag: ${EXPERIMENT_TAG}"
echo "Project:            ${WANDB_PROJECT}"
echo "Job ID / task:      ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
echo "Host:               $(hostname)"
echo "GPU:                ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:              $(date)"
echo "=========================================="

srun uv run --no-sync run_distribution_analysis.py \
  --seed "${SEED}" \
  --experiment-tag "${EXPERIMENT_TAG}"

echo "Completed"
echo "End: $(date)"
echo "=========================================="
