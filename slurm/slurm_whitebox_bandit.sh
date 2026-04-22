#!/bin/bash
#SBATCH --job-name=whitebox-bandit
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH --time=1:00:00
# White-box bandit supervised benchmark (IQN, QR-DQN, C51, optional CQN, MoG-CQN).
# Launches 5 Slurm array tasks → seeds 0..4 via SLURM_ARRAY_TASK_ID (override with SEED=...).
#
# Training writes ``purejaxql/whitebox_bandit_run{figure_tag}.npz`` per seed (minimal export; no per-seed CSV).
# After all tasks finish, aggregate mean±SE tables + panel ribbons + single combined CSV from NPZ:
#
#   uv run --no-sync python -m purejaxql.whitebox_bandit_supervised \
#     --aggregate-metrics-dir figures/whitebox_bandit \
#     --aggregate-tag-stem _omega_15 \
#     --aggregate-npz-dir purejaxql
#
# Match ``--aggregate-tag-stem`` to TrainConfig.figure_tag before ``_seed<N>`` (default ``_omega_15``).
#
# Submit a single seed only: sbatch --array=3-3 slurm/slurm_whitebox_bandit.sh   # or --array=0-0

# sbatch slurm/slurm_whitebox_bandit.sh 

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-WhiteboxBandit_Omega15}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"

echo "=========================================="
echo "White-box bandit (supervised)"
echo "=========================================="
echo "Seed:               ${SEED}"
echo "W&B experiment tag: ${EXPERIMENT_TAG}"
echo "Project:            ${WANDB_PROJECT}"
echo "Job ID / task:      ${SLURM_JOB_ID:-}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
echo "Host:               $(hostname)"
echo "GPU:                ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:              $(date)"
echo "=========================================="

srun uv run --no-sync python -m purejaxql.whitebox_bandit_supervised \
  --seed "${SEED}" \
  --append-seed-to-tag \
  --wandb \
  --wandb-experiment-tag "${EXPERIMENT_TAG}" \
  --no-cqn \

echo "Completed"
echo "End: $(date)"
echo "=========================================="
