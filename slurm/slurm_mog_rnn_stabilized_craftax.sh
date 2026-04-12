#!/bin/bash
#SBATCH --job-name=mog-rnn-stab
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
# Single run: MoG-PQN-RNN stabilized (Double DQN + Polyak). See mog_pqn_rnn_stabilized_craftax.py
#
# Submit: sbatch slurm/slurm_mog_rnn_stabilized_craftax.sh
#
# Optional env overrides (same idea as the 2-algo script):
#   export WANDB_EXPERIMENT_TAG=MyTag
#   export CRAFTAX_ENV_NAME=Craftax-Symbolic-v1
#   export SEED=0

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax_MoG_RNN_stabilized}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"
SEED="${SEED:-0}"

echo "=========================================="
echo "MoG-PQN-RNN stabilized (Craftax)"
echo "=========================================="
echo "Seed:       ${SEED}"
echo "Env:        ${CRAFTAX_ENV_NAME}"
echo "W&B tag:    ${EXPERIMENT_TAG}"
echo "Project:    ${WANDB_PROJECT}"
echo "Job ID:     ${SLURM_JOB_ID:-}"
echo "Host:       $(hostname)"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:      $(date)"
echo "=========================================="

srun uv run --no-sync python -m purejaxql.mog_pqn_rnn_stabilized_craftax \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "SEED=${SEED}" \
  NUM_SEEDS=1 \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}" \
  "+alg=mog_pqn_rnn_stabilized_craftax"

echo "Completed"
echo "End:      $(date)"
echo "=========================================="
