#!/bin/bash
#SBATCH --job-name=cqn-rnn-ddp-craftax
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#
# CQN-RNN distributed (pmap / DDP-style) on 2x L40S, single node.
# Submit: sbatch slurm/slurm_cqn_rnn_distributed_craftax.sh
#
# Optional env overrides:
#   export WANDB_EXPERIMENT_TAG=Craftax_CQN_RNN_DDP
#   export CRAFTAX_ENV_NAME=Craftax-Symbolic-v1
#   export SEED=0
#   export DDP_NUM_DEVICES=2

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax_CQN_RNN_DDP}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"
SEED="${SEED:-0}"
DDP_NUM_DEVICES="${DDP_NUM_DEVICES:-2}"

echo "=========================================="
echo "CQN-RNN distributed (Craftax, 2 GPU DDP)"
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

srun uv run --no-sync python -m purejaxql.cqn_rnn_distributed_craftax \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "SEED=${SEED}" \
  NUM_SEEDS=1 \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}" \
  "alg.DISTRIBUTED_NUM_DEVICES=${DDP_NUM_DEVICES}" \
  "+alg=cqn_rnn_distributed_craftax"

echo "Completed"
echo "End:      $(date)"
echo "=========================================="
