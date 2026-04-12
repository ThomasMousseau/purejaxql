#!/bin/bash
#SBATCH --job-name=cqn-rnn-craftax
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Single run: CQN-RNN (Craftax). See cqn_rnn_craftax.py
#
# GPU: L40S (48GB). Job 9244859 OOM’d on A100 40GB (~43GiB single alloc); L40S fits the default batch.
#
# Submit: sbatch slurm/slurm_cqn_rnn_craftax.sh
#
# Optional env overrides (same idea as the 2-algo script):
#   export WANDB_EXPERIMENT_TAG=MyTag
#   export CRAFTAX_ENV_NAME=Craftax-Symbolic-v1
#   export SEED=0
#   export NUM_ENVS=768   # if you still hit OOM, lower parallel envs (default 1024)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Helps fragmentation on long JAX runs (see JAX log hints on OOM)
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax_CQN_RNN}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"
SEED="${SEED:-0}"

echo "=========================================="
echo "CQN-RNN (Craftax)"
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

srun uv run --no-sync python -m purejaxql.cqn_rnn_craftax \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "SEED=${SEED}" \
  NUM_SEEDS=1 \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}" \
  "+alg=cqn_rnn_craftax"

echo "Completed"
echo "End:      $(date)"
echo "=========================================="
