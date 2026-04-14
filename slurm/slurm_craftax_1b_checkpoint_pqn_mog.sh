#!/bin/bash
#SBATCH --job-name=craftax-1b-ckpt-pqn-mog
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-9%10
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
# Craftax 1B env steps: **PQN-RNN** vs **MoG-PQN-RNN**, **5 seeds** each → **10** array tasks.
#
# W&B: tag ``Craftax-1B-checkpoint`` (via ``+alg.EXPERIMENT_TAG``). Algo tags come from trainers
# (``PQN_RNN`` / ``MOG_PQN_RNN``).
#
# Model weights: after training finishes, each job writes **one** final ``.safetensors`` per vmap seed
# under ``SAVE_PATH`` (``{alg}_{env}_seed{SEED}_vmap{i}.safetensors``) — the last optimizer state.
#
# Mapping: TID 0–9 → ALGO_IDX = TID/5, SEED_IDX = TID%5
#   TID 0..4   → PQN-RNN   seeds 0–4
#   TID 5..9   → MoG-PQN-RNN seeds 0–4
#
# Submit: sbatch slurm/slurm_craftax_1b_checkpoint_pqn_mog.sh
#
# Optional overrides:
#   export WANDB_PROJECT=Deep-CVI-Experiments
#   export SAVE_PATH=/path/to/ckpt/dir
#   export CRAFTAX_ENV_NAME=Craftax-Symbolic-v1
#

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax-1B-checkpoint}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"
SAVE_PATH="${SAVE_PATH:-${ROOT}/checkpoints/Craftax-1B-checkpoint}"
mkdir -p "${SAVE_PATH}"

TID="${SLURM_ARRAY_TASK_ID:?}"
ALGO_IDX=$((TID / 5))
SEED_IDX=$((TID % 5))

SEEDS=(34 76 118 160 192)
SEED="${SEEDS[$SEED_IDX]}"

echo "=========================================="
echo "Craftax 1B — PQN-RNN vs MoG-PQN-RNN (5 seeds each, final weights to SAVE_PATH)"
echo "=========================================="
echo "Task:       ${TID} (algo_idx ${ALGO_IDX}, seed_idx ${SEED_IDX})"
echo "Seed:       ${SEED}"
echo "Env:        ${CRAFTAX_ENV_NAME}"
echo "W&B tag:    ${EXPERIMENT_TAG}"
echo "SAVE_PATH:  ${SAVE_PATH}"
echo "Project:    ${WANDB_PROJECT}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:      $(date)"
echo "=========================================="

PY_MODULE=(
  purejaxql.pqn_rnn_craftax
  purejaxql.mog_pqn_rnn_craftax
)
HYDRA_ALG=(
  pqn_rnn_craftax
  mog_pqn_rnn_craftax
)

HYDRA_COMMON=(
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config
  "SEED=${SEED}"
  NUM_SEEDS=1
  "SAVE_PATH=${SAVE_PATH}"
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}"
)

case "${ALGO_IDX}" in
  0|1)
    srun uv run --no-sync python -m "${PY_MODULE[$ALGO_IDX]}" \
      "${HYDRA_COMMON[@]}" \
      "+alg=${HYDRA_ALG[$ALGO_IDX]}"
    ;;
  *)
    echo "Invalid ALGO_IDX=${ALGO_IDX} (expected 0 or 1)" >&2
    exit 1
    ;;
esac

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
