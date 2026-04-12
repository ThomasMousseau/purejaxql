#!/bin/bash
#SBATCH --job-name=craftax-1b-rnn-2algo
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1
#SBATCH --array=0-5%6
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
# GPU: use --gpus-per-task (below). Do not also set a conflicting --gres=gpu:… line;
# "gpu:48gb:1" is not valid GRES syntax on many sites. If your cluster requires GRES
# explicitly, try e.g. `#SBATCH --gres=gpu:1` *instead of* --gpus-per-task, per local docs.


################################################################################
# Craftax 1B env steps: **2 algos** (PQN-RNN / MoG-PQN-RNN) × **3 seeds** = **6 tasks**
#
# W&B: Craftax_1B_RNN_3algo + PQN_RNN | MOG_PQN_RNN  (tags from purejaxql trainers)
#
# Mapping: TID 0–5 → ALGO_IDX = TID/3, SEED_IDX = TID%3
#   TID 0,1,2   → PQN-RNN   seeds 0,1,2
#   TID 3,4,5   → MoG-PQN-RNN seeds 0,1,2
#
# Memory (--mem): 1024 envs + LSTM; default 64G (edit if your cluster differs).
#
# Array throttle: %6 = up to 6 concurrent jobs. Use e.g. %2 to cap GPUs.
#
# Submit: sbatch slurm/slurm_craftax_1b_rnn_2algo.sh
#
# NUM_SEEDS=1: one PRNG seed per task; the three seeds come from this array, not vmap.
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax_1B_RNN_3algo}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"
ALGO_IDX=$((TID / 3))
SEED_IDX=$((TID % 3))

SEEDS=(0 1 2)
SEED="${SEEDS[$SEED_IDX]}"

echo "=========================================="
echo "Craftax 1B — PQN-RNN vs MoG-PQN-RNN (3 seeds each)"
echo "=========================================="
echo "Task:       ${TID} (algo_idx ${ALGO_IDX}, seed_idx ${SEED_IDX})"
echo "Seed:       ${SEED}"
echo "Env:        ${CRAFTAX_ENV_NAME}"
echo "W&B tag:    ${EXPERIMENT_TAG}"
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
