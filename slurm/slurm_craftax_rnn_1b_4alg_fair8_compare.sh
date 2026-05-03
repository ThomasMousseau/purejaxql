#!/bin/bash
#SBATCH --job-name=craftax-1b-rnn-4alg-fair8
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-19%20
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=8:00:00

################################################################################
# Craftax 1B env steps: **4 algorithms** × **5 seeds (0–4)** = **20 tasks** (array 0–19).
#
# Apples-to-apples (yaml-aligned): **8** categorical atoms / quantiles / Phi ``M_PARTICLES``;
# CTD/QTD use **LAMBDA=0** and **AUX_MEAN_LOSS_WEIGHT=0** (also set in ``ctd_rnn_craftax`` /
# ``qtd_rnn_craftax``; overrides below keep runs explicit). Shared env, rollout, LSTM trunk,
# optimizer, exploration match across these Craftax RNN configs.
#
#   ALGO_IDX 0 → CTD-RNN  (``purejaxql.ctd_rnn_craftax`` + ``+alg=ctd_rnn_craftax``)
#   ALGO_IDX 1 → QTD-RNN  (``purejaxql.qtd_rnn_craftax`` + ``+alg=qtd_rnn_craftax``)
#   ALGO_IDX 2 → Phi-TD categorical (``purejaxql.phi_td_pqn_rnn_craftax`` + ``+alg=phi_td_rnn_craftax_categorical``)
#   ALGO_IDX 3 → Phi-TD quantile    (``purejaxql.phi_td_pqn_rnn_craftax`` + ``+alg=phi_td_rnn_craftax_quantile``)
#
# Mapping: TID → ALGO_IDX = TID/5, SEED_IDX = TID%5.
#
# W&B: ``WANDB_EXPERIMENT_TAG`` (default ``Craftax-1B-RNN-fair8-4alg``; must be ≤64 chars).
#
# Submit: sbatch slurm/slurm_craftax_rnn_1b_4alg_fair8_compare.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax-1B-RNN-fair8-4alg}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"
NUM_SEEDS=5
ALGO_IDX=$((TID / NUM_SEEDS))
SEED_IDX=$((TID % NUM_SEEDS))

SEEDS=(0 1 2 3 4)
if (( SEED_IDX < 0 || SEED_IDX >= ${#SEEDS[@]} )); then
  echo "Invalid SEED_IDX=${SEED_IDX} for TID=${TID}" >&2
  exit 1
fi
SEED="${SEEDS[$SEED_IDX]}"

PY_MODULE=(
  purejaxql.ctd_rnn_craftax
  purejaxql.qtd_rnn_craftax
  purejaxql.phi_td_pqn_rnn_craftax
  purejaxql.phi_td_pqn_rnn_craftax
)
HYDRA_ALG=(
  ctd_rnn_craftax
  qtd_rnn_craftax
  phi_td_rnn_craftax_categorical
  phi_td_rnn_craftax_quantile
)

if (( ALGO_IDX < 0 || ALGO_IDX >= ${#HYDRA_ALG[@]} )); then
  echo "Invalid ALGO_IDX=${ALGO_IDX} for TID=${TID} (expected 0..$((${#HYDRA_ALG[@]} - 1)))" >&2
  exit 1
fi

echo "=========================================="
echo "Craftax 1B — CTD / QTD / Phi-TD cat & qt (RNN, fair-8)"
echo "=========================================="
echo "Task:       ${TID} (algo_idx ${ALGO_IDX}, seed_idx ${SEED_IDX})"
echo "Seed:       ${SEED}"
echo "Module:     ${PY_MODULE[$ALGO_IDX]}"
echo "Alg yaml:   ${HYDRA_ALG[$ALGO_IDX]}"
echo "Env:        ${CRAFTAX_ENV_NAME}"
echo "W&B tag:    ${EXPERIMENT_TAG}"
echo "Project:    ${WANDB_PROJECT}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:      $(date)"
echo "=========================================="

HYDRA_COMMON=(
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config
  "SEED=${SEED}"
  NUM_SEEDS=1
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}"
)

if (( ALGO_IDX == 0 || ALGO_IDX == 1 )); then
  HYDRA_COMMON+=("alg.LAMBDA=0.0" "alg.AUX_MEAN_LOSS_WEIGHT=0.0")
fi

srun uv run --no-sync python -m "${PY_MODULE[$ALGO_IDX]}" \
  "${HYDRA_COMMON[@]}" \
  "+alg=${HYDRA_ALG[$ALGO_IDX]}"

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
