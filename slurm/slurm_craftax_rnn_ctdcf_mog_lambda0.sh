#!/bin/bash
#SBATCH --job-name=craftax-ctd-cf-mog-lambda0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-5%6
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00

################################################################################
# Craftax RNN — **CTD weighted CF** (`ctd_cf_rnn_craftax`) and **MoG-PQN-RNN**
# (same CF loss with 1/ω² as in alg yaml), both with **LAMBDA=0** and
# **AUX_MEAN_LOSS_WEIGHT=0** (pure distributional / CF term only for CTD; MoG uses
# only CF loss when aux weight is 0).
#
# 2 algorithms × 3 seeds = **6 tasks**
# Mapping: TID 0–5 → ALGO_IDX = TID/3, SEED_IDX = TID%3
#   TID 0,1,2   → CTD-RNN + weighted CF (`+alg=ctd_cf_rnn_craftax`)
#   TID 3,4,5   → MoG-RNN (`+alg=mog_pqn_rnn_craftax`)
#
# Submit: sbatch slurm/slurm_craftax_rnn_ctdcf_mog_lambda0.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax-RNN-CTD-MoG-WeightedCF}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"
ALGO_IDX=$((TID / 3))
SEED_IDX=$((TID % 3))

SEEDS=(0 1 2)
SEED="${SEEDS[$SEED_IDX]}"

echo "=========================================="
echo "Craftax RNN — CTD weighted CF & MoG (λ=0, aux=0)"
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
  purejaxql.ctd_rnn_craftax
  purejaxql.mog_pqn_rnn_craftax
)
HYDRA_ALG=(
  ctd_cf_rnn_craftax_weighted_cf
  mog_pqn_rnn_craftax
)

HYDRA_COMMON=(
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config
  "SEED=${SEED}"
  NUM_SEEDS=1
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}"
  "alg.LAMBDA=0.0"
  "alg.AUX_MEAN_LOSS_WEIGHT=0.0"
)

case "${ALGO_IDX}" in
  0|1)
    srun uv run --no-sync python -m "${PY_MODULE[$ALGO_IDX]}" \
      "${HYDRA_COMMON[@]}" \
      "+alg=${HYDRA_ALG[$ALGO_IDX]}"
    ;;
  *)
    echo "Invalid ALGO_IDX=${ALGO_IDX} (expected 0..1)" >&2
    exit 1
    ;;
esac

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
