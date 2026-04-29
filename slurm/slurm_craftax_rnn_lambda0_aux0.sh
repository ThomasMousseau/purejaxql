#!/bin/bash
#SBATCH --job-name=craftax-rnn-lambda0-aux0
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-7%8
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00

################################################################################
# Craftax RNN — **CTD** (`ctd_rnn_craftax`) and **QTD** (`qtd_rnn_craftax`) and **MoG-PQN-RNN** (`mog_pqn_rnn_craftax`) and **PQN-RNN** (`pqn_rnn_craftax`)
# all with **LAMBDA=0** and **AUX_MEAN_LOSS_WEIGHT=0** (pure distributional / CF term only for CTD and QTD; MoG and PQN use only CF loss when aux weight is 0).
#
# 4 algorithms × 2 seeds = **8 tasks**
# Mapping: TID 0–7 → ALGO_IDX = TID/2, SEED_IDX = TID%2
#   TID 0,1   → CTD-RNN (`+alg=ctd_rnn_craftax`)
#   TID 2,3   → QTD-RNN (`+alg=qtd_rnn_craftax`)
#   TID 4,5   → MoG-RNN (`+alg=mog_pqn_rnn_craftax`)
#   TID 6,7   → PQN-RNN (`+alg=pqn_rnn_craftax`)
#
# Submit: sbatch slurm/slurm_craftax_rnn_lambda0_aux0.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax-1B-5ALG}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"
ALGO_IDX=$((TID / 2))
SEED_IDX=$((TID % 2))

SEEDS=(3 4) # 0 1 2 3 4
SEED="${SEEDS[$SEED_IDX]}"

echo "=========================================="
echo "Craftax RNN — CTD & QTD & MoG & PQN (λ=0, aux=0)"
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
  purejaxql.qtd_rnn_craftax
  purejaxql.mog_pqn_rnn_craftax
  purejaxql.pqn_rnn_craftax

)
HYDRA_ALG=(
  ctd_rnn_craftax
  qtd_rnn_craftax
  mog_pqn_rnn_craftax
  pqn_rnn_craftax
)

ALGO_EXTRA=("alg.LAMBDA=0.0")
if [[ "${HYDRA_ALG[$ALGO_IDX]}" != "pqn_rnn_craftax" ]]; then
  # Distributional variants only: CTD/QTD/MoG (never plain PQN-RNN)
  ALGO_EXTRA+=("alg.AUX_MEAN_LOSS_WEIGHT=0.0")
fi

HYDRA_COMMON=(
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config
  "SEED=${SEED}"
  NUM_SEEDS=1
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}"
  "${ALGO_EXTRA[@]}"
)

case "${ALGO_IDX}" in
  0|1|2|3)
    srun uv run --no-sync python -m "${PY_MODULE[$ALGO_IDX]}" \
      "${HYDRA_COMMON[@]}" \
      "+alg=${HYDRA_ALG[$ALGO_IDX]}"
    ;;
  *)
    echo "Invalid ALGO_IDX=${ALGO_IDX} (expected 0..3)" >&2
    exit 1
    ;;
esac

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
