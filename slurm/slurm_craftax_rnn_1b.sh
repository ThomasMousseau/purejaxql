#!/bin/bash
#SBATCH --job-name=craftax-1b-rnn-ctd-qtd
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-3%4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=24:00:00


################################################################################
# Craftax 1B env steps: active subset **CTD / QTD** × **2 seeds** = **4 tasks**
# Full 5-algorithm setup is preserved below; only ACTIVE_ALGO_INDICES controls launches.
#
# W&B: Craftax_1B_RNN_4algo + algorithm tags from purejaxql trainers.
#
# Mapping: TID 0–3 → ACTIVE_IDX = TID/2, SEED_IDX = TID%2
#   TID 0,1   → CTD-RNN   seeds 3,4
#   TID 2,3   → QTD-RNN   seeds 3,4

# Submit: sbatch slurm/slurm_craftax_rnn_1b.sh

################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax-1B-checkpoint}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"
ACTIVE_IDX=$((TID / 3))
SEED_IDX=$((TID % 3))

# Keep the full algorithm lists below; select which ones to launch here.
ACTIVE_ALGO_INDICES=(0 1) # CTD, QTD
ALGO_IDX="${ACTIVE_ALGO_INDICES[$ACTIVE_IDX]}"

SEEDS=(3 4) # 0 1 2 3 4
SEED="${SEEDS[$SEED_IDX]}"

echo "=========================================="
echo "Craftax 1B — CTD/QTD RNN (2 seeds each: 3 and 4)"
echo "=========================================="
echo "Task:       ${TID} (active_idx ${ACTIVE_IDX}, algo_idx ${ALGO_IDX}, seed_idx ${SEED_IDX})"
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
  purejaxql.iqn_rnn_craftax
  purejaxql.mog_pqn_rnn_craftax
  purejaxql.pqn_rnn_craftax
)
HYDRA_ALG=(
  ctd_rnn_craftax
  qtd_rnn_craftax
  iqn_rnn_craftax
  mog_pqn_rnn_craftax
  pqn_rnn_craftax
)

HYDRA_COMMON=(
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config
  "SEED=${SEED}"
  NUM_SEEDS=1
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}"
)

ALGO_EXTRA=("alg.LAMBDA=0.5")
if [[ "${ALGO_IDX}" -le 3 ]]; then
  # Distributional variants only: CTD/QTD/IQN/MoG.
  ALGO_EXTRA+=("alg.AUX_MEAN_LOSS_WEIGHT=1.0")
fi

case "${ALGO_IDX}" in
  0|1|2|3|4)
    srun uv run --no-sync python -m "${PY_MODULE[$ALGO_IDX]}" \
      "${HYDRA_COMMON[@]}" \
      "+alg=${HYDRA_ALG[$ALGO_IDX]}" \
      "${ALGO_EXTRA[@]}"
    ;;
  *)
    echo "Invalid ALGO_IDX=${ALGO_IDX} (expected 0..4)" >&2
    exit 1
    ;;
esac

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
