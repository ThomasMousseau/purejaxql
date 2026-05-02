#!/bin/bash
#SBATCH --job-name=craftax-1b-rnn-phi-td-compare
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-39%40
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=10:00:00

################################################################################
# Craftax 1B env steps: **8 Phi-TD families** × **5 seeds (0–4)** = **40 tasks** (array 0–39).
#
# Baselines (CTD-RNN, QTD-RNN, PQN-RNN) are **not** launched here; compare them in W&B plots
# via ``plot_craftax_phi_td_with_baselines`` using this tag plus your baseline experiment tag.
#
# Apples-to-apples (shared across phi YAMLs): env, rollout, LSTM trunk, optimizer, exploration.
#
# W&B: ``WANDB_EXPERIMENT_TAG`` (default Craftax-1B-phi-td-compare).
#
# Mapping: TID → ALGO_IDX = TID/5, SEED_IDX = TID%5 (ALGO_IDX 0–7 → phi family yaml).
#
# Submit: sbatch slurm/slurm_craftax_rnn_1b_phi_td_compare.sh
#
# Subset: set ACTIVE_ALGO_INDICES to a list of indices into HYDRA_ALG, and set
# ``#SBATCH --array=0-$(( ${#ACTIVE_ALGO_INDICES[@]} * NUM_SEEDS - 1 ))``.
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax-1B-phi-td-compare}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"
NUM_SEEDS=5
ACTIVE_IDX=$((TID / NUM_SEEDS))
SEED_IDX=$((TID % NUM_SEEDS))

HYDRA_ALG=(
  phi_td_rnn_craftax_categorical
  phi_td_rnn_craftax_quantile
  phi_td_rnn_craftax_dirac
  phi_td_rnn_craftax_mog
  phi_td_rnn_craftax_cauchy
  phi_td_rnn_craftax_gamma
  phi_td_rnn_craftax_laplace
  phi_td_rnn_craftax_logistic
)

ACTIVE_ALGO_INDICES=(0 1 2 3 4 5 6 7)
if (( ACTIVE_IDX < 0 || ACTIVE_IDX >= ${#ACTIVE_ALGO_INDICES[@]} )); then
  echo "Invalid ACTIVE_IDX=${ACTIVE_IDX} for TID=${TID} (expected 0..$((${#ACTIVE_ALGO_INDICES[@]} - 1)))" >&2
  exit 1
fi
ALGO_IDX="${ACTIVE_ALGO_INDICES[$ACTIVE_IDX]}"

SEEDS=(0 1 2 3 4)
if (( SEED_IDX < 0 || SEED_IDX >= ${#SEEDS[@]} )); then
  echo "Invalid SEED_IDX=${SEED_IDX} for TID=${TID}" >&2
  exit 1
fi
SEED="${SEEDS[$SEED_IDX]}"

echo "=========================================="
echo "Craftax 1B — Phi-TD families (RNN Craftax)"
echo "=========================================="
echo "Task:       ${TID} (active_idx ${ACTIVE_IDX}, algo_idx ${ALGO_IDX}, seed_idx ${SEED_IDX})"
echo "Seed:       ${SEED}"
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

srun uv run --no-sync python -m purejaxql.phi_td_pqn_rnn_craftax \
  "${HYDRA_COMMON[@]}" \
  "+alg=${HYDRA_ALG[$ALGO_IDX]}"

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
