#!/bin/bash
#SBATCH --job-name=minatar-phi-families
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-19%20
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH --time=1:00:00

################################################################################
# MinAtar 10M — five algorithms × four envs = 20 Slurm tasks.
#
# Compare learning curves (same m = 51 atoms in CTD/QTD/Phi configs):
#   CTD  (categorical CE + TD(λ) aux), QTD (quantile Huber + TD(λ) aux),
#   Phi-TD with F_m (mixture of Diracs), F_{C,m} (categorical CF-only),
#   F_{Q,m} (quantile empirical CF-only).
#
# Phi-TD runs use ``purejaxql.phi_td_pqn_minatar`` (weighted CF Bellman only).
# CTD/QTD use ``distributional_pqn_minatar`` (standard losses).
#
# NUM_SEEDS=5 vmap per task; W&B multi_seed tagging mirrors slurm_minatar.sh.
#
# TID → env_idx = TID/5, algo_idx = TID%5
#
# Submit from repo root:  sbatch slurm/slurm_minatar_phi_families.sh
################################################################################

#SBATCH --gres=gpu:l40s:1
#SBATCH --gres=gpu:rtx8000:1


set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_10M_PhiTD_Families_v2"

TID="${SLURM_ARRAY_TASK_ID:?}"
ENV_IDX=$((TID / 5))
ALGO_IDX=$((TID % 5))

ENV_IDS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

PY_MODULES=(
    "purejaxql.ctd_minatar"
    "purejaxql.qtd_minatar"
    "purejaxql.phi_td_pqn_minatar"
    "purejaxql.phi_td_pqn_minatar"
    "purejaxql.phi_td_pqn_minatar"
)
HYDRA_ALG=(
    "ctd_minatar"
    "qtd_minatar"
    "phi_td_minatar_dirac"
    "phi_td_minatar_categorical"
    "phi_td_minatar_quantile"
)
WANDB_ALGO_TAGS=(
    "CTD"
    "QTD"
    "PhiTD-Fm"
    "PhiTD-FCm"
    "PhiTD-FQm"
)

ENV_ID="${ENV_IDS[$ENV_IDX]}"
PY_MODULE="${PY_MODULES[$ALGO_IDX]}"
ALG_GROUP="${HYDRA_ALG[$ALGO_IDX]}"
WANDB_TAG_ALGO="${WANDB_ALGO_TAGS[$ALGO_IDX]}"

RUN_NAME="${ENV_ID}__${WANDB_TAG_ALGO}__s5"

echo "=========================================="
echo "MinAtar 10M — Phi-TD family comparison (5 algos × 4 envs)"
echo "=========================================="
echo "Task:     ${TID} (env ${ENV_IDX}, algo ${ALGO_IDX})"
echo "Module:   ${PY_MODULE}"
echo "Hydra:    +alg=${ALG_GROUP}"
echo "Env:      ${ENV_ID}"
echo "Run name: ${RUN_NAME}"
echo "W&B exp:  ${EXPERIMENT_TAG}"
echo "Tags:     ${WANDB_TAG_ALGO}, multi_seed"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Host:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:    $(date)"
echo "=========================================="

uv run --no-sync python -m "${PY_MODULE}" \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "+alg=${ALG_GROUP}" \
  "alg.ENV_NAME=${ENV_ID}" \
  "SEED=7" \
  "NUM_SEEDS=5" \
  "alg.WANDB_LOG_ALL_SEEDS=true" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "+alg.WANDB_EXTRA_TAGS=[\"${WANDB_TAG_ALGO}\",\"multi_seed\",\"phi_family_compare\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=10_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=10_000_000" \

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
