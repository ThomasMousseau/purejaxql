#!/bin/bash
#SBATCH --job-name=minatar-30m-pqn
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-11%12
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00

################################################################################
# MinAtar 30M: 4 envs × 3 algos (CTD, QTD, IQN only) = 12 Slurm tasks.
# MoG-CQN, CQN, and PQN are omitted (assume those runs already exist).
#
# Each task: **NUM_SEEDS=5** (vmap) — one process, one W&B run per (env, algo).
# Tag **multi_seed** + ``WANDB_LOG_ALL_SEEDS`` → logs ``seed_i/charts/episodic_return``.
#
# Algorithms: ctd_minatar, qtd_minatar, iqn_minatar
#
# W&B: EXPERIMENT_TAG=MinAtar_30M_PQN, algo tags identify each baseline.
#
# TID → env_idx = TID/3, algo_idx = TID%3
#
# Submit from repo root:  sbatch slurm/slurm_minatar_30m_pqn_compare.sh
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_30M_PQN"

TID="${SLURM_ARRAY_TASK_ID:?}"
ENV_IDX=$((TID / 3))
ALGO_IDX=$((TID % 3))

ENV_IDS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

PY_MODULES=(
    "purejaxql.ctd_minatar"
    "purejaxql.qtd_minatar"
    "purejaxql.iqn_minatar"
)
HYDRA_ALG=(
    "ctd_minatar"
    "qtd_minatar"
    "iqn_minatar"
)
WANDB_ALGO_TAGS=(
    "CTD"
    "QTD"
    "IQN"
)

ENV_ID="${ENV_IDS[$ENV_IDX]}"
PY_MODULE="${PY_MODULES[$ALGO_IDX]}"
ALG_GROUP="${HYDRA_ALG[$ALGO_IDX]}"
WANDB_TAG_ALGO="${WANDB_ALGO_TAGS[$ALGO_IDX]}"

RUN_NAME="${ENV_ID}__${WANDB_TAG_ALGO}__x5"

echo "=========================================="
echo "MinAtar 30M — 12 tasks (4 envs × 3 algos: CTD/QTD/IQN), NUM_SEEDS=5 vmap, W&B multi_seed"
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

ALGO_EXTRA=()

uv run --no-sync python -m "${PY_MODULE}" \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "+alg=${ALG_GROUP}" \
  "alg.ENV_NAME=${ENV_ID}" \
  "SEED=7" \
  "NUM_SEEDS=5" \
  "alg.WANDB_LOG_ALL_SEEDS=true" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "+alg.WANDB_EXTRA_TAGS=[\"${WANDB_TAG_ALGO}\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=30_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=30_000_000" \
  "${ALGO_EXTRA[@]}"

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
