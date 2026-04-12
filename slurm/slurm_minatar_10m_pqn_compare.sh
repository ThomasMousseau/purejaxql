#!/bin/bash
#SBATCH --job-name=minatar-10m-pqn
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0,3,4,5,7,8,11,12,15%9
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00

################################################################################
# MinAtar 10M: 4 envs × 4 algos = 16 Slurm tasks.
#
# Each task: **NUM_SEEDS=5** (vmap) — one process, one W&B run per (env, algo).
# Tag **multi_seed** + ``WANDB_LOG_ALL_SEEDS`` → logs ``seed_i/charts/episodic_return``.
# Plots: ``old_cvi/plot_wandb_minatar.py`` / ``plot_wandb_craftax_rnn_compare.py`` use
# :func:`curves_from_wandb_run` to expand those into mean ± std (same as 5 separate runs).
#
# Algorithms:
#   - mog_pqn_minatar_stabilized (Double DQN + Polyak TAU)
#   - mog_pqn_minatar (PQN shell; MoG CF loss)
#   - cqn_minatar
#   - pqn_minatar
#
# W&B tags: EXPERIMENT_TAG=MinAtar_10M_PQN, WANDB_EXTRA_TAGS=[algo, multi_seed]
#
# TID → env_idx = TID/4, algo_idx = TID%4
#
# Submit from repo root:  sbatch slurm/slurm_minatar_10m_pqn_compare.sh
#
# GPU: ``--gres=gpu:l40s:1``. Array below reruns only tasks that failed in job 9243202
# (already succeeded: 1,2,6,9,10,13,14). For a full 16-task sweep use ``--array=0-15%16``.
################################################################################

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Reduce cuDNN "No valid config" failures with vmap+Conv (MoG vanilla on some envs).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0"

EXPERIMENT_TAG="MinAtar_10M_PQN"

TID="${SLURM_ARRAY_TASK_ID:?}"
ENV_IDX=$((TID / 4))
ALGO_IDX=$((TID % 4))

ENV_IDS=(
    "Asterix-MinAtar"
    "Breakout-MinAtar"
    "Freeway-MinAtar"
    "SpaceInvaders-MinAtar"
)

PY_MODULES=(
    "purejaxql.mog_pqn_minatar_stabilized"
    "purejaxql.mog_pqn_minatar"
    "purejaxql.cqn_minatar"
    "purejaxql.pqn_minatar"
)
HYDRA_ALG=(
    "mog_pqn_minatar_stabilized"
    "mog_pqn_minatar"
    "cqn_minatar"
    "pqn_minatar"
)
WANDB_ALGO_TAGS=(
    "MoG-PQN-stab"
    "MoG-PQN"
    "CQN"
    "PQN"
)

ENV_ID="${ENV_IDS[$ENV_IDX]}"
PY_MODULE="${PY_MODULES[$ALGO_IDX]}"
ALG_GROUP="${HYDRA_ALG[$ALGO_IDX]}"
WANDB_TAG_ALGO="${WANDB_ALGO_TAGS[$ALGO_IDX]}"

# One run name per (env, algo) — no per-seed suffix (seeds live under seed_i/* in W&B)
RUN_NAME="${ENV_ID}__${WANDB_TAG_ALGO}__x5"

echo "=========================================="
echo "MinAtar 10M PQN — 16 tasks (4 envs × 4 algos), NUM_SEEDS=5 vmap, W&B multi_seed"
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
if [[ "${ALGO_IDX}" -eq 2 ]]; then
  # CQN vmaps 5 seeds; default NUM_ENVS=128 can OOM on a single GPU.
  ALGO_EXTRA+=(++alg.NUM_ENVS=64 ++alg.TEST_NUM_ENVS=64)
fi
if [[ "${ALGO_IDX}" -eq 3 ]]; then
  ALGO_EXTRA+=(++alg.ALG_NAME=pqn)
fi

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
  "${ALGO_EXTRA[@]}"

echo "Task ${TID} completed"
echo "End:      $(date)"
echo "=========================================="
