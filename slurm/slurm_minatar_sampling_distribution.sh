#!/bin/bash
#SBATCH --job-name=minatar-mog-sampling-dist
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-11%12
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

################################################################################
# MinAtar sampling-distribution ablation for MoG-PQN:
# - 4 envs: Asterix / Breakout / Freeway / SpaceInvaders
# - 3 omega sampling distributions:
#   1) half_laplacian (default)
#   2) uniform
#   3) half_gaussian (mu=0, sigma=0.5)
#
# Total tasks: 4 * 3 = 12 (array 0..11)
# Submit: sbatch slurm/slurm_minatar_sampling_distribution.sh
################################################################################

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"
export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_MoG_Sampling_Distribution"
TID="${SLURM_ARRAY_TASK_ID:?}"

ENV_IDS=(
  "Asterix-MinAtar"
  "Breakout-MinAtar"
  "Freeway-MinAtar"
  "SpaceInvaders-MinAtar"
)

DISTRIBUTIONS=(
  "half_laplacian"
  "uniform"
  "half_gaussian"
)

DIST_TAGS=(
  "HALF_LAPLACIAN"
  "UNIFORM"
  "HALF_GAUSSIAN"
)

ENV_IDX=$((TID / 3))
DIST_IDX=$((TID % 3))

ENV_ID="${ENV_IDS[$ENV_IDX]}"
DIST="${DISTRIBUTIONS[$DIST_IDX]}"
DIST_TAG="${DIST_TAGS[$DIST_IDX]}"

RUN_NAME="${ENV_ID}__MoG__${DIST_TAG}__x5"

echo "=========================================="
echo "MinAtar MoG sampling distribution ablation"
echo "Task:         ${TID} (env ${ENV_IDX}, dist ${DIST_IDX})"
echo "Module:       purejaxql.mog_pqn_minatar"
echo "Hydra:        +alg=mog_pqn_minatar"
echo "Env:          ${ENV_ID}"
echo "Distribution: ${DIST}"
echo "Run name:     ${RUN_NAME}"
echo "W&B exp:      ${EXPERIMENT_TAG}"
echo "Tags:         MoG, ${DIST_TAG}, multi_seed"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:        $(date)"
echo "=========================================="

uv run --no-sync python -m purejaxql.mog_pqn_minatar \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "+alg=mog_pqn_minatar" \
  "alg.ENV_NAME=${ENV_ID}" \
  "SEED=7" \
  "NUM_SEEDS=5" \
  "alg.WANDB_LOG_ALL_SEEDS=true" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "+alg.WANDB_EXTRA_TAGS=[\"MoG\",\"${DIST_TAG}\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.OMEGA_SAMPLING_DISTRIBUTION=${DIST}" \
  "alg.OMEGA_HALF_GAUSSIAN_MEAN=0.0" \
  "alg.OMEGA_HALF_GAUSSIAN_STD=0.5" \
  "alg.TOTAL_TIMESTEPS=10_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=10_000_000" \

echo "Task ${TID} completed"
echo "End:          $(date)"
echo "=========================================="
