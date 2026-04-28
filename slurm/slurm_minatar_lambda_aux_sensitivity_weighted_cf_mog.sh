#!/bin/bash
#SBATCH --job-name=minatar-20m-mog-weightedcf-lambda-aux
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-11%12
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

################################################################################
# MinAtar 20M MoG-PQN (weighted CF) TD(lambda) + AUX sensitivity:
# - 4 envs: Asterix / Breakout / Freeway / SpaceInvaders
# - 3 experiments:
#   Exp1: LAMBDA=0.0,  AUX=0.0
#   Exp2: LAMBDA=0.0,  AUX=1.0
#   Exp3: LAMBDA=0.65, AUX=1.0
#
# Total tasks: 3 experiments x 4 envs = 12 tasks (array 0..11).
#
# Always enforced:
# - alg.IS_DIVIDED_BY_SIGMA_SQUARED=True (weighted CF loss)
# - multi-seed setup (SEED=7, NUM_SEEDS=5)
#
# Submit: sbatch slurm/slurm_minatar_lambda_aux_sensitivity_weighted_cf_mog.sh
################################################################################

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"
export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_20M_Td_Lambda_WeightedCF_MoG"
TID="${SLURM_ARRAY_TASK_ID:?}"

ENV_IDS=(
  "Asterix-MinAtar"
  "Breakout-MinAtar"
  "Freeway-MinAtar"
  "SpaceInvaders-MinAtar"
)

build_task() {
  local exp_name="$1"
  local lambda="$2"
  local aux="$3"
  local e
  for e in "${!ENV_IDS[@]}"; do
    TASK_EXP_NAMES+=("${exp_name}")
    TASK_LAMBDAS+=("${lambda}")
    TASK_AUXES+=("${aux}")
    TASK_ENV_IDXS+=("${e}")
  done
}

declare -a TASK_EXP_NAMES=()
declare -a TASK_LAMBDAS=()
declare -a TASK_AUXES=()
declare -a TASK_ENV_IDXS=()

build_task "EXP1" "0.0" "0.0"
build_task "EXP2" "0.0" "1.0"
build_task "EXP3" "0.65" "1.0"

if (( TID < 0 || TID >= ${#TASK_EXP_NAMES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TID}; expected [0, $(( ${#TASK_EXP_NAMES[@]} - 1 ))]"
  exit 1
fi

EXP_NAME="${TASK_EXP_NAMES[$TID]}"
LAMBDA_VAL="${TASK_LAMBDAS[$TID]}"
AUX_VAL="${TASK_AUXES[$TID]}"
ENV_IDX="${TASK_ENV_IDXS[$TID]}"

ENV_ID="${ENV_IDS[$ENV_IDX]}"
PY_MODULE="purejaxql.mog_pqn_minatar"
ALG_GROUP="mog_pqn_minatar"
WANDB_TAG_ALGO="MoG"

LAMBDA_TAG="LAMBDA-${LAMBDA_VAL}"
AUX_TAG="AUX_MEAN_LOSS_WEIGHT-${AUX_VAL}"
CF_TAG="WEIGHTED_CF"
OMEGA_DIV_TAG="IS_DIVIDED_BY_OMEGA_SQUARED-True"
RUN_NAME="${ENV_ID}__${WANDB_TAG_ALGO}__${CF_TAG}__${EXP_NAME}__x5"

echo "=========================================="
echo "MinAtar 20M MoG weighted-CF TD(lambda)+AUX sensitivity"
echo "Task:        ${TID}"
echo "Experiment:  ${EXP_NAME}"
echo "Module:      ${PY_MODULE}"
echo "Hydra:       +alg=${ALG_GROUP}"
echo "Env:         ${ENV_ID}"
echo "Run name:    ${RUN_NAME}"
echo "W&B exp:     ${EXPERIMENT_TAG}"
echo "Tags:        ${WANDB_TAG_ALGO}, ${CF_TAG}, ${OMEGA_DIV_TAG}, ${LAMBDA_TAG}, ${AUX_TAG}, multi_seed"
echo "LAMBDA:      ${LAMBDA_VAL}"
echo "AUX weight:  ${AUX_VAL}"
echo "Weighted CF: True"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Host:        $(hostname)"
echo "GPU:         ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:       $(date)"
echo "=========================================="

uv run --no-sync python -m "${PY_MODULE}" \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "+alg=${ALG_GROUP}" \
  "alg.ENV_NAME=${ENV_ID}" \
  "SEED=7" \
  "NUM_SEEDS=5" \
  "alg.WANDB_LOG_ALL_SEEDS=true" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "+alg.WANDB_EXTRA_TAGS=[\"${WANDB_TAG_ALGO}\",\"${CF_TAG}\",\"${OMEGA_DIV_TAG}\",\"${LAMBDA_TAG}\",\"${AUX_TAG}\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=20_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=20_000_000" \
  "alg.IS_DIVIDED_BY_OMEGA_SQUARED=True" \
  "alg.LAMBDA=${LAMBDA_VAL}" \
  "alg.AUX_MEAN_LOSS_WEIGHT=${AUX_VAL}"

echo "Task ${TID} completed"
echo "End:         $(date)"
echo "=========================================="
