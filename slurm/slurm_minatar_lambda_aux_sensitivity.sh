#!/bin/bash
#SBATCH --job-name=minatar-20m-td-lambda-aux
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-55%20
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

################################################################################
# MinAtar 20M TD(lambda) + AUX sensitivity:
# - 4 envs: Asterix / Breakout / Freeway / SpaceInvaders
# - 5 algos: CTD / QTD / IQN / MoG / PQN
# - 3 experiments:
#   Exp1: LAMBDA=0.0,  AUX=0.0   (all 5 algos)
#   Exp2: LAMBDA=0.0,  AUX=1.0   (CTD/QTD/IQN/MoG only; PQN reused from Exp1)
#   Exp3: LAMBDA=0.65, AUX=1.0   (all 5 algos)
#
# Total tasks: 20 + 16 + 20 = 56 tasks (array 0..55).
#
# W&B tags on each run:
# - MinAtar_20M_Td_Lambda
# - LAMBDA-X.XX
# - AUX_MEAN_LOSS_WEIGHT-X.X
# - algo tag + multi_seed

# Submit: sbatch slurm/slurm_minatar_lambda_aux_sensitivity.sh
################################################################################

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"
export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_20M_Td_Lambda"
TID="${SLURM_ARRAY_TASK_ID:?}"

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
  "purejaxql.mog_pqn_minatar"
  "purejaxql.pqn_minatar"
)
HYDRA_ALG=(
  "ctd_minatar"
  "qtd_minatar"
  "iqn_minatar"
  "mog_pqn_minatar"
  "pqn_minatar"
)
WANDB_ALGO_TAGS=(
  "CTD"
  "QTD"
  "IQN"
  "MoG"
  "PQN"
)

build_task() {
  local exp_name="$1"
  local lambda="$2"
  local aux="$3"
  local algo_count="$4"
  local e a
  for e in "${!ENV_IDS[@]}"; do
    for ((a=0; a<algo_count; a++)); do
      TASK_EXP_NAMES+=("${exp_name}")
      TASK_LAMBDAS+=("${lambda}")
      TASK_AUXES+=("${aux}")
      TASK_ENV_IDXS+=("${e}")
      TASK_ALGO_IDXS+=("${a}")
    done
  done
}

declare -a TASK_EXP_NAMES=()
declare -a TASK_LAMBDAS=()
declare -a TASK_AUXES=()
declare -a TASK_ENV_IDXS=()
declare -a TASK_ALGO_IDXS=()

# Exp 1: all 5 algos
build_task "EXP1" "0.0" "0.0" 5
# Exp 2: skip PQN (first 4 algos)
build_task "EXP2" "0.0" "1.0" 4
# Exp 3: all 5 algos
build_task "EXP3" "0.65" "1.0" 5

if (( TID < 0 || TID >= ${#TASK_EXP_NAMES[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TID}; expected [0, $(( ${#TASK_EXP_NAMES[@]} - 1 ))]"
  exit 1
fi

EXP_NAME="${TASK_EXP_NAMES[$TID]}"
LAMBDA_VAL="${TASK_LAMBDAS[$TID]}"
AUX_VAL="${TASK_AUXES[$TID]}"
ENV_IDX="${TASK_ENV_IDXS[$TID]}"
ALGO_IDX="${TASK_ALGO_IDXS[$TID]}"

ENV_ID="${ENV_IDS[$ENV_IDX]}"
PY_MODULE="${PY_MODULES[$ALGO_IDX]}"
ALG_GROUP="${HYDRA_ALG[$ALGO_IDX]}"
WANDB_TAG_ALGO="${WANDB_ALGO_TAGS[$ALGO_IDX]}"

LAMBDA_TAG="LAMBDA-${LAMBDA_VAL}"
AUX_TAG="AUX_MEAN_LOSS_WEIGHT-${AUX_VAL}"
RUN_NAME="${ENV_ID}__${WANDB_TAG_ALGO}__${EXP_NAME}__x5"

ALGO_EXTRA=()
# AUX_MEAN_LOSS_WEIGHT only applies to non-PQN algos.
if [[ "${WANDB_TAG_ALGO}" != "PQN" ]]; then
  ALGO_EXTRA+=("alg.AUX_MEAN_LOSS_WEIGHT=${AUX_VAL}")
fi

echo "=========================================="
echo "MinAtar 20M TD(lambda)+AUX sensitivity"
echo "Task:        ${TID}"
echo "Experiment:  ${EXP_NAME}"
echo "Module:      ${PY_MODULE}"
echo "Hydra:       +alg=${ALG_GROUP}"
echo "Env:         ${ENV_ID}"
echo "Run name:    ${RUN_NAME}"
echo "W&B exp:     ${EXPERIMENT_TAG}"
echo "Tags:        ${WANDB_TAG_ALGO}, ${LAMBDA_TAG}, ${AUX_TAG}, multi_seed"
echo "LAMBDA:      ${LAMBDA_VAL}"
echo "AUX weight:  ${AUX_VAL} (not passed to PQN)"
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
  "+alg.WANDB_EXTRA_TAGS=[\"${WANDB_TAG_ALGO}\",\"${LAMBDA_TAG}\",\"${AUX_TAG}\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=20_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=20_000_000" \
  "alg.LAMBDA=${LAMBDA_VAL}" \
  "${ALGO_EXTRA[@]}"

echo "Task ${TID} completed"
echo "End:         $(date)"
echo "=========================================="
