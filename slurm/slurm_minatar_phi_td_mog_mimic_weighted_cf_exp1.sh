#!/bin/bash
#SBATCH --job-name=minatar-phitd-mog-mimic-wcf-exp1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-3%4
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=1:00:00

set -euo pipefail

################################################################################
# φTD-MoG mimic of MoG-PQN Exp 1 (λ=0, AUX mean loss weight=0) from the weighted-CF sweep.
#
# Matches ``slurm/slurm_minatar_lambda_aux_sensitivity_weighted_cf_mog.sh`` Exp1 hyperparameters
# on ``purejaxql/phi_td_pqn_minatar.py`` so learning curves can be compared to W&B runs tagged
# ``MinAtar_20M_Td_Lambda_WeightedCF_MoG`` (MoG + LAMBDA-0.0 + AUX_MEAN_LOSS_WEIGHT-0.0).
#
# Critical alignment vs default ``phi_td_minatar_mog.yaml``:
# - ``M_PARTICLES=8`` (same mixture order as ``NUM_COMPONENTS`` in ``mog_pqn_minatar``)
# - ``IS_DIVIDED_BY_OMEGA_SQUARED=True``, ``OMEGA_SAMPLING_DISTRIBUTION=half_laplacian``
# - Same 20M steps and multi-seed protocol as the MoG Slurm script (SEED=7, NUM_SEEDS=5).
#
# Total tasks: 4 envs (array 0..3).
#
# Submit: sbatch slurm/slurm_minatar_phi_td_mog_mimic_weighted_cf_exp1.sh
################################################################################

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"
export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_20M_PhiTD_MoG_Mimic_WeightedCF"
TID="${SLURM_ARRAY_TASK_ID:?}"

ENV_IDS=(
  "Asterix-MinAtar"
  "Breakout-MinAtar"
  "Freeway-MinAtar"
  "SpaceInvaders-MinAtar"
)

if (( TID < 0 || TID >= ${#ENV_IDS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TID}; expected [0, $(( ${#ENV_IDS[@]} - 1 ))]"
  exit 1
fi

ENV_ID="${ENV_IDS[$TID]}"
PY_MODULE="purejaxql.phi_td_pqn_minatar"
ALG_GROUP="phi_td_minatar_mog"

CF_TAG="WEIGHTED_CF"
OMEGA_DIV_TAG="IS_DIVIDED_BY_OMEGA_SQUARED-True"
LAMBDA_TAG="LAMBDA-0.0"
AUX_TAG="AUX_MEAN_LOSS_WEIGHT-0.0"
RUN_NAME="${ENV_ID}__PhiTD-MoG-Mimic-WCF__EXP1__x5"

echo "=========================================="
echo "MinAtar 20M φTD-MoG mimic (weighted CF, Exp1-style tags)"
echo "Task:        ${TID}"
echo "Module:      ${PY_MODULE}"
echo "Hydra:       +alg=${ALG_GROUP}"
echo "Env:         ${ENV_ID}"
echo "Run name:    ${RUN_NAME}"
echo "W&B exp:     ${EXPERIMENT_TAG}"
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
  "+alg.WANDB_EXTRA_TAGS=[\"PhiTD-MoG\",\"${CF_TAG}\",\"${OMEGA_DIV_TAG}\",\"${LAMBDA_TAG}\",\"${AUX_TAG}\",\"EXP1_PHI_TD_MIMIC\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=20_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=20_000_000" \
  "alg.M_PARTICLES=8" \
  "alg.IS_DIVIDED_BY_OMEGA_SQUARED=True" \
  "alg.NUM_OMEGA_SAMPLES=128" \
  "alg.OMEGA_SAMPLING_DISTRIBUTION=half_laplacian" \
  "alg.OMEGA_MAX=1.0"

echo "Task ${TID} completed"
echo "End:         $(date)"
echo "=========================================="
