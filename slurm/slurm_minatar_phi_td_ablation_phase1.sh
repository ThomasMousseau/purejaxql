#!/bin/bash
#SBATCH --job-name=minatar-phitd-ablation-p1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-15%16
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

################################################################################
# φTD ablation Phase 1 — Gaussian (MoG) one-factor sweeps on Breakout / SpaceInvaders.
#
# Hold: OMEGA_SAMPLING_DISTRIBUTION=pareto_1, FAMILY=mog, NUM_SEEDS=5 (vmap).
#
# Axis A (TID 0..7):  NUM_OMEGA_SAMPLES ∈ {32,64,128,256}, M_PARTICLES=51
# Axis B (TID 8..15): M_PARTICLES ∈ {10,20,51,100},       NUM_OMEGA_SAMPLES=128
#
# Layout: TID = axis*8 + env*4 + value_idx
#   env_idx  = (TID % 8) / 4   → 0 Breakout, 1 SpaceInvaders
#   val_idx  = TID % 4
#   axis     = TID / 8         → 0 omega-count, 1 components
#
# Total: 16 tasks. After this, pick best NUM_OMEGA_SAMPLES and M_PARTICLES,
# then launch Phase 2.
#
# Submit: sbatch slurm/slurm_minatar_phi_td_ablation_phase1.sh
################################################################################

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"
export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_PhiTD_Ablation_Phase1"
TID="${SLURM_ARRAY_TASK_ID:?}"

ENV_IDS=(
  "Breakout-MinAtar"
  "SpaceInvaders-MinAtar"
)

OMEGA_COUNTS=(32 64 128 256)
COMPONENT_COUNTS=(10 20 51 100)

AXIS=$((TID / 8))
REST=$((TID % 8))
ENV_IDX=$((REST / 4))
VAL_IDX=$((REST % 4))

ENV_ID="${ENV_IDS[$ENV_IDX]}"
PY_MODULE="purejaxql.phi_td_pqn_minatar"
ALG_GROUP="phi_td_minatar_mog"

# Defaults for the held factor (yaml defaults).
NUM_OMEGA=128
M_PARTICLES=51

if (( AXIS == 0 )); then
  NUM_OMEGA="${OMEGA_COUNTS[$VAL_IDX]}"
  AXIS_TAG="N_OMEGA-${NUM_OMEGA}"
  AXIS_NAME="omega_count"
else
  M_PARTICLES="${COMPONENT_COUNTS[$VAL_IDX]}"
  AXIS_TAG="M-${M_PARTICLES}"
  AXIS_NAME="components"
fi

RUN_NAME="${ENV_ID}__PhiTD-MoG__${AXIS_TAG}__x5"

echo "=========================================="
echo "φTD Ablation Phase 1 (Gaussian / one-factor)"
echo "Task:         ${TID} (axis=${AXIS_NAME}, env=${ENV_IDX}, val=${VAL_IDX})"
echo "Module:       ${PY_MODULE}"
echo "Hydra:        +alg=${ALG_GROUP}"
echo "Env:          ${ENV_ID}"
echo "NUM_OMEGA:    ${NUM_OMEGA}"
echo "M_PARTICLES:  ${M_PARTICLES}"
echo "Sampling:     pareto_1 (held)"
echo "Family:       mog (held)"
echo "NUM_SEEDS:    5 (vmap; 10 seeds risk OOM)"
echo "Run name:     ${RUN_NAME}"
echo "W&B exp:      ${EXPERIMENT_TAG}"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:        $(date)"
echo "=========================================="

uv run --no-sync python -m "${PY_MODULE}" \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "+alg=${ALG_GROUP}" \
  "alg.ENV_NAME=${ENV_ID}" \
  "SEED=7" \
  "NUM_SEEDS=5" \
  "alg.WANDB_LOG_ALL_SEEDS=true" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "+alg.WANDB_EXTRA_TAGS=[\"PhiTD-MoG\",\"${AXIS_TAG}\",\"ablation_phase1\",\"${AXIS_NAME}\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=10_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=10_000_000" \
  "alg.NUM_OMEGA_SAMPLES=${NUM_OMEGA}" \
  "alg.M_PARTICLES=${M_PARTICLES}" \
  "alg.OMEGA_SAMPLING_DISTRIBUTION=pareto_1" \
  "alg.OMEGA_MIN=0.01" \
  "alg.OMEGA_MAX=1.0" \
  "alg.IS_DIVIDED_BY_OMEGA_SQUARED=False"

echo "Task ${TID} completed"
echo "End:          $(date)"
echo "=========================================="
