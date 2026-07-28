#!/bin/bash
#SBATCH --job-name=minatar-phitd-ablation-p2
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --array=0-17%18
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH --time=2:00:00

set -euo pipefail

################################################################################
# φTD ablation Phase 2 — frequency sampling × distribution family grid.
#
# Requires Phase-1 winners for NUM_OMEGA_SAMPLES and M_PARTICLES.
# Set them via env when submitting (defaults match yaml / Phase-1 held factors):
#
#   BEST_NUM_OMEGA=128 BEST_M_PARTICLES=51 \
#     sbatch slurm/slurm_minatar_phi_td_ablation_phase2.sh
#
# Optional per-env winners (override shared defaults when set):
#   BEST_NUM_OMEGA_BREAKOUT / BEST_M_PARTICLES_BREAKOUT
#   BEST_NUM_OMEGA_SPACEINVADERS / BEST_M_PARTICLES_SPACEINVADERS
#
# Grid (TID 0..17):
#   env_idx  = TID / 9          → 0 Breakout, 1 SpaceInvaders
#   combo    = TID % 9          → sampling_idx = combo/3, family_idx = combo%3
#
# Sampling: pareto_1 | half_laplacian (Exponential) | uniform
# Family:   mog (Gaussian) | categorical | quantile
#
# Total: 3 × 3 × 2 = 18 tasks. NUM_SEEDS=5 (vmap; 10 seeds risk OOM).
################################################################################

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"
export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="MinAtar_PhiTD_Ablation_Phase2"
TID="${SLURM_ARRAY_TASK_ID:?}"

# Shared Phase-1 winners (override at sbatch time).
BEST_NUM_OMEGA="${BEST_NUM_OMEGA:-128}"
BEST_M_PARTICLES="${BEST_M_PARTICLES:-51}"

ENV_IDS=(
  "Breakout-MinAtar"
  "SpaceInvaders-MinAtar"
)

SAMPLINGS=(
  "pareto_1"
  "half_laplacian"
  "uniform"
)
SAMPLING_TAGS=(
  "PARETO_1"
  "EXPONENTIAL"
  "UNIFORM"
)

FAMILIES=(
  "mog"
  "categorical"
  "quantile"
)
FAMILY_ALGS=(
  "phi_td_minatar_mog"
  "phi_td_minatar_categorical"
  "phi_td_minatar_quantile"
)
FAMILY_TAGS=(
  "PhiTD-MoG"
  "PhiTD-FCm"
  "PhiTD-FQm"
)

ENV_IDX=$((TID / 9))
COMBO=$((TID % 9))
SAMP_IDX=$((COMBO / 3))
FAM_IDX=$((COMBO % 3))

ENV_ID="${ENV_IDS[$ENV_IDX]}"
DIST="${SAMPLINGS[$SAMP_IDX]}"
DIST_TAG="${SAMPLING_TAGS[$SAMP_IDX]}"
ALG_GROUP="${FAMILY_ALGS[$FAM_IDX]}"
FAM_TAG="${FAMILY_TAGS[$FAM_IDX]}"
PY_MODULE="purejaxql.phi_td_pqn_minatar"

# Per-env overrides if provided.
if [[ "${ENV_ID}" == "Breakout-MinAtar" ]]; then
  NUM_OMEGA="${BEST_NUM_OMEGA_BREAKOUT:-${BEST_NUM_OMEGA}}"
  M_PARTICLES="${BEST_M_PARTICLES_BREAKOUT:-${BEST_M_PARTICLES}}"
else
  NUM_OMEGA="${BEST_NUM_OMEGA_SPACEINVADERS:-${BEST_NUM_OMEGA}}"
  M_PARTICLES="${BEST_M_PARTICLES_SPACEINVADERS:-${BEST_M_PARTICLES}}"
fi

# Unique combo tag so W&B plots can group without colliding on family+sampling tags.
COMBO_TAG="ABL2-${FAM_TAG}-${DIST_TAG}"
RUN_NAME="${ENV_ID}__${FAM_TAG}__${DIST_TAG}__N${NUM_OMEGA}_M${M_PARTICLES}__x5"

echo "=========================================="
echo "φTD Ablation Phase 2 (sampling × family)"
echo "Task:         ${TID} (env=${ENV_IDX}, samp=${SAMP_IDX}, fam=${FAM_IDX})"
echo "Module:       ${PY_MODULE}"
echo "Hydra:        +alg=${ALG_GROUP}"
echo "Env:          ${ENV_ID}"
echo "Family:       ${FAM_TAG}"
echo "Sampling:     ${DIST} (${DIST_TAG})"
echo "Combo tag:    ${COMBO_TAG}"
echo "NUM_OMEGA:    ${NUM_OMEGA}  (Phase-1 best)"
echo "M_PARTICLES:  ${M_PARTICLES}  (Phase-1 best)"
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
  "+alg.WANDB_EXTRA_TAGS=[\"${FAM_TAG}\",\"${DIST_TAG}\",\"${COMBO_TAG}\",\"N_OMEGA-${NUM_OMEGA}\",\"M-${M_PARTICLES}\",\"ablation_phase2\",\"multi_seed\"]" \
  "+alg.NAME=${RUN_NAME}" \
  "alg.TOTAL_TIMESTEPS=10_000_000" \
  "alg.TOTAL_TIMESTEPS_DECAY=10_000_000" \
  "alg.NUM_OMEGA_SAMPLES=${NUM_OMEGA}" \
  "alg.M_PARTICLES=${M_PARTICLES}" \
  "alg.OMEGA_SAMPLING_DISTRIBUTION=${DIST}" \
  "alg.OMEGA_MIN=0.01" \
  "alg.OMEGA_MAX=1.0" \
  "alg.IS_DIVIDED_BY_OMEGA_SQUARED=False"

echo "Task ${TID} completed"
echo "End:          $(date)"
echo "=========================================="
