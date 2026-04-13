#!/bin/bash
#SBATCH --job-name=craftax-mog-ablate
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-53%27
#
# Craftax MoG-PQN-RNN ablation: NUM_COMPONENTS × OMEGA_MAX × NUM_OMEGA_SAMPLES.
# Runs both **vanilla** (mog_pqn_rnn_craftax) and **stabilized** (mog_pqn_rnn_stabilized_craftax).
# Each task: 200M env steps; NUM_SEEDS defaults to 1 (override with env NUM_SEEDS).
#
# Full **3×3** grid in (M, ω_max), including **M=8, OMEGA_MAX=1.0** (matches your earlier baseline
# settings) so the study is report-ready in one W&B group. If you already logged that point at
# 1B steps, these 200M runs are still comparable within this ablation block.
#
# Per algorithm: 9 (M, ω) pairs × NUM_OMEGA_SAMPLES ∈ {16, 32, 64} = **27** tasks.
# Total array: 2 algorithms × 27 = **54** tasks (TID **0..53**).
#
# (M, OMEGA_MAX) pairs — index 0..8 (row-major over M ∈ {8,16,32}, ω ∈ {0.5,1.0,2.0}):
#   (8,0.5) (8,1.0) (8,2.0) (16,0.5) (16,1.0) (16,2.0) (32,0.5) (32,1.0) (32,2.0)
#
# Task index mapping (SLURM_ARRAY_TASK_ID = TID):
#   algo_idx   = TID / 27     → 0 = vanilla, 1 = stabilized
#   sub        = TID % 27
#   pair_idx   = sub / 3     → 0..8  → (M, ω) row above
#   nom_idx    = sub % 3     → 0,1,2 → NUM_OMEGA_SAMPLES = 16, 32, 64
#
# Experiment tag / W&B name pattern:
#   Craftax_MoG_ablate_<vanilla|stab>_M<M>_om<omega>_nOm<nOmega>
#
# Submit from repo root:
#   sbatch slurm/slurm_craftax_mog_rnn_ablation_M_omega.sh
#
# Weights & Biases — sweep-style analysis without ``wandb agent``:
#   - Default **project** is ``Craftax-MoG-Ablation`` so these runs stay separate from
#     your main project; override with ``WANDB_PROJECT`` if you prefer.
#   - **WANDB_RUN_GROUP** (default below) puts every array task in the same *group* so
#     you can use Compare runs, Parallel coordinates, and correlation on hyperparameters
#     vs ``returned_episode_returns`` in one workspace. This matches how people analyze
#     fixed grids launched from SLURM (official ``wandb sweep`` + agent is for runs that
#     *pull* configs from W&B; here each job already has its Hydra overrides).
#   - **WANDB_JOB_TYPE**=ablation labels runs in the UI.
#
# Optional env overrides:
#   export WANDB_PROJECT=Craftax-MoG-Ablation
#   export WANDB_RUN_GROUP=Craftax_MoG_ablation_M_omega_200M   # same for whole batch
#   export WANDB_JOB_TYPE=ablation
#   export WANDB_NOTES='optimize returned_episode_returns'
#   export WANDB_MODE=online          # or offline / disabled
#   export CRAFTAX_ENV_NAME=Craftax-Symbolic-v1
#   export NUM_SEEDS=1                # vmap over seeds: >1 uses more VRAM (see trainer)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Craftax-MoG-Ablation}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-Craftax_MoG_ablation_M_omega_200M}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-ablation}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"

TID="${SLURM_ARRAY_TASK_ID:?}"

# 9 pairs: full grid M ∈ {8,16,32} × ω ∈ {0.5,1.0,2.0}
M_FOR_PAIR=(8 8 8 16 16 16 32 32 32)
O_FOR_PAIR=(0.5 1.0 2.0 0.5 1.0 2.0 0.5 1.0 2.0)

NOMEGA_VALUES=(16 32 64)

ALGO_IDX=$((TID / 27))
SUB=$((TID % 27))
PAIR_IDX=$((SUB / 3))
NOM_IDX=$((SUB % 3))

M="${M_FOR_PAIR[$PAIR_IDX]}"
OMEGA_MAX="${O_FOR_PAIR[$PAIR_IDX]}"
NUM_OMEGA_SAMPLES="${NOMEGA_VALUES[$NOM_IDX]}"

TOTAL_STEPS=200000000 #200M
NUM_SEEDS="${NUM_SEEDS:-1}"

if [[ "${ALGO_IDX}" -eq 0 ]]; then
  VARIANT="vanilla"
  PY_MODULE="purejaxql.mog_pqn_rnn_craftax"
  HYDRA_ALG="mog_pqn_rnn_craftax"
else
  VARIANT="stab"
  PY_MODULE="purejaxql.mog_pqn_rnn_stabilized_craftax"
  HYDRA_ALG="mog_pqn_rnn_stabilized_craftax"
fi

# Human-readable tag for W&B (parameters in the name)
EXPERIMENT_TAG="Craftax_MoG_ablate_${VARIANT}_M${M}_om${OMEGA_MAX}_nOm${NUM_OMEGA_SAMPLES}"

echo "=========================================="
echo "Craftax MoG-RNN ablation (M × ω_max × NUM_OMEGA_SAMPLES)"
echo "=========================================="
echo "Task ID:              ${TID}"
echo "Variant:              ${VARIANT} (${HYDRA_ALG})"
echo "NUM_COMPONENTS:       ${M}"
echo "OMEGA_MAX:            ${OMEGA_MAX}"
echo "NUM_OMEGA_SAMPLES:    ${NUM_OMEGA_SAMPLES}"
echo "TOTAL_TIMESTEPS:      ${TOTAL_STEPS}"
echo "NUM_SEEDS:            ${NUM_SEEDS}"
echo "WANDB_PROJECT:        ${WANDB_PROJECT}"
echo "WANDB_RUN_GROUP:      ${WANDB_RUN_GROUP}"
echo "Env:                  ${CRAFTAX_ENV_NAME}"
echo "Experiment tag:       ${EXPERIMENT_TAG}"
echo "Job ID:               ${SLURM_JOB_ID:-}"
echo "Array Job ID:         ${SLURM_ARRAY_JOB_ID:-}"
echo "Host:                 $(hostname)"
echo "GPU:                  ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:                $(date)"
echo "=========================================="

srun uv run --no-sync python -m "${PY_MODULE}" \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  SEED=0 \
  NUM_SEEDS="${NUM_SEEDS}" \
  WANDB_MODE="${WANDB_MODE}" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}" \
  "alg.TOTAL_TIMESTEPS=${TOTAL_STEPS}" \
  "alg.TOTAL_TIMESTEPS_DECAY=${TOTAL_STEPS}" \
  "alg.NUM_COMPONENTS=${M}" \
  "alg.OMEGA_MAX=${OMEGA_MAX}" \
  "alg.NUM_OMEGA_SAMPLES=${NUM_OMEGA_SAMPLES}" \
  "+alg=${HYDRA_ALG}"

echo "Completed task ${TID}"
echo "End: $(date)"
echo "=========================================="
