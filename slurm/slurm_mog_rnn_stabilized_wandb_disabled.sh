#!/bin/bash
#SBATCH --job-name=mog-rnn-stab
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
# Single run: MoG-PQN-RNN stabilized (Double DQN + Polyak). See mog_pqn_rnn_stabilized_craftax.py
#
# **W&B is disabled** for maximum throughput (no jax.debug.callback in the compiled loop).
# Metrics are written once at the end to METRICS_NPZ (see below). Use the commented block at the
# bottom of this file to replay curves into W&B after the job finishes.
#
# Submit: sbatch slurm/slurm_mog_rnn_stabilized_wandb_disabled.sh
#
# Optional env overrides (same idea as the 2-algo script):
#   export WANDB_EXPERIMENT_TAG=MyTag
#   export CRAFTAX_ENV_NAME=Craftax-Symbolic-v1
#   export SEED=0

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
mkdir -p slurm/logs

HYDRA_CONFIG_DIR="${ROOT}/purejaxql/config"

export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPERIMENT_TAG="${WANDB_EXPERIMENT_TAG:-Craftax_MoG_RNN_stabilized}"
CRAFTAX_ENV_NAME="${CRAFTAX_ENV_NAME:-Craftax-Symbolic-v1}"
SEED="${SEED:-0}"

# One .npz per job (full training curves; no live W&B during the run)
METRICS_DIR="${ROOT}/slurm/metrics/${SLURM_JOB_ID:-local}"
mkdir -p "${METRICS_DIR}"
METRICS_NPZ="${METRICS_DIR}/train_metrics.npz"

echo "=========================================="
echo "MoG-PQN-RNN stabilized (Craftax) — WANDB_MODE=disabled (speed)"
echo "=========================================="
echo "Seed:          ${SEED}"
echo "Env:           ${CRAFTAX_ENV_NAME}"
echo "W&B tag:       ${EXPERIMENT_TAG}"
echo "Project:       ${WANDB_PROJECT}"
echo "Metrics .npz:  ${METRICS_NPZ}"
echo "Job ID:        ${SLURM_JOB_ID:-}"
echo "Host:          $(hostname)"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-}"
echo "Start:         $(date)"
echo "=========================================="

srun uv run --no-sync python -m purejaxql.mog_pqn_rnn_stabilized_craftax \
  --config-path "${HYDRA_CONFIG_DIR}" --config-name config \
  "SEED=${SEED}" \
  NUM_SEEDS=1 \
  WANDB_MODE=disabled \
  "METRICS_SAVE_PATH=${METRICS_NPZ}" \
  "+alg.EXPERIMENT_TAG=${EXPERIMENT_TAG}" \
  "alg.ENV_NAME=${CRAFTAX_ENV_NAME}" \
  "+alg=mog_pqn_rnn_stabilized_craftax"

echo "Completed"
echo "End:      $(date)"
echo "Metrics:  ${METRICS_NPZ}"
echo "=========================================="

# -----------------------------------------------------------------------------
# After the experiment: push saved curves to W&B (run from repo root on a machine
# with network; edit PROJECT / NAME / entity as needed). This does not affect training speed.
#
#   export WANDB_PROJECT="${WANDB_PROJECT:-Deep-CVI-Experiments}"
#   export METRICS_NPZ="${ROOT}/slurm/metrics/<JOB_ID>/train_metrics.npz"
#
#   uv run python <<'PY'
#   import os
#   import numpy as np
#   import wandb
#
#   path = os.environ["METRICS_NPZ"]
#   data = np.load(path)
#   keys = list(data.files)
#   step_key = "update_steps" if "update_steps" in keys else "env_step"
#   y = data[step_key]
#   # NUM_SEEDS=1: shape (NUM_UPDATES,). If you vmap seeds, use index 0: y = data[step_key][0]
#   if y.ndim > 1:
#       y = y[0]
#   wandb.init(
#       project=os.environ.get("WANDB_PROJECT", "Deep-CVI-Experiments"),
#       name="posthoc-mog-rnn-stabilized",
#       job_type="metrics_replay",
#       reinit=True,
#   )
#   for i in range(y.shape[0]):
#       row = {}
#       for k in keys:
#           v = data[k]
#           if v.ndim > 1:
#               v = v[0, i]
#           else:
#               v = v[i]
#           row[k] = float(np.asarray(v))
#       wandb.log(row, step=int(row[step_key]))
#   wandb.finish()
#   PY
#
# Or sync an offline W&B run instead (only if you trained with WANDB_MODE=offline):
#   wandb sync /path/to/wandb/offline-run-*
# -----------------------------------------------------------------------------
