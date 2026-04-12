"""Replay ``save_train_metrics_npz`` output into a new W&B run (no JAX / no GPU)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import wandb


def _scalar_at_step(arr: np.ndarray, seed_index: int, step_index: int) -> float:
    if arr.ndim == 1:
        return float(np.asarray(arr[step_index]))
    if arr.ndim == 2:
        return float(np.asarray(arr[seed_index, step_index]))
    raise ValueError(
        f"Expected 1D or 2D array, got shape {arr.shape} (use --seed-index for vmapped runs)"
    )


def replay(
    npz_path: str,
    *,
    project: str,
    entity: str | None,
    name: str,
    job_type: str,
    seed_index: int,
) -> None:
    data = np.load(npz_path)
    keys = list(data.files)
    if not keys:
        raise ValueError(f"No arrays in {npz_path}")

    step_key = "update_steps" if "update_steps" in keys else "env_step"
    if step_key not in keys:
        raise ValueError(f"Need {step_key!r} or 'env_step' in npz keys, got {keys}")

    y = data[step_key]
    n = y.shape[-1] if y.ndim >= 1 else 1
    if y.ndim == 0:
        n = 1

    init_kw: dict = {
        "project": project,
        "name": name,
        "job_type": job_type,
        "reinit": True,
    }
    if entity:
        init_kw["entity"] = entity
    wandb.init(**init_kw)

    try:
        for i in range(n):
            row: dict[str, float] = {}
            for k in keys:
                row[k] = _scalar_at_step(data[k], seed_index, i)
            wandb.log(row, step=int(row[step_key]))
    finally:
        wandb.finish()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: uv run python -m purejaxql.utils.replay_train_metrics_npz_to_wandb "
        "slurm/metrics/12345/train_metrics.npz --name posthoc-my-run",
    )
    p.add_argument(
        "npz_path",
        nargs="?",
        default=os.environ.get("METRICS_NPZ"),
        help="Path to train_metrics .npz (default: METRICS_NPZ env)",
    )
    p.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "Deep-CVI-Experiments"),
        help="W&B project (default: WANDB_PROJECT or Deep-CVI-Experiments)",
    )
    p.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity (optional; default: WANDB_ENTITY)",
    )
    p.add_argument(
        "--name",
        default="posthoc-train-metrics",
        help="Run name in W&B",
    )
    p.add_argument(
        "--job-type",
        default="metrics_replay",
        help="W&B job_type",
    )
    p.add_argument(
        "--seed-index",
        type=int,
        default=0,
        help="When arrays are 2D (vmapped seeds), which seed row to log",
    )
    args = p.parse_args()
    if not args.npz_path:
        p.error("Pass npz_path or set METRICS_NPZ")

    replay(
        args.npz_path,
        project=args.project,
        entity=args.entity or None,
        name=args.name,
        job_type=args.job_type,
        seed_index=args.seed_index,
    )


if __name__ == "__main__":
    main()
