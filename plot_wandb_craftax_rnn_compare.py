#!/usr/bin/env python3
"""
Plot Craftax **RNN** comparison (**PQN-RNN** vs **MoG-PQN-RNN**) from Weights & Biases.

- Filters runs by **experiment tag** (default: ``Craftax_1B_RNN_3algo`` — name is historical; matches
  ``WANDB_EXPERIMENT_TAG`` / ``alg.EXPERIMENT_TAG`` in ``slurm_craftax_1b_rnn_2algo.sh``) plus exactly
  one **algo tag** per run: ``PQN_RNN`` or ``MOG_PQN_RNN`` (from ``ALG_NAME`` in the purejaxql trainers).
  Override or extend via ``--algo-tags`` if needed.
- X-axis: ``env_step`` (default) or ``update_steps`` — purejaxql RNN trainers log both.
- Y-axis: ``returned_episode_returns`` (mean over parallel envs at episode end; same as other Craftax plots).
- Curve: **mean across seeds**; shaded band: **std** or **SEM** (see ``--error-type``).
- **Smoothing**: centered moving average on the mean (and on std/sem bands) via ``--smooth-window``.
- **multi_seed** (optional): runs tagged ``multi_seed`` with ``seed_i/returned_episode_returns`` (``WANDB_LOG_ALL_SEEDS``)
  are expanded into multiple curves per run — same idea as :func:`old_cvi.plot_wandb_minatar.curves_from_wandb_run`.

Colors (fixed):
  - **MoG-PQN-RNN**: mid blue ``#2077b4``
  - **PQN-RNN**: light green ``#7CCD7C``

Requires: ``wandb``, ``matplotlib``, ``numpy``. Login: ``wandb login``

Example::

    uv run python plot_wandb_craftax_rnn_compare.py \\
        --entity fatty_data --project Deep-CVI-Experiments \\
        --experiment-tag Craftax_1B_RNN_3algo \\
        --out figures/craftax_1b_rnn_compare.png
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

try:
    import wandb
except ImportError as e:
    raise SystemExit("Please install wandb: pip install wandb") from e

from old_cvi.plot_wandb_minatar import (
    _interp_on_grid,
    _run_matches_required,
    _smooth_1d,
    _wandb_path,
    curves_from_wandb_run,
)

# Fills: MoG mid blue; PQN light green.
ALGO_FILL: dict[str, str] = {
    "MOG_PQN_RNN": "#2077b4",
    "PQN_RNN": "#7CCD7C",
}
ALGO_LINE: dict[str, str] = {
    "MOG_PQN_RNN": "#2077b4",
    "PQN_RNN": "#5CB85C",
}

ALGO_LABELS: dict[str, str] = {
    "MOG_PQN_RNN": "MoG-PQN-RNN",
    "PQN_RNN": "PQN-RNN",
}


def _algo_group(run, algo_tags: list[str]) -> str | None:
    tags = set(run.tags or [])
    found = [a for a in algo_tags if a in tags]
    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        return None
    raise ValueError(
        f"Run {run.id} has multiple algo tags among {algo_tags}: {found}; use one tag per run."
    )


def _band(mean: np.ndarray, std: np.ndarray, n: int, error_type: str) -> tuple[np.ndarray, np.ndarray]:
    if error_type == "sem" and n > 1:
        e = std / np.sqrt(n)
    else:
        e = std
    return mean - e, mean + e


def plot_craftax_rnn_compare(
    *,
    project: str,
    entity: str | None,
    experiment_tag: str,
    algo_tags: list[str],
    metric: str,
    step_metric: str,
    out: str,
    grid_points: int,
    max_runs: int,
    smooth_window: int,
    error_type: str,
    env_name_filter: str | None,
    multi_seed_tag: str = "multi_seed",
) -> None:
    required = [experiment_tag]
    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(api.runs(path, order="-created_at"))[:max_runs]

    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        if not _run_matches_required(run, required):
            continue
        if env_name_filter:
            cfg = getattr(run, "config", None)
            ev = None
            if cfg is not None and hasattr(cfg, "get"):
                ev = cfg.get("ENV_NAME") or cfg.get("env_name")
            if ev is not None and str(ev) != env_name_filter:
                continue
        g = _algo_group(run, algo_tags)
        if g is None:
            continue
        for series in curves_from_wandb_run(
            run,
            metric=metric,
            step_metric=step_metric,
            multi_seed_tag=multi_seed_tag,
        ):
            by_algo[g].append(series)

    missing = [a for a in algo_tags if len(by_algo.get(a, [])) == 0]
    if len(missing) == len(algo_tags):
        raise RuntimeError(
            f"No runs matched (experiment_tag={experiment_tag!r}, algos={algo_tags}). "
            "Check tags, entity/project, and metric/step keys."
        )
    if missing:
        print(f"Warning: no runs for algo tags: {missing}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.0))
    ymax_track = -np.inf
    ymin_track = np.inf

    for algo_tag in algo_tags:
        curves = by_algo.get(algo_tag, [])
        if not curves:
            continue
        all_steps = np.concatenate([c[0] for c in curves])
        s_min, s_max = np.nanmin(all_steps), np.nanmax(all_steps)
        if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
            continue
        grid = np.linspace(s_min, s_max, grid_points)
        mat = _interp_on_grid(curves, grid)
        n_per_step = np.sum(np.isfinite(mat), axis=0)
        n_seeds = float(len(curves))
        std_valid = n_per_step >= 2
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        std[~std_valid] = np.nan

        if smooth_window and smooth_window > 1:
            mean = _smooth_1d(mean, smooth_window)
            std = _smooth_1d(std, smooth_window)
            std[~std_valid] = np.nan

        lower, upper = _band(mean, std, int(n_seeds), error_type)
        ymax_track = max(ymax_track, float(np.nanmax(upper)))
        ymin_track = min(ymin_track, float(np.nanmin(lower)))

        fill_c = ALGO_FILL.get(algo_tag, "#444444")
        line_c = ALGO_LINE.get(algo_tag, fill_c)
        label = ALGO_LABELS.get(algo_tag, algo_tag)
        ax.plot(grid, mean, color=line_c, linewidth=2.0, label=f"{label} (n={len(curves)})")
        ax.fill_between(grid, lower, upper, color=fill_c, alpha=0.28)

    ax.set_xlabel(step_metric)
    ax.set_ylabel(metric)
    err_name = "SEM" if error_type == "sem" else "std"
    ax.set_title(
        f"{experiment_tag} — mean ± {err_name} over seeds (smoothing window={smooth_window})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if np.isfinite(ymax_track) and np.isfinite(ymin_track):
        pad = 0.08 * max(ymax_track - ymin_track, 1e-6)
        ax.set_ylim(bottom=max(0.0, ymin_track - pad), top=ymax_track + pad)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Craftax RNN PQN-RNN vs MoG-PQN-RNN comparison from wandb.")
    p.add_argument("--project", default="Deep-CVI-Experiments")
    p.add_argument("--entity", default=None)
    p.add_argument(
        "--experiment-tag",
        default="Craftax_1B_RNN_3algo",
        help="Must match W&B experiment tag (same as alg.EXPERIMENT_TAG / WANDB_EXPERIMENT_TAG in slurm).",
    )
    p.add_argument(
        "--algo-tags",
        nargs="+",
        default=["PQN_RNN", "MOG_PQN_RNN"],
        help="W&B algo tags (one per run; ALG_NAME.upper() from trainers). Default: PQN-RNN then MoG-PQN-RNN.",
    )
    p.add_argument("--metric", default="returned_episode_returns")
    p.add_argument("--step-metric", default="env_step", help="e.g. env_step or update_steps")
    p.add_argument("--out", default="figures/craftax_rnn_2algo_compare.png")
    p.add_argument("--grid-points", type=int, default=800)
    p.add_argument("--max-runs", type=int, default=5000)
    p.add_argument("--smooth-window", type=int, default=51)
    p.add_argument(
        "--error-type",
        choices=("std", "sem"),
        default="std",
        help="Shaded band: std across seeds, or SEM (std/sqrt(n)).",
    )
    p.add_argument(
        "--env-name",
        default=None,
        help="If set, filter runs where config ENV_NAME matches (e.g. Craftax-Symbolic-v1).",
    )
    p.add_argument(
        "--multi-seed-tag",
        default="multi_seed",
        help="If present on a run, load seed_i/metric series (see curves_from_wandb_run). Use '' to disable.",
    )
    args = p.parse_args()

    plot_craftax_rnn_compare(
        project=args.project,
        entity=args.entity,
        experiment_tag=args.experiment_tag,
        algo_tags=list(args.algo_tags),
        metric=args.metric,
        step_metric=args.step_metric,
        out=args.out,
        grid_points=args.grid_points,
        max_runs=args.max_runs,
        smooth_window=args.smooth_window,
        error_type=args.error_type,
        env_name_filter=args.env_name,
        multi_seed_tag=args.multi_seed_tag or "",
    )


if __name__ == "__main__":
    main()
