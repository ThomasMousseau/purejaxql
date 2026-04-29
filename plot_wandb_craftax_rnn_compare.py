#!/usr/bin/env python3
"""
Plot Craftax **RNN** comparison across five algos (**MoG-PQN-RNN**, **PQN-RNN**, **IQN-RNN**, **QTD-RNN**, **CTD-RNN**) from Weights & Biases.

- Filters runs by **experiment tag** (default: ``Craftax_1B_RNN_3algo``; matches
  ``WANDB_EXPERIMENT_TAG`` / ``alg.EXPERIMENT_TAG`` in ``slurm_craftax_1b_rnn_2algo.sh``) plus exactly
  one **algo tag** per run: ``MOG_PQN_RNN``, ``PQN_RNN``, ``IQN_RNN``, ``QTD_RNN``, or ``CTD_RNN`` (from ``ALG_NAME`` in the
  purejaxql trainers).
  Override or extend via ``--algo-tags`` if needed.
- X-axis: ``env_step`` (default) or ``update_steps`` — purejaxql RNN trainers log both.
- Y-axis: ``returned_episode_returns`` (mean over parallel envs at episode end; same as other Craftax plots).
- Curve: **mean across seeds**; shaded band: **95% CI** (mean ± 1.96 * SEM).
- **Smoothing**: centered moving average on the mean and CI band width via ``--smooth-window``.
- **multi_seed** (optional): runs tagged ``multi_seed`` with ``seed_i/returned_episode_returns`` (``WANDB_LOG_ALL_SEEDS``)
  are expanded into multiple curves per run — same idea as :func:`old_cvi.plot_wandb_minatar.curves_from_wandb_run`.

Colors (shared defaults via ``plot_colors``):
  - **MoG-PQN-RNN**: dark blue ``#003a7d``
  - **IQN-RNN**: medium blue ``#008dff``
  - **CTD-RNN**: pink ``#ff73b6``
  - **QTD-RNN**: purple ``#c701ff``
  - **PQN-RNN**: green ``#4ecb8d`` (used as the 5th algo color)

Requires: ``wandb``, ``matplotlib``, ``numpy``. Login: ``wandb login``

In ``main()``, edit the active ``plot_craftax_rnn_compare(...)`` call directly
to replot a previous experiment quickly (same style as ``plot_wandb_minatar.py``).
"""
from __future__ import annotations

import os
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
from plot_colors import algo_color

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

try:
    import wandb
except ImportError as e:
    raise SystemExit("Please install wandb: pip install wandb") from e

from plot_wandb_minatar import (
    _interp_on_grid,
    _run_matches_required,
    _smooth_1d,
    _wandb_path,
    curves_from_wandb_run,
)

# Defaults for 5-algo Craftax RNN compare.
ALGO_FILL: dict[str, str] = {
    "MOG_PQN_RNN": algo_color("MOG_PQN_RNN"),
    "PQN_RNN": algo_color("PQN_RNN"),
    "IQN_RNN": algo_color("IQN_RNN"),
    "QTD_RNN": algo_color("QTD_RNN"),
    "CTD_RNN": algo_color("CTD_RNN"),
}
ALGO_LINE: dict[str, str] = {
    "MOG_PQN_RNN": algo_color("MOG_PQN_RNN"),
    "PQN_RNN": algo_color("PQN_RNN"),
    "IQN_RNN": algo_color("IQN_RNN"),
    "QTD_RNN": algo_color("QTD_RNN"),
    "CTD_RNN": algo_color("CTD_RNN"),
}

ALGO_LABELS: dict[str, str] = {
    "MOG_PQN_RNN": "MoG-CQN-RNN",
    "PQN_RNN": "PQN-RNN",
    "IQN_RNN": "IQN-RNN",
    "QTD_RNN": "QTD-RNN",
    "CTD_RNN": "CTD-RNN",
}


def _pretty_metric_label(metric: str) -> str:
    metric_label_map: dict[str, str] = {
        "charts/episodic_return": "Episodic Return",
        "charts/episode_return": "Episode Return",
        "episodic_return": "Episodic Return",
        "returned_episode_returns": "Episodic Return",
        "charts/grad_norm": "Gradient Norm",
        "grad_norm": "Gradient Norm",
    }
    if metric in metric_label_map:
        return metric_label_map[metric]
    tail = metric.split("/")[-1]
    return tail.replace("_", " ").strip().title()


def _pretty_step_label(step_metric: str) -> str:
    step_label_map: dict[str, str] = {
        "global_step": "Environment Steps",
        "env_step": "Environment Steps",
        "update_steps": "Update Steps",
        "_step": "Logging Step",
    }
    if step_metric in step_label_map:
        return step_label_map[step_metric]
    return step_metric.replace("_", " ").strip().title()


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


def _band(mean: np.ndarray, std: np.ndarray, n_per_step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 95% normal-approx confidence interval: mean ± 1.96 * SEM.
    sem = std / np.sqrt(np.maximum(n_per_step.astype(np.float64), 1.0))
    sem[n_per_step < 2] = np.nan
    e = 1.96 * sem
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
    env_name_filter: str | None,
    display_titles: bool = True,
) -> None:
    required = [experiment_tag]
    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

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
            multi_seed_tag="",
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

    def _title_variant_path(path: str) -> str:
        p = Path(path)
        return str(p.with_name(f"{p.stem}_with_titles{p.suffix}"))

    def _seed_summary_text() -> str:
        parts: list[str] = []
        for algo_tag in algo_tags:
            curves = by_algo.get(algo_tag, [])
            if not curves:
                continue
            parts.append(f"{ALGO_LABELS.get(algo_tag, algo_tag)}: n={len(curves)}")
        return "; ".join(parts)

    def _render_single(output_path: str, *, include_titles: bool) -> None:
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
            std_valid = n_per_step >= 2
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            std[~std_valid] = np.nan

            if smooth_window and smooth_window > 1:
                mean = _smooth_1d(mean, smooth_window)
                std = _smooth_1d(std, smooth_window)
                std[~std_valid] = np.nan

            lower, upper = _band(mean, std, n_per_step)
            ymax_track = max(ymax_track, float(np.nanmax(upper)))
            ymin_track = min(ymin_track, float(np.nanmin(lower)))

            fill_c = ALGO_FILL.get(algo_tag, "#444444")
            line_c = ALGO_LINE.get(algo_tag, fill_c)
            label = ALGO_LABELS.get(algo_tag, algo_tag)
            ax.plot(grid, mean, color=line_c, linewidth=2.0, label=label)
            ax.fill_between(grid, lower, upper, color=fill_c, alpha=0.28)

        ax.set_xlabel(_pretty_step_label(step_metric))
        ax.set_ylabel(_pretty_metric_label(metric))
        if include_titles:
            metric_title = _pretty_metric_label(metric)
            seed_summary = _seed_summary_text() or "no matched runs"
            env_txt = env_name_filter if env_name_filter else "all environments"
            ax.set_title(
                f"{experiment_tag} | env={env_txt} | metric={metric_title} | shaded=95% CI | {seed_summary}",
                fontsize=10,
            )
        else:
            ax.set_title("Craftax 1B", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if np.isfinite(ymax_track) and np.isfinite(ymin_track):
            pad = 0.08 * max(ymax_track - ymin_track, 1e-6)
            ax.set_ylim(bottom=max(0.0, ymin_track - pad), top=ymax_track + pad)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output_path}")

    _render_single(out, include_titles=display_titles)
    if not display_titles:
        _render_single(_title_variant_path(out), include_titles=True)


def plot_craftax_rnn_compare_multi_source(
    *,
    project: str,
    entity: str | None,
    groups: list[dict[str, str]],
    metric: str,
    step_metric: str,
    out: str,
    grid_points: int,
    max_runs: int,
    smooth_window: int,
    env_name_filter: str | None,
    display_titles: bool = True,
) -> None:
    """
    Plot compare curves when the same algo tag should appear multiple times
    from different experiment tags.

    Each group entry must contain:
      - name: unique display key for legend ordering
      - experiment_tag: required tag for run selection
      - algo_tag: single algo tag required in run tags
    Optional:
      - label: legend label override
      - linestyle: matplotlib line style (defaults to solid)
    """
    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

    by_name: dict[str, list] = defaultdict(list)
    for group in groups:
        name = group["name"]
        required = [group["experiment_tag"]]
        algo_tag = group["algo_tag"]
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
            g = _algo_group(run, [algo_tag])
            if g is None:
                continue
            for series in curves_from_wandb_run(
                run,
                metric=metric,
                step_metric=step_metric,
                multi_seed_tag="",
            ):
                by_name[name].append(series)

    missing = [g["name"] for g in groups if len(by_name.get(g["name"], [])) == 0]
    if len(missing) == len(groups):
        raise RuntimeError(
            "No runs matched any configured group. "
            "Check experiment/algo tags, entity/project, and metric/step keys."
        )
    if missing:
        print(f"Warning: no runs for groups: {missing}")

    def _title_variant_path(path: str) -> str:
        p = Path(path)
        return str(p.with_name(f"{p.stem}_with_titles{p.suffix}"))

    def _group_seed_summary_text() -> str:
        parts: list[str] = []
        for group in groups:
            name = group["name"]
            algo_tag = group["algo_tag"]
            default_label = ALGO_LABELS.get(algo_tag, algo_tag)
            label = group.get("label", default_label)
            n = len(by_name.get(name, []))
            if n > 0:
                parts.append(f"{label}: n={n}")
        return "; ".join(parts)

    def _render_single(output_path: str, *, include_titles: bool) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.0))
        ymax_track = -np.inf
        ymin_track = np.inf

        for group in groups:
            name = group["name"]
            algo_tag = group["algo_tag"]
            curves = by_name.get(name, [])
            if not curves:
                continue
            all_steps = np.concatenate([c[0] for c in curves])
            s_min, s_max = np.nanmin(all_steps), np.nanmax(all_steps)
            if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
                continue
            grid = np.linspace(s_min, s_max, grid_points)
            mat = _interp_on_grid(curves, grid)
            n_per_step = np.sum(np.isfinite(mat), axis=0)
            std_valid = n_per_step >= 2
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            std[~std_valid] = np.nan

            if smooth_window and smooth_window > 1:
                mean = _smooth_1d(mean, smooth_window)
                std = _smooth_1d(std, smooth_window)
                std[~std_valid] = np.nan

            lower, upper = _band(mean, std, n_per_step)
            ymax_track = max(ymax_track, float(np.nanmax(upper)))
            ymin_track = min(ymin_track, float(np.nanmin(lower)))

            fill_c = ALGO_FILL.get(algo_tag, "#444444")
            line_c = ALGO_LINE.get(algo_tag, fill_c)
            default_label = ALGO_LABELS.get(algo_tag, algo_tag)
            label = group.get("label", default_label)
            linestyle = group.get("linestyle", "-")
            ax.plot(
                grid,
                mean,
                color=line_c,
                linewidth=2.0,
                linestyle=linestyle,
                label=label,
            )
            ax.fill_between(grid, lower, upper, color=fill_c, alpha=0.22)

        ax.set_xlabel(_pretty_step_label(step_metric))
        ax.set_ylabel(_pretty_metric_label(metric))
        if include_titles:
            metric_title = _pretty_metric_label(metric)
            seed_summary = _group_seed_summary_text() or "no matched runs"
            tags = ", ".join(sorted({g["experiment_tag"] for g in groups}))
            env_txt = env_name_filter if env_name_filter else "all environments"
            ax.set_title(
                f"{tags} | env={env_txt} | metric={metric_title} | shaded=95% CI | {seed_summary}",
                fontsize=10,
            )
        else:
            ax.set_title("Craftax 1B", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if np.isfinite(ymax_track) and np.isfinite(ymin_track):
            pad = 0.08 * max(ymax_track - ymin_track, 1e-6)
            ax.set_ylim(bottom=max(0.0, ymin_track - pad), top=ymax_track + pad)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {output_path}")

    _render_single(out, include_titles=display_titles)
    if not display_titles:
        _render_single(_title_variant_path(out), include_titles=True)


def main() -> None:
    
    #! Craftax 1B (using lambda=0.5 and aux mean loss weight=1.0)
    plot_craftax_rnn_compare(
        project="Deep-CVI-Experiments",
        entity="fatty_data",
        experiment_tag="Craftax-1B-checkpoint",
        algo_tags=["MOG_PQN_RNN", "PQN_RNN", "QTD_RNN", "CTD_RNN"], #IQN-RNN
        metric="returned_episode_returns",
        step_metric="env_step",
        out="figures/craftax_1b_rnn_lambda0.5_auxmeanlossweight1.0_compare.png",
        grid_points=800,
        max_runs=5000,
        smooth_window=51,
        env_name_filter=None,
        display_titles=False,
    )

    #! Craftax 1B (using lambda=0.0 and aux mean loss weight=0.0)
    plot_craftax_rnn_compare_multi_source(
        project="Deep-CVI-Experiments",
        entity="fatty_data",
        groups=[
            {"name": "ctd_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "CTD_RNN"},
            {"name": "qtd_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "QTD_RNN"},
            # {"name": "mog_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "MOG_PQN_RNN", "label": "MOG-CQN-RNN (5ALG)"},
            {"name": "pqn_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "PQN_RNN"},
            {
                "name": "mog_weightedcf",
                "experiment_tag": "Craftax-RNN-CTD-MoG-WeightedCF",
                "algo_tag": "MOG_PQN_RNN",
                "label": "MoG-CQN-RNN",
            },
        ],
        metric="returned_episode_returns",
        step_metric="env_step",
        out="figures/craftax_1b_rnn_lambda0.0_auxmeanlossweight0.0_compare.png",
        grid_points=800,
        max_runs=5000,
        smooth_window=51,
        env_name_filter=None,
        display_titles=False,
    )



if __name__ == "__main__":
    main()
