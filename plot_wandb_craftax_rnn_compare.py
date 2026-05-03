#!/usr/bin/env python3
"""
Plot Craftax **RNN** comparison from Weights & Biases using the **same** curve styling as
``plot_wandb_minatar.plot_episodic_return``: canonical colors from ``plot_colors.algo_color``,
legend text from ``_legend_for_algo_tag`` (e.g. **MoG-PQN-RNN**, **PQN-RNN**), 95% CI band,
``style_axes_wandb_curve``, and a **single** figure title (``Craftax`` or a pretty-printed
``env_name_filter`` when set).

- Filters runs by experiment tag(s) plus exactly one **algo tag** per run
  (``MOG_PQN_RNN``, ``PQN_RNN``, …).
- X-axis: ``env_step`` (default) or ``update_steps``.
- Y-axis: ``returned_episode_returns`` → label **Episode Return** (same as MinAtar).
- Curve aggregation and smoothing match ``plot_wandb_minatar._draw_algo_curves_on_ax``.
- Use ``plot_craftax_phi_td_with_baselines`` for: legacy φTD + baselines; **fair** M=8 vs M=51
  (dotted, ``(M=51)`` legend); or ``final_report`` with a custom group list (typically M=8 only).

Requires: ``wandb``, ``matplotlib``, ``numpy``. Login: ``wandb login``
"""
from __future__ import annotations

import os
from collections import defaultdict
from itertools import islice

import numpy as np
from plot_colors import algo_color
from paper_plots import (
    DEFAULT_CURVE_Y_MARGIN_FRAC,
    configure_matplotlib,
    save_figure_png_and_pdf,
    style_axes_wandb_curve,
)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

try:
    import wandb
except ImportError as e:
    raise SystemExit("Please install wandb: pip install wandb") from e

configure_matplotlib()

from plot_wandb_minatar import (
    _algo_colors,
    _draw_algo_curves_on_ax,
    _interp_on_grid,
    _legend_for_algo_tag,
    _pretty_env_title,
    _pretty_metric_label,
    _pretty_step_label,
    _run_matches_required,
    _smooth_1d,
    _wandb_path,
    curves_from_wandb_run,
)


def _figure_title_craftax(env_name_filter: str | None) -> str:
    """Single-panel title: match MinAtar convention (short env-style name or benchmark name)."""
    if env_name_filter and str(env_name_filter).strip():
        return _pretty_env_title(str(env_name_filter))
    return "Craftax"


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


def _plot_curves_multi_groups(
    ax,
    *,
    groups: list[dict],
    by_name: dict[str, list],
    grid_points: int,
    smooth_window: int,
    metric: str,
    step_metric: str,
) -> None:
    """
    Same mean / 95% CI / smoothing as ``_draw_algo_curves_on_ax``, but one pooled curve per group
    (legend label may override ``_legend_for_algo_tag``).
    """
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
        spread = 1.96 * (std / np.sqrt(np.maximum(n_per_step.astype(np.float64), 1.0)))
        spread[~std_valid] = np.nan
        if smooth_window and smooth_window > 1:
            mean = _smooth_1d(mean, smooth_window)
            spread = _smooth_1d(spread, smooth_window)
            spread[~std_valid] = np.nan
        upper = mean + spread
        lower = mean - spread
        ymax_track = max(ymax_track, float(np.nanmax(upper)))
        ymin_track = min(ymin_track, float(np.nanmin(lower)))

        color = algo_color(algo_tag)
        default_label = _legend_for_algo_tag(algo_tag)
        label = group.get("label", default_label)
        linestyle = group.get("linestyle", "-")
        ax.plot(grid, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)
        ax.fill_between(grid, lower, upper, color=color, alpha=0.2)

    ax.set_xlabel(_pretty_step_label(step_metric))
    ax.set_ylabel(_pretty_metric_label(metric))
    n_leg = len(groups)
    leg_fs = 6 if n_leg > 6 else 8
    style_axes_wandb_curve(ax)
    ax.legend(fontsize=leg_fs, loc="best")

    if np.isfinite(ymax_track) and ymax_track > 0:
        span = max(float(ymax_track - ymin_track), 1e-6)
        pad = DEFAULT_CURVE_Y_MARGIN_FRAC * span
        ax.set_ylim(bottom=0.0, top=float(ymax_track) + pad)


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

    seed_parts: list[str] = []
    for algo_tag in algo_tags:
        curves = by_algo.get(algo_tag, [])
        if not curves:
            continue
        seed_parts.append(f"{_legend_for_algo_tag(algo_tag)}: n={len(curves)}")
    if seed_parts:
        print("Curves:", "; ".join(seed_parts))

    # Match ``plot_episodic_return`` single panel: same figsize, shared curve helper, one title.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = _algo_colors(algo_tags)
    _draw_algo_curves_on_ax(
        ax,
        dict(by_algo),
        algo_tags,
        colors,
        grid_points=grid_points,
        smooth_window=smooth_window,
        step_metric=step_metric,
        metric=metric,
        show_ylabel=True,
        autoscale_y=True,
        y_top_margin=DEFAULT_CURVE_Y_MARGIN_FRAC,
        y_bottom=0.0,
    )
    ax.set_title(_figure_title_craftax(env_name_filter), fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    png_path, pdf_path = save_figure_png_and_pdf(fig, out, dpi_png=150, dpi_pdf=300)
    plt.close(fig)
    print(f"Wrote {png_path}\n      {pdf_path}")


def _experiment_tags_for_group(group: dict) -> list[str]:
    """Resolve experiment tag(s) for a group: ``experiment_tags`` (list) or ``experiment_tag`` (str)."""
    if "experiment_tags" in group:
        v = group["experiment_tags"]
        if isinstance(v, str):
            return [v]
        return list(v)
    return [group["experiment_tag"]]


def _run_matches_any_experiment_tag(run, experiment_tags: list[str]) -> bool:
    tags = set(run.tags or [])
    return any(t in tags for t in experiment_tags)


def plot_craftax_rnn_compare_multi_source(
    *,
    project: str,
    entity: str | None,
    groups: list[dict],
    metric: str,
    step_metric: str,
    out: str,
    grid_points: int,
    max_runs: int,
    smooth_window: int,
    env_name_filter: str | None,
) -> None:
    """
    Plot compare curves when the same algo tag should appear multiple times
    from different experiment tags.

    Each group entry must contain:
      - name: unique display key for legend ordering
      - experiment_tag: required tag for run selection (unless experiment_tags is set)
      - algo_tag: single algo tag required in run tags
    Optional:
      - experiment_tags: list of W&B tags; run matches if it has **any** of them
        (use this to pool seeds across experiments into one curve)
      - label: legend label override
      - linestyle: matplotlib line style (defaults to solid)
    """
    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

    by_name: dict[str, list] = defaultdict(list)
    for group in groups:
        name = group["name"]
        experiment_tags = _experiment_tags_for_group(group)
        algo_tag = group["algo_tag"]
        for run in runs:
            if not _run_matches_any_experiment_tag(run, experiment_tags):
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

    parts: list[str] = []
    for group in groups:
        name = group["name"]
        algo_tag = group["algo_tag"]
        default_label = _legend_for_algo_tag(algo_tag)
        label = group.get("label", default_label)
        n = len(by_name.get(name, []))
        if n > 0:
            parts.append(f"{label}: n={n}")
    if parts:
        print("Curves:", "; ".join(parts))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    _plot_curves_multi_groups(
        ax,
        groups=groups,
        by_name=dict(by_name),
        grid_points=grid_points,
        smooth_window=smooth_window,
        metric=metric,
        step_metric=step_metric,
    )
    ax.set_title(_figure_title_craftax(env_name_filter), fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    png_path, pdf_path = save_figure_png_and_pdf(fig, out, dpi_png=150, dpi_pdf=300)
    plt.close(fig)
    print(f"Wrote {png_path}\n      {pdf_path}")


# W&B algo tags from ``phi_td_pqn_rnn_craftax`` (``config.ALG_NAME.upper()``); order matches
# ``slurm/slurm_craftax_rnn_1b_phi_td_compare.sh`` HYDRA_ALG.
PHI_TD_CRAFTAX_WANDB_ALGO_TAGS: tuple[str, ...] = (
    "PHI_TD_CATEGORICAL_RNN",
    "PHI_TD_QUANTILE_RNN",
    "PHI_TD_DIRAC_RNN",
    "PHI_TD_MOG_RNN",
    "PHI_TD_CAUCHY_RNN",
    "PHI_TD_GAMMA_RNN",
    "PHI_TD_LAPLACE_RNN",
    "PHI_TD_LOGISTIC_RNN",
)

# φ-sweep families only (no categorical / quantile heads) — e.g. one panel with **PQN** baseline.
PHI_TD_CRAFTAX_WANDB_ALGO_TAGS_NO_CAT_QT: tuple[str, ...] = (
    "PHI_TD_DIRAC_RNN",
    "PHI_TD_MOG_RNN",
    "PHI_TD_CAUCHY_RNN",
    "PHI_TD_GAMMA_RNN",
    "PHI_TD_LAPLACE_RNN",
    "PHI_TD_LOGISTIC_RNN",
)

# Fair 4-way RNN compare (``M_PARTICLES`` / ``NUM_ATOMS`` / ``NUM_QUANTILES`` aligned), e.g.
# ``slurm/slurm_craftax_rnn_1b_4alg_fair8_compare.sh``.
FAIR_4ALG_CRAFTAX_WANDB_TAGS: tuple[str, ...] = (
    "PHI_TD_CATEGORICAL_RNN",
    "PHI_TD_QUANTILE_RNN",
    "CTD_RNN",
    "QTD_RNN",
)


def build_craftax_rnn_pqn_ctd_qtd_phi_cat_qt_m8_m51_groups(
    *,
    m8_fair_experiment_tag: str,
    m51_experiment_tag_ctd_qtd: str,
    m51_experiment_tag_phi_cat_qt: str,
    pqn_m8_experiment_tag: str,
    pqn_m51_experiment_tag: str,
) -> list[dict]:
    """
    **PQN**, **CTD**, **QTD**, **φ categorical**, **φ quantile**: solid lines from the fair-8
    four-way tag (``m8_fair_experiment_tag``); dotted ``(M=51)`` lines from separate tags so
    CTD/QTD (e.g. ``Craftax-1B-5ALG``) and φ cat/qt (e.g. ``Craftax-1B-phiTD-hLap-wOmega2``) can
    resolve to different W&B experiments.     **PQN** uses ``pqn_m8_experiment_tag`` (solid) and ``pqn_m51_experiment_tag`` (dotted, ``(M=51)``).
    If both PQN tags are identical, only **one** PQN curve is added (same W&B pool twice is not
    duplicated).
    """
    four_m8: tuple[str, ...] = (
        "CTD_RNN",
        "QTD_RNN",
        "PHI_TD_CATEGORICAL_RNN",
        "PHI_TD_QUANTILE_RNN",
    )
    groups: list[dict] = []
    for i, algo_tag in enumerate(four_m8):
        base = _legend_for_algo_tag(algo_tag)
        groups.append(
            {
                "name": f"fiveway_m8_{i}_{algo_tag.lower()}",
                "experiment_tag": m8_fair_experiment_tag,
                "algo_tag": algo_tag,
                "linestyle": "-",
                "label": base,
            }
        )
        m51_tag = (
            m51_experiment_tag_ctd_qtd
            if algo_tag in ("CTD_RNN", "QTD_RNN")
            else m51_experiment_tag_phi_cat_qt
        )
        groups.append(
            {
                "name": f"fiveway_m51_{i}_{algo_tag.lower()}",
                "experiment_tag": m51_tag,
                "algo_tag": algo_tag,
                "linestyle": ":",
                "label": f"{base} (M=51)",
            }
        )

    pqn_base = _legend_for_algo_tag("PQN_RNN")
    groups.append(
        {
            "name": "fiveway_m8_pqn",
            "experiment_tag": pqn_m8_experiment_tag,
            "algo_tag": "PQN_RNN",
            "linestyle": "-",
            "label": pqn_base,
        }
    )
    if pqn_m51_experiment_tag != pqn_m8_experiment_tag:
        groups.append(
            {
                "name": "fiveway_m51_pqn",
                "experiment_tag": pqn_m51_experiment_tag,
                "algo_tag": "PQN_RNN",
                "linestyle": ":",
                "label": f"{pqn_base} (M=51)",
            }
        )
    return groups


def _build_fair_compare_groups(
    *,
    experiment_tag_m8: str,
    experiment_tag_m51: str | None,
    algo_tags: tuple[str, ...],
) -> list[dict]:
    """
    One experiment tag for M=8 runs; optional second tag for M=51. Same ``algo_tag`` and color
    for each method; M=51 curves use a dotted linestyle and ``(M=51)`` in the legend.
    """
    groups: list[dict] = []
    for i, algo_tag in enumerate(algo_tags):
        base_legend = _legend_for_algo_tag(algo_tag)
        groups.append(
            {
                "name": f"fair_m8_{i}_{algo_tag.lower()}",
                "experiment_tag": experiment_tag_m8,
                "algo_tag": algo_tag,
                "linestyle": "-",
                "label": base_legend,
            }
        )
        if experiment_tag_m51 is not None:
            groups.append(
                {
                    "name": f"fair_m51_{i}_{algo_tag.lower()}",
                    "experiment_tag": experiment_tag_m51,
                    "algo_tag": algo_tag,
                    "linestyle": ":",
                    "label": f"{base_legend} (M=51)",
                }
            )
    return groups


def _resolve_final_report_groups(
    *,
    final_report_groups: list[dict] | None,
    final_report_experiment_tag: str | None,
    final_report_algo_tags: tuple[str, ...],
) -> list[dict]:
    if final_report_groups is not None:
        if not final_report_groups:
            raise ValueError("final_report_groups must be non-empty when provided.")
        return list(final_report_groups)
    if final_report_experiment_tag is None:
        raise ValueError(
            "With final_report=True, pass final_report_groups=[...] "
            "or final_report_experiment_tag=... (and optional final_report_algo_tags), "
            "or set final_report_composite=True with fair8 + φ-sweep tags."
        )
    out: list[dict] = []
    for i, algo_tag in enumerate(final_report_algo_tags):
        out.append(
            {
                "name": f"final_{i}_{algo_tag.lower()}",
                "experiment_tag": final_report_experiment_tag,
                "algo_tag": algo_tag,
            }
        )
    return out


def build_final_report_composite_craftax_groups(
    *,
    fair8_experiment_tag: str,
    phi_sweep_experiment_tag: str,
    pqn_baseline_experiment_tag: str | None = None,
    fair8_algo_tags: tuple[str, ...] = FAIR_4ALG_CRAFTAX_WANDB_TAGS,
    phi_sweep_algo_tags: tuple[str, ...] = PHI_TD_CRAFTAX_WANDB_ALGO_TAGS,
    fair8_legend_suffix: str = " (fair-8)",
) -> list[dict]:
    """
    Build ``groups`` for ``final_report=True``: **fair-8** sweep (CTD, QTD, φ categorical, φ
    quantile) with legend suffix so they do not collide with the same φ tags from the **φ-sweep**
    tag; optional **PQN_RNN** from a baseline tag (e.g. ``Craftax-1B-5ALG``); then all **φ-sweep**
    families (default: all eight ``PHI_TD_*_RNN`` from ``phi_sweep_experiment_tag``).

    Pass the result as ``final_report_groups=...``, or use ``plot_craftax_phi_td_with_baselines``
    with ``final_report_composite=True`` and the matching ``final_report_*_experiment_tag`` args.
    """
    groups: list[dict] = []
    for i, algo_tag in enumerate(fair8_algo_tags):
        base = _legend_for_algo_tag(algo_tag)
        groups.append(
            {
                "name": f"fr_fair8_{i}_{algo_tag.lower()}",
                "experiment_tag": fair8_experiment_tag,
                "algo_tag": algo_tag,
                "label": f"{base}{fair8_legend_suffix}",
            }
        )
    if pqn_baseline_experiment_tag:
        groups.append(
            {
                "name": "fr_pqn_baseline",
                "experiment_tag": pqn_baseline_experiment_tag,
                "algo_tag": "PQN_RNN",
            }
        )
    for j, algo_tag in enumerate(phi_sweep_algo_tags):
        groups.append(
            {
                "name": f"fr_phi_{j}_{algo_tag.lower()}",
                "experiment_tag": phi_sweep_experiment_tag,
                "algo_tag": algo_tag,
            }
        )
    return groups


def plot_craftax_phi_td_with_baselines(
    *,
    project: str,
    entity: str | None,
    baseline_experiment_tag: str = "",
    phi_experiment_tag: str = "Craftax-1B-phi-td-compare",
    baseline_algo_tags: tuple[str, ...] = ("CTD_RNN", "QTD_RNN", "PQN_RNN"),
    phi_algo_tags: tuple[str, ...] = PHI_TD_CRAFTAX_WANDB_ALGO_TAGS,
    metric: str = "returned_episode_returns",
    step_metric: str = "env_step",
    out: str = "figures/craftax_1b_rnn_phi_td_with_baselines.png",
    grid_points: int = 800,
    max_runs: int = 5000,
    smooth_window: int = 51,
    env_name_filter: str | None = None,
    final_report: bool = False,
    final_report_groups: list[dict] | None = None,
    final_report_experiment_tag: str | None = None,
    final_report_algo_tags: tuple[str, ...] = FAIR_4ALG_CRAFTAX_WANDB_TAGS,
    final_report_composite: bool = False,
    final_report_fair8_experiment_tag: str | None = None,
    final_report_phi_sweep_experiment_tag: str | None = None,
    final_report_pqn_experiment_tag: str | None = None,
    final_report_phi_sweep_algo_tags: tuple[str, ...] = PHI_TD_CRAFTAX_WANDB_ALGO_TAGS,
    fair_m8_experiment_tag: str | None = None,
    fair_m51_experiment_tag: str | None = None,
    fair_comparison_algo_tags: tuple[str, ...] = FAIR_4ALG_CRAFTAX_WANDB_TAGS,
) -> None:
    """
    One figure; **three** ways to choose curves (first match wins):

    1. **final_report** — publication-style panel: only the runs you list, typically all **M=8**.
       - **Explicit:** ``final_report_groups=[...]`` (same schema as
         ``plot_craftax_rnn_compare_multi_source``), or
       - **Single tag:** ``final_report_experiment_tag`` + ``final_report_algo_tags`` (only that
         tag’s algos — *not* multi-tag by itself), or
       - **Composite:** ``final_report_composite=True`` with
         ``final_report_fair8_experiment_tag`` + ``final_report_phi_sweep_experiment_tag`` and
         optionally ``final_report_pqn_experiment_tag`` (e.g. fair-8 four-way + PQN from 5ALG +
         full φ family sweep). Same φ tag appearing in fair8 and the φ-sweep uses a ``(fair-8)``
         legend suffix on the fair8 curves. Override φ list with ``final_report_phi_sweep_algo_tags``
         to drop families (“almost all”).

    2. **Fair M=8 [+ optional M=51]** — when ``fair_m8_experiment_tag`` is set, plot the algos in
       ``fair_comparison_algo_tags`` from that tag (solid, default legend text). If
       ``fair_m51_experiment_tag`` is also set, add the **same** algos from that tag as **dotted**
       lines with legend suffix ``(M=51)``.

    3. **Legacy φTD + baselines** — if ``fair_m8_experiment_tag`` is ``None``, build curves from
       ``phi_experiment_tag`` / ``phi_algo_tags`` and ``baseline_experiment_tag`` /
       ``baseline_algo_tags`` (requires non-empty ``baseline_experiment_tag``).

    Colors use ``plot_colors`` / ``algo_tag`` (same color for M=8 and M=51 of the same method).
    """
    if final_report:
        if final_report_groups is not None:
            if not final_report_groups:
                raise ValueError("final_report_groups must be non-empty when provided.")
            groups = list(final_report_groups)
        elif final_report_composite:
            if not final_report_fair8_experiment_tag or not final_report_phi_sweep_experiment_tag:
                raise ValueError(
                    "final_report_composite=True requires "
                    "final_report_fair8_experiment_tag and final_report_phi_sweep_experiment_tag."
                )
            groups = build_final_report_composite_craftax_groups(
                fair8_experiment_tag=final_report_fair8_experiment_tag,
                phi_sweep_experiment_tag=final_report_phi_sweep_experiment_tag,
                pqn_baseline_experiment_tag=final_report_pqn_experiment_tag,
                phi_sweep_algo_tags=final_report_phi_sweep_algo_tags,
            )
        else:
            groups = _resolve_final_report_groups(
                final_report_groups=None,
                final_report_experiment_tag=final_report_experiment_tag,
                final_report_algo_tags=final_report_algo_tags,
            )
    elif fair_m8_experiment_tag is not None:
        groups = _build_fair_compare_groups(
            experiment_tag_m8=fair_m8_experiment_tag,
            experiment_tag_m51=fair_m51_experiment_tag,
            algo_tags=fair_comparison_algo_tags,
        )
    else:
        if not str(baseline_experiment_tag).strip():
            raise ValueError(
                "Set fair_m8_experiment_tag=... for the 4-alg fair plot, "
                "or pass baseline_experiment_tag=... for the legacy φTD+baseline plot."
            )
        groups = []
        for i, algo_tag in enumerate(phi_algo_tags):
            groups.append(
                {
                    "name": f"phi_td_{i}",
                    "experiment_tag": phi_experiment_tag,
                    "algo_tag": algo_tag,
                }
            )
        for j, algo_tag in enumerate(baseline_algo_tags):
            groups.append(
                {
                    "name": f"baseline_{j}_{algo_tag.lower()}",
                    "experiment_tag": baseline_experiment_tag,
                    "algo_tag": algo_tag,
                }
            )

    plot_craftax_rnn_compare_multi_source(
        project=project,
        entity=entity,
        groups=groups,
        metric=metric,
        step_metric=step_metric,
        out=out,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        env_name_filter=env_name_filter,
    )


def main() -> None:
    #! Craftax 1B (using lambda=0.5 and aux mean loss weight=1.0)
    # plot_craftax_rnn_compare(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     experiment_tag="Craftax-1B-checkpoint",
    #     algo_tags=["MOG_PQN_RNN", "PQN_RNN", "QTD_RNN", "CTD_RNN"], #IQN-RNN
    #     metric="returned_episode_returns",
    #     step_metric="env_step",
    #     out="figures/craftax_1b_rnn_lambda0.5_auxmeanlossweight1.0_compare.png",
    #     grid_points=800,
    #     max_runs=5000,
    #     smooth_window=51,
    #     env_name_filter=None,
    # )

    #! Craftax 1B (using lambda=0.0 and aux mean loss weight=0.0)
    # plot_craftax_rnn_compare_multi_source(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     groups=[
    #         {"name": "ctd_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "CTD_RNN"},
    #         {"name": "qtd_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "QTD_RNN"},
    #         {"name": "pqn_5alg", "experiment_tag": "Craftax-1B-5ALG", "algo_tag": "PQN_RNN"},
    #         {
    #             "name": "mog_combined",
    #             "experiment_tags": [
    #                 "Craftax-RNN-CTD-MoG-WeightedCF",
    #             ],
    #             "algo_tag": "MOG_PQN_RNN",
    #         },
    #     ],
    #     metric="returned_episode_returns",
    #     step_metric="env_step",
    #     out="figures/craftax_1b_rnn_lambda0.0_auxmeanlossweight0.0_compare.png",
    #     grid_points=800,
    #     max_runs=5000,
    #     smooth_window=51,
    #     env_name_filter=None,
    # )
    
    # ! φTD (Craftax-1B-phi-td-compare) + baselines from another experiment tag.
    # plot_craftax_phi_td_with_baselines(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     phi_experiment_tag="Craftax-1B-phi-td-compare",
    #     baseline_experiment_tag="Craftax-1B-5ALG",
    #     out="figures/craftax_1b_rnn_phi_td_with_baselines_pareto_1_sampling_distribution.png",
    # )
    # plot_craftax_phi_td_with_baselines(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     phi_experiment_tag="Craftax-1B-phi-td-compare-half-laplace-sampling-distribution",
    #     baseline_experiment_tag="Craftax-1B-5ALG",
    #     out="figures/craftax_1b_rnn_phi_td_with_baselines_half_laplace_sampling_distribution.png",
    # )
    #! Panel 1: six φTD families (no categorical / quantile) + PQN (5ALG), all from φ-sweep + baseline tags.
    plot_craftax_phi_td_with_baselines(
        project="Deep-CVI-Experiments",
        entity="fatty_data",
        phi_experiment_tag="Craftax-1B-phiTD-hLap-wOmega2",
        baseline_experiment_tag="Craftax-1B-5ALG",
        phi_algo_tags=PHI_TD_CRAFTAX_WANDB_ALGO_TAGS_NO_CAT_QT,
        baseline_algo_tags=("PQN_RNN",),
        out="figures/craftax_1b_rnn_phi_td_six_families_plus_pqn_hLap_wOmega2.png",
    )
    #! Panel 2: PQN, CTD, QTD, φ categorical, φ quantile — M=8 (fair-8 tag) vs M=51 (split tags).
    plot_craftax_rnn_compare_multi_source(
        project="Deep-CVI-Experiments",
        entity="fatty_data",
        groups=build_craftax_rnn_pqn_ctd_qtd_phi_cat_qt_m8_m51_groups(
            m8_fair_experiment_tag="Craftax-1B-RNN-fair8-4alg",
            m51_experiment_tag_ctd_qtd="Craftax-1B-5ALG",
            m51_experiment_tag_phi_cat_qt="Craftax-1B-phiTD-hLap-wOmega2",
            pqn_m8_experiment_tag="Craftax-1B-5ALG",
            pqn_m51_experiment_tag="Craftax-1B-5ALG",
        ),
        metric="returned_episode_returns",
        step_metric="env_step",
        out="figures/craftax_1b_rnn_pqn_ctd_qtd_phi_cat_qt_m8_m51.png",
        grid_points=800,
        max_runs=5000,
        smooth_window=51,
        env_name_filter=None,
    )
    
    


if __name__ == "__main__":
    main()
