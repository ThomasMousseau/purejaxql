#!/usr/bin/env python3
"""
Fetch Weights & Biases runs by tags and plot charts/episodic_return (mean with 95% CI per algorithm).

Requires: pip install wandb matplotlib numpy
Login once: wandb login

**Single environment** (e.g. Breakout only): pass ``env_name`` for the title; leave ``env_ids`` unset.

**MinAtar 10M sweep**: pass ``experiment_tag="MinAtar_10M"`` and ``env_ids``. Runs are filtered by
tags (experiment + one algo tag). Environment is taken from ``config.env_id`` when present; otherwise
from the W&B **run name**, which is ``{env_id}__{exp_name}__{seed}__...`` (first segment = env).

**MinAtar 10M — MoG / CQN / PQN** (``slurm/slurm_minatar_10m_pqn_compare.sh``): use
``experiment_tag="MinAtar_10M_PQN"`` and algo tags ``MoG``, ``CQN``, ``PQN`` — the canonical
MoG trainer — :func:`plot_minatar_10m_mog_cqn_pqn`. (Older MoG-only ablations may use ``MoG-PQN`` /
``MoG-PQN-stab`` tags on other figures.)

**Legacy four-way PQN compare** (MoG-PQN-stab / MoG-PQN / CQN / PQN): call
:func:`plot_minatar_10m_benchmark_and_pqn_compare` (older Slurm layout),
plus :func:`plot_minatar_10m_all_algos_combined` for **all** benchmark + PQN curves on **one** figure
(four env panels; default output ``figures/minatar_10m_episodic_return_all_algos.png``).

**Vmap’d seeds (one W&B run):** tag runs with ``multi_seed`` and log ``seed_i/charts/episodic_return``
(``WANDB_LOG_ALL_SEEDS`` in trainers). :func:`curves_from_wandb_run` expands those into
multiple curves so mean ± 95% CI matches multi-run aggregation.

**Phi-TD MinAtar 10M** (``slurm/slurm_minatar_phi_families.sh``,
``slurm/slurm_minatar_phi_mog_gamma_laplace_logistic.sh``): use
:func:`plot_minatar_10m_phi_td_families` and :func:`plot_minatar_10m_phi_td_mog_gamma_laplace_logistic`
with the experiment tags and algo tags emitted by those scripts (plus ``phi_family_compare`` /
``phi_mog_gamma_laplace_logistic`` on runs).

**φTD-MoG vs MoG-PQN weighted CF (Exp 1):** train with
``slurm/slurm_minatar_phi_td_mog_mimic_weighted_cf_exp1.sh`` and compare to historical
``MinAtar_20M_Td_Lambda_WeightedCF_MoG`` using :func:`plot_minatar_20m_exp1_mog_weighted_cf_vs_phi_td_mimic`.
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import numpy as np
from plot_colors import algo_color, legend_label_for_wandb_algo_tag

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("Please install matplotlib: pip install matplotlib") from e

try:
    import wandb
except ImportError as e:
    raise SystemExit("Please install wandb: pip install wandb") from e

from paper_plots import (
    DEFAULT_CURVE_Y_MARGIN_FRAC,
    configure_matplotlib,
    save_figure_png_and_pdf,
    style_axes_wandb_curve,
    xlabel_for_step_metric,
    ylabel_for_metric,
)

configure_matplotlib()

def _algo_colors(algo_tags: list[str]) -> list[str]:
    """One canonical hex per W&B algo tag (see ``plot_colors.algo_color``)."""
    return [algo_color(t) for t in algo_tags]


_LEGACY_ALGO_LEGEND: dict[str, str] = {
    "dqn": "DQN",
    "C51": "C51",
    "MoG": "MoG",
    "FFT": "FFT",
    "QR-DQN": "QR-DQN",
    "IQN": "IQN",
    "FQF": "FQF",
    "MoG-PQN-stab": "MoG-PQN (stab.)",
    "MoG-PQN": "MoG-PQN",
    "CQN": "CQN",
    "PQN": "PQN",
    "PQN_RNN": "PQN-RNN",
    "MOG_PQN_RNN": "MoG-PQN-RNN",
    "HALF_LAPLACIAN": "Half-Laplacian",
    "UNIFORM": "Uniform",
    "HALF_GAUSSIAN": "Half-Gaussian",
    "CTD": "CTD",
    "QTD": "QTD",
}


def _legend_for_algo_tag(algo_tag: str, *, phi_families_compare: bool = False) -> str:
    """Legend text: φTD variants use mathtext (``plot_colors``)."""
    lab = legend_label_for_wandb_algo_tag(algo_tag, phi_families_compare=phi_families_compare)
    if lab != algo_tag:
        return lab
    return _LEGACY_ALGO_LEGEND.get(algo_tag, algo_tag)


def _wandb_path(entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project


def _parse_env_id_from_run_name(name: str | None) -> str | None:
    """Training uses ``wandb.init(..., name=f"{env_id}__{exp_name}__{seed}__{time}")`` — env is the first segment."""
    if not name:
        return None
    s = str(name).strip()
    if "__" not in s:
        return None
    return s.split("__", 1)[0].strip() or None


def _get_run_env_id(run, *, use_run_name: bool = True) -> str | None:
    """Prefer ``config.env_id``; if missing, parse from ``run.name`` (see ``_parse_env_id_from_run_name``)."""
    cfg = getattr(run, "config", None)
    if cfg is not None and hasattr(cfg, "get"):
        v = cfg.get("env_id")
        if v is not None and str(v).strip() != "":
            return str(v)
    if use_run_name:
        return _parse_env_id_from_run_name(getattr(run, "name", None))
    return None


def _run_matches_required(run, required: list[str]) -> bool:
    tags = set(run.tags or [])
    return all(t in tags for t in required)


def _algo_group(run, algo_tags: list[str]) -> str | None:
    tags = set(run.tags or [])
    found = [a for a in algo_tags if a in tags]
    if len(found) == 1:
        return found[0]
    if len(found) == 0:
        return None
    raise ValueError(f"Run {run.id} has multiple algo tags {found}; use unique algo tags per run.")


def _load_series(run, metric: str, step_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (steps, values) sorted by step; drops NaNs."""
    df = None
    try:
        # Downsample on the server side to keep WANDB reads bounded.
        df = run.history(keys=[metric, step_key], pandas=True, samples=1200)
    except Exception:
        df = None
    if df is None or getattr(df, "empty", True):
        try:
            # Fallback still sampled to avoid expensive full-history pulls.
            df = run.history(pandas=True, samples=1200)
        except Exception:
            return None
    if df is None or df.empty or step_key not in df.columns or metric not in df.columns:
        return None
    work = df[[step_key, metric]].dropna()
    if work.empty:
        return None
    steps = work[step_key].to_numpy(dtype=np.float64)
    vals = work[metric].to_numpy(dtype=np.float64)
    order = np.argsort(steps)
    return steps[order], vals[order]


def _unique_ordered(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _metric_candidates(metric: str) -> list[str]:
    """Return ordered fallbacks for reward/return metrics."""
    # For MinAtar plots we expect and require the exact metric key.
    # Do not silently fall back when charts/episodic_return is requested.
    if metric == "charts/episodic_return":
        return [metric]
    cands = [metric]
    if metric.startswith("charts/") and len(metric) > len("charts/"):
        # Some runs log metrics without the charts/ prefix (e.g., grad_norm).
        cands.append(metric.split("/", 1)[1])
    if metric == "returned_episode_returns":
        cands.extend(["charts/episodic_return", "charts/episode_return", "episodic_return"])
    return _unique_ordered(cands)


def _step_candidates(step_metric: str) -> list[str]:
    """Return ordered fallbacks for x-axis step metrics."""
    cands = [step_metric]
    if step_metric == "global_step":
        cands.extend(["env_step", "update_steps", "_step"])
    elif step_metric == "env_step":
        cands.extend(["global_step", "update_steps", "_step"])
    elif step_metric == "update_steps":
        cands.extend(["env_step", "global_step", "_step"])
    else:
        cands.extend(["global_step", "env_step", "update_steps", "_step"])
    return _unique_ordered(cands)


def _load_series_any(
    run, *, metric_candidates: list[str], step_candidates: list[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Try metric/step combinations in order; return first available series.

    Performance note: fetch candidate columns once from W&B history instead of
    issuing one API call per (metric, step) pair.
    """
    uniq_metrics = _unique_ordered(metric_candidates)
    uniq_steps = _unique_ordered(step_candidates)
    all_cols = _unique_ordered([*uniq_metrics, *uniq_steps])

    df = None
    try:
        # Single targeted request per run/seed, sampled for responsiveness.
        df = run.history(keys=all_cols, pandas=True, samples=1200)
    except Exception:
        df = None
    if df is None or getattr(df, "empty", True):
        try:
            # Keep compatibility with runs where keys(...) misses sparse columns.
            df = run.history(pandas=True, samples=1200)
        except Exception:
            df = None
    if df is None or getattr(df, "empty", True):
        return None

    cols = set(df.columns)
    for m in metric_candidates:
        if m not in cols:
            continue
        for s in step_candidates:
            if s not in cols:
                continue
            work = df[[s, m]].dropna()
            if work.empty:
                continue
            steps = work[s].to_numpy(dtype=np.float64)
            vals = work[m].to_numpy(dtype=np.float64)
            order = np.argsort(steps)
            return steps[order], vals[order]

    return None


def _num_seeds_from_run_config(run) -> int:
    """Read ``NUM_SEEDS`` from flattened W&B config (purejaxql merges ``alg`` into top-level config)."""
    cfg = getattr(run, "config", None)
    if cfg is None:
        return 5
    if hasattr(cfg, "get"):
        v = cfg.get("NUM_SEEDS")
        if v is not None:
            return max(1, int(v))
        alg = cfg.get("alg")
        if isinstance(alg, dict) and alg.get("NUM_SEEDS") is not None:
            return max(1, int(alg["NUM_SEEDS"]))
    return 5


def curves_from_wandb_run(
    run,
    *,
    metric: str,
    step_metric: str,
    multi_seed_tag: str = "multi_seed",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return one (steps, values) series per seed, or one series for a normal run.

    If the run is tagged with ``multi_seed_tag`` (e.g. ``multi_seed``), loads
    ``seed_i/{metric}`` vs ``seed_i/{step_metric}`` for ``i = 1 .. NUM_SEEDS`` — the same
    layout logged by purejaxql when ``WANDB_LOG_ALL_SEEDS`` is true and ``NUM_SEEDS`` > 1.

    Otherwise loads a single ``metric`` / ``step_metric`` series (same as :func:`_load_series`).
    """
    tags = set(run.tags or [])
    metric_cands = _metric_candidates(metric)
    step_cands = _step_candidates(step_metric)
    if multi_seed_tag not in tags:
        s = _load_series_any(run, metric_candidates=metric_cands, step_candidates=step_cands)
        return [s] if s is not None else []

    n = _num_seeds_from_run_config(run)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, n + 1):
        mkeys = [f"seed_{i}/{m}" for m in metric_cands]
        skeys = [f"seed_{i}/{s}" for s in step_cands]
        s = _load_series_any(run, metric_candidates=mkeys, step_candidates=skeys)
        if s is not None:
            out.append(s)
    return out


def _interp_on_grid(
    curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray
) -> np.ndarray:
    """Each curve (s, v); returns array shape (n_curves, len(grid)) with linear interp and nan outside."""
    out = np.full((len(curves), len(grid)), np.nan, dtype=np.float64)
    for i, (s, v) in enumerate(curves):
        if len(s) == 0:
            continue
        mask = np.isfinite(s) & np.isfinite(v)
        s, v = s[mask], v[mask]
        if len(s) < 2:
            continue
        out[i] = np.interp(grid, s, v, left=np.nan, right=np.nan)
    return out


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average (edge-padded). ``window`` is coerced to an odd length ≤ ``len(y)``."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if window <= 1 or n < 2:
        return y
    w = max(3, int(window) | 1)
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w < 3:
        return y
    bad = ~np.isfinite(y)
    if np.any(bad) and np.any(~bad):
        idx = np.arange(n)
        y = y.copy()
        y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(yp, kernel, mode="valid")


def _pretty_metric_label(metric: str) -> str:
    """Y-axis label (short, publication-style)."""
    metric_label_map: dict[str, str] = {
        "charts/episodic_return": "Episode Return",
        "charts/episode_return": "Episode Return",
        "episodic_return": "Episode Return",
        "returned_episode_returns": "Episode Return",
        "charts/grad_norm": "Gradient Norm",
        "grad_norm": "Gradient Norm",
        "trunk_grad_cosine_similarity": "Cosine Similarity",
        "trunk_td_lambda_grad_norm": "TD(lambda) Gradient Norm",
        "trunk_dist_grad_norm": "Distributional Gradient Norm",
        "centered_shape_volatility": "Shape Volatility",
        "aux_td_lambda_loss_for_grad_probe": "Aux TD(lambda) Loss",
        "dist_loss_for_grad_probe": "Distributional Loss",
    }
    if metric in metric_label_map:
        return metric_label_map[metric]
    return ylabel_for_metric(metric)


def _pretty_step_label(step_metric: str) -> str:
    return xlabel_for_step_metric(step_metric)


def _pretty_env_title(env_id: str) -> str:
    if env_id.endswith("-MinAtar"):
        return env_id.replace("-MinAtar", " MinAtar").replace("-", " ")
    return env_id.replace("-", " ").replace("_", " ").title()


def _draw_algo_curves_on_ax(
    ax,
    by_algo: dict[str, list],
    algo_tags: list[str],
    colors: list[str],
    *,
    grid_points: int,
    smooth_window: int,
    step_metric: str,
    metric: str,
    show_ylabel: bool,
    autoscale_y: bool = False,
    y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    y_bottom: float | None = 0.0,
    phi_td_families_compare_legend: bool = False,
) -> None:
    """If ``autoscale_y``, y-limits use a small pad (``y_top_margin`` × span of the band)."""
    ymax_track = -np.inf
    ymin_track = np.inf

    for idx, algo_tag in enumerate(algo_tags):
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
        # 95% normal-approx confidence interval: mean ± 1.96 * SEM.
        spread = 1.96 * (std / np.sqrt(np.maximum(n_per_step.astype(np.float64), 1.0)))
        spread[~std_valid] = np.nan
        if smooth_window and smooth_window > 1:
            mean = _smooth_1d(mean, smooth_window)
            spread = _smooth_1d(spread, smooth_window)
            # Keep variance hidden where fewer than 2 runs contribute, even after smoothing.
            spread[~std_valid] = np.nan
        upper = mean + spread
        lower = mean - spread
        ymax_track = max(ymax_track, float(np.nanmax(upper)))
        ymin_track = min(ymin_track, float(np.nanmin(lower)))

        label = _legend_for_algo_tag(algo_tag, phi_families_compare=phi_td_families_compare_legend)
        c = colors[idx % len(colors)]
        ax.plot(grid, mean, color=c, linewidth=2.0, label=label)
        ax.fill_between(grid, lower, upper, color=c, alpha=0.2)

    ax.set_xlabel(_pretty_step_label(step_metric))
    if show_ylabel:
        ax.set_ylabel(_pretty_metric_label(metric))
    leg_fs = 6 if len(algo_tags) > 6 else 8
    style_axes_wandb_curve(ax)
    ax.legend(fontsize=leg_fs, loc="best")

    if autoscale_y and np.isfinite(ymax_track) and ymax_track > 0:
        span = max(float(ymax_track - ymin_track), 1e-6)
        pad = y_top_margin * span
        top = float(ymax_track) + pad
        if y_bottom is not None:
            bot = float(y_bottom)
        elif np.isfinite(ymin_track) and ymin_track < 0:
            bot = float(ymin_track) - pad
        else:
            bot = max(0.0, float(ymin_track) - pad) if np.isfinite(ymin_track) else 0.0
        ax.set_ylim(bottom=bot, top=top)


def _truncate_curve_series(
    series: tuple[np.ndarray, np.ndarray],
    max_step: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only points with step ≤ ``max_step`` (inclusive) for fair comparison (e.g. 10M vs 20M runs)."""
    if max_step is None or not np.isfinite(float(max_step)):
        return series
    steps, vals = series
    if len(steps) == 0:
        return series
    cap = float(max_step)
    mask = steps <= cap
    if not np.any(mask):
        return series
    return steps[mask], vals[mask]


def _unpack_extra_curve_spec(
    spec: tuple[list[str], str] | tuple[list[str], str, float | None],
) -> tuple[list[str], str, float | None]:
    if len(spec) == 3:
        a, b, c = spec
        return a, b, c
    a, b = spec
    return a, b, None


def _curve_mean_ci_on_grid(
    curves: list[tuple[np.ndarray, np.ndarray]],
    *,
    grid_points: int,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute (grid, mean, lower, upper) from raw seed/run curves."""
    if not curves:
        return None
    all_steps = np.concatenate([c[0] for c in curves])
    s_min, s_max = np.nanmin(all_steps), np.nanmax(all_steps)
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max <= s_min:
        return None
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
    return grid, mean, mean - spread, mean + spread


def plot_episodic_return(
    *,
    project: str,
    entity: str | None = None,
    required_tag: list[str] | None = None,
    algo_tags: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    out: str = "wandb_episodic_return.png",
    grid_points: int = 800,
    max_runs: int = 500,
    smooth_window: int = 41,
    env_name: str = "Breakout MinAtar",
    experiment_tag: str | None = None,
    env_ids: list[str] | None = None,
    use_run_name_for_env: bool = True,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
    phi_td_families_compare_legend: bool = False,
    extra_curve_specs: list[tuple[list[str], str] | tuple[list[str], str, float | None]] | None = None,
) -> None:
    """Plot mean ± 95% CI episodic return from W&B.

    ``extra_curve_specs`` adds curves whose runs match **all** tags in each tuple's first element,
    grouped under the W&B algo tag given by the second element (e.g. pull **PQN** from
    ``MinAtar_20M_Td_Lambda`` without requiring the primary ``experiment_tag``). An optional
    third element caps the x-axis (``step_metric``): only points with step ≤ that value are kept.

    - **Legacy (single panel):** leave ``env_ids`` as ``None``. Optionally set ``experiment_tag`` to filter runs.
    - **One env, filtered by id:** ``env_ids=[\"Breakout-MinAtar\"]`` and ``experiment_tag`` if needed.
    - **Five-env grid:** pass ``experiment_tag`` (e.g. ``\"MinAtar_10M\"``) and ``env_ids`` with 2+ entries;
      one subplot per env, **independent y-axes** (not shared). Y-limits add a small pad
      (``multi_env_y_top_margin`` × span of the data in each panel).

    If ``config.env_id`` is missing on older runs, set ``use_run_name_for_env=True`` (default) to recover
    env from ``run.name``.

    Runs tagged with ``multi_seed_tag`` (default ``"multi_seed"``) use :func:`curves_from_wandb_run`
    to load ``seed_i/*`` metrics so one W&B run contributes multiple seed curves for mean ± 95% CI.

    Set ``phi_td_families_compare_legend=True`` only for :func:`plot_minatar_10m_phi_td_families` so
    Dirac / Categorical / Quantile legends include the ``$(F_{\cdot})$`` suffix.
    """
    if required_tag is None:
        required_tag = []
    if algo_tags is None:
        algo_tags = ["MoG", "dqn", "C51", "QR-DQN", "IQN", "FQF"]

    required_effective = list(required_tag)
    if experiment_tag:
        required_effective.append(experiment_tag)

    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

    colors = _algo_colors(algo_tags)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    # ----- Multi-env grid (e.g. MinAtar_10M × 5 games) -----
    if env_ids is not None and len(env_ids) > 1:
        by_env_algo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for run in runs:
            if not _run_matches_required(run, required_effective):
                continue
            eid = _get_run_env_id(run, use_run_name=use_run_name_for_env)
            if eid is None or eid not in env_ids:
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
                by_env_algo[eid][g].append(series)

        if extra_curve_specs:
            for spec in extra_curve_specs:
                req_tags, algo_key, max_step = _unpack_extra_curve_spec(spec)
                for run in runs:
                    if not _run_matches_required(run, req_tags):
                        continue
                    eid = _get_run_env_id(run, use_run_name=use_run_name_for_env)
                    if eid is None or eid not in env_ids:
                        continue
                    for series in curves_from_wandb_run(
                        run,
                        metric=metric,
                        step_metric=step_metric,
                        multi_seed_tag=multi_seed_tag,
                    ):
                        by_env_algo[eid][algo_key].append(
                            _truncate_curve_series(series, max_step)
                        )

        if not by_env_algo:
            raise RuntimeError(
                "No runs matched. Check experiment_tag, env_ids, algo tags, and env (config.env_id or run.name)."
            )

        n = len(env_ids)
        fig_w = max(14, 3.2 * n)
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2), sharey=False, squeeze=False)
        ax_flat = axes[0]

        for j, eid in enumerate(env_ids):
            ax = ax_flat[j]
            by_algo = by_env_algo.get(eid, {})
            if not any(len(by_algo.get(t, [])) > 0 for t in algo_tags):
                ax.text(0.5, 0.5, "No runs", ha="center", va="center", transform=ax.transAxes)
            else:
                _draw_algo_curves_on_ax(
                    ax,
                    dict(by_algo),
                    algo_tags,
                    colors,
                    grid_points=grid_points,
                    smooth_window=smooth_window,
                    step_metric=step_metric,
                    metric=metric,
                    show_ylabel=(j == 0),
                    autoscale_y=True,
                    y_top_margin=multi_env_y_top_margin,
                    y_bottom=0.0,
                    phi_td_families_compare_legend=phi_td_families_compare_legend,
                )
            ax.set_title(_pretty_env_title(eid), fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 1])
        png_path, pdf_path = save_figure_png_and_pdf(fig, out, dpi_png=150, dpi_pdf=300)
        plt.close(fig)
        print(f"Wrote {png_path}\n      {pdf_path}")
        return

    # ----- Single panel -----
    by_algo: dict[str, list] = defaultdict(list)
    for run in runs:
        if not _run_matches_required(run, required_effective):
            continue
        if env_ids is not None and len(env_ids) == 1:
            if _get_run_env_id(run, use_run_name=use_run_name_for_env) != env_ids[0]:
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

    if extra_curve_specs:
        for spec in extra_curve_specs:
            req_tags, algo_key, max_step = _unpack_extra_curve_spec(spec)
            for run in runs:
                if not _run_matches_required(run, req_tags):
                    continue
                if env_ids is not None and len(env_ids) == 1:
                    if _get_run_env_id(run, use_run_name=use_run_name_for_env) != env_ids[0]:
                        continue
                for series in curves_from_wandb_run(
                    run,
                    metric=metric,
                    step_metric=step_metric,
                    multi_seed_tag=multi_seed_tag,
                ):
                    by_algo[algo_key].append(_truncate_curve_series(series, max_step))

    if not by_algo:
        raise RuntimeError(
            "No runs matched. Check entity, project, experiment_tag, required_tag, env_ids, and algo_tags."
        )

    plt.figure(figsize=(10, 6))
    _draw_algo_curves_on_ax(
        plt.gca(),
        dict(by_algo),
        algo_tags,
        colors,
        grid_points=grid_points,
        smooth_window=smooth_window,
        step_metric=step_metric,
        metric=metric,
        show_ylabel=True,
        autoscale_y=True,
        y_top_margin=multi_env_y_top_margin,
        y_bottom=0.0,
        phi_td_families_compare_legend=phi_td_families_compare_legend,
    )
    plt.title(env_name, fontsize=12)
    plt.tight_layout()
    png_path, pdf_path = save_figure_png_and_pdf(fig=plt.gcf(), path=out, dpi_png=150, dpi_pdf=300)
    plt.close()
    print(f"Wrote {png_path}\n      {pdf_path}")


def plot_minatar_20m_td_lambda_aux_3exp(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str | list[str] = "MinAtar_20M_Td_Lambda",
    out: str = "figures/minatar_20m_td_lambda_aux_3exp.png",
    env_ids: list[str] | None = None,
    algo_tags: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 600,
    max_runs_per_group: int = 2,
    smooth_window: int = 41,
    use_run_name_for_env: bool = True,
    multi_seed_tag: str = "multi_seed",
    weighted_cf_as_true_mog: bool = False,
    showing_exp: list[int] | None = None,
    showing_env: list[int] | None = None,
    display_titles: bool = True,
    panel_layout: str = "horizontal",
) -> None:
    """Plot a 4x3 grid (rows=envs, cols=experiments) with per-row shared x/y limits.

    Experiments are selected by tags:
    - ``LAMBDA-0.0`` + ``AUX_MEAN_LOSS_WEIGHT-0.0``
    - ``LAMBDA-0.0`` + ``AUX_MEAN_LOSS_WEIGHT-1.0``
    - ``LAMBDA-0.65`` + ``AUX_MEAN_LOSS_WEIGHT-1.0``
    """
    if isinstance(experiment_tag, str):
        experiment_tags = [experiment_tag]
    else:
        experiment_tags = [str(t) for t in experiment_tag if str(t).strip()]
    if not experiment_tags:
        raise ValueError("experiment_tag must contain at least one non-empty tag.")
    primary_experiment_tag = experiment_tags[0]
    weighted_cf_experiment_tag = next((t for t in experiment_tags[1:] if "WeightedCF" in t), None)
    weighted_cf_algo_tag = "MoG-WeightedCF"

    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
    if algo_tags is None:
        algo_tags = ["MoG", "PQN", "QTD", "CTD"]  #IQN

    exp_specs = [
        {
            "label": "Exp 1",
            "tag_label": "LAMBDA-0.0 | AUX_MEAN_LOSS_WEIGHT-0.0",
            "required_tags": [primary_experiment_tag, "LAMBDA-0.0", "AUX_MEAN_LOSS_WEIGHT-0.0"],
        },
        {
            "label": "Exp 2",
            "tag_label": "LAMBDA-0.0 | AUX_MEAN_LOSS_WEIGHT-1.0",
            "required_tags": [primary_experiment_tag, "LAMBDA-0.0", "AUX_MEAN_LOSS_WEIGHT-1.0"],
        },
        {
            "label": "Exp 3",
            "tag_label": "LAMBDA-0.65 | AUX_MEAN_LOSS_WEIGHT-1.0",
            "required_tags": [primary_experiment_tag, "LAMBDA-0.65", "AUX_MEAN_LOSS_WEIGHT-1.0"],
        },
    ]
    if showing_exp is not None:
        exp_specs = [exp_specs[i] for i in showing_exp]
    if showing_env is not None:
        env_ids = [env_ids[i] for i in showing_env]
    exp_keys = [spec["label"] for spec in exp_specs]
    algo_tags_plot = list(algo_tags)
    if weighted_cf_experiment_tag and not weighted_cf_as_true_mog and weighted_cf_algo_tag not in algo_tags_plot:
        algo_tags_plot.append(weighted_cf_algo_tag)

    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

    by_exp_env_algo: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    run_jobs: list[tuple] = []
    group_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for run in runs:
        tags = set(run.tags or [])
        if weighted_cf_experiment_tag and "MoG" in tags and weighted_cf_experiment_tag in tags:
            algo = "MoG" if weighted_cf_as_true_mog else weighted_cf_algo_tag
        else:
            algo = _algo_group(run, algo_tags)
        if algo is None:
            continue
        if (
            weighted_cf_as_true_mog
            and weighted_cf_experiment_tag
            and algo == "MoG"
            and weighted_cf_experiment_tag not in tags
        ):
            # In "weighted as true MoG" mode, avoid mixing base MoG and weighted-MoG runs.
            continue
        env_id = _get_run_env_id(run, use_run_name=use_run_name_for_env)
        if env_id is None or env_id not in env_ids:
            continue

        matched_exps: list[str] = []
        for spec in exp_specs:
            if _run_matches_required(run, list(spec["required_tags"])):
                matched_exps.append(str(spec["label"]))

        # Exp 2 reuses PQN from Exp 1:
        # PQN does not use AUX_MEAN_LOSS_WEIGHT, so runs may only exist with AUX=0.0 tags.
        if algo == "PQN" and _run_matches_required(
            run, [primary_experiment_tag, "LAMBDA-0.0", "AUX_MEAN_LOSS_WEIGHT-0.0"]
        ):
            matched_exps.append("Exp 2")
        if weighted_cf_experiment_tag and (
            (not weighted_cf_as_true_mog and algo == weighted_cf_algo_tag)
            or (weighted_cf_as_true_mog and algo == "MoG" and weighted_cf_experiment_tag in tags)
        ):
            matched_exps = []
            for spec in exp_specs:
                weighted_required = [
                    weighted_cf_experiment_tag,
                    *(t for t in spec["required_tags"] if t != primary_experiment_tag),
                ]
                if _run_matches_required(run, weighted_required):
                    matched_exps.append(str(spec["label"]))

        matched_exps = _unique_ordered(matched_exps)
        if not matched_exps:
            continue

        accepted_exps: list[str] = []
        for exp in matched_exps:
            key = (exp, env_id, algo)
            if group_counts[key] >= max_runs_per_group:
                continue
            group_counts[key] += 1
            accepted_exps.append(exp)
        if not accepted_exps:
            continue

        run_jobs.append((run, env_id, algo, accepted_exps))

    if run_jobs:
        # Moderate parallelism helps hide WANDB latency while avoiding heavy throttling.
        workers = min(12, len(run_jobs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    curves_from_wandb_run,
                    run,
                    metric=metric,
                    step_metric=step_metric,
                    multi_seed_tag=multi_seed_tag,
                ): (env_id, algo, matched_exps)
                for run, env_id, algo, matched_exps in run_jobs
            }
            for fut in as_completed(futures):
                env_id, algo, matched_exps = futures[fut]
                try:
                    series_list = fut.result()
                except Exception:
                    continue
                for matched_exp in matched_exps:
                    for series in series_list:
                        by_exp_env_algo[matched_exp][env_id][algo].append(series)

    if not by_exp_env_algo:
        raise RuntimeError(
            "No runs matched for MinAtar 20M TD(lambda)+AUX plot. Check tags and env ids."
        )

    colors = _algo_colors(algo_tags_plot)
    if weighted_cf_experiment_tag and weighted_cf_algo_tag in algo_tags_plot:
        colors[algo_tags_plot.index(weighted_cf_algo_tag)] = "#ff0000"
    label_map = {
        "MoG": "MoG-CQN",
        weighted_cf_algo_tag: "MoG-CQN (Weighted CF)",
        "PQN": "PQN",
        "CTD": "CTD",
        "QTD": "QTD",
        "IQN": "IQN",
    }

    def _label_algo(algo: str) -> str:
        return label_map.get(algo, _legend_for_algo_tag(algo))

    stats: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    row_limits: dict[str, dict[str, float]] = {
        env_id: {
            "x_min": np.inf,
            "x_max": -np.inf,
            "y_min": np.inf,
            "y_max": -np.inf,
        }
        for env_id in env_ids
    }

    for exp_key in exp_keys:
        for env_id in env_ids:
            for algo in algo_tags_plot:
                curves = by_exp_env_algo.get(exp_key, {}).get(env_id, {}).get(algo, [])
                computed = _curve_mean_ci_on_grid(
                    curves,
                    grid_points=grid_points,
                    smooth_window=smooth_window,
                )
                if computed is None:
                    continue
                stats[exp_key][env_id][algo] = computed
                grid, _, lower, upper = computed
                if np.isfinite(np.nanmin(grid)):
                    row_limits[env_id]["x_min"] = min(row_limits[env_id]["x_min"], float(np.nanmin(grid)))
                if np.isfinite(np.nanmax(grid)):
                    row_limits[env_id]["x_max"] = max(row_limits[env_id]["x_max"], float(np.nanmax(grid)))
                if np.isfinite(np.nanmin(lower)):
                    row_limits[env_id]["y_min"] = min(row_limits[env_id]["y_min"], float(np.nanmin(lower)))
                if np.isfinite(np.nanmax(upper)):
                    row_limits[env_id]["y_max"] = max(row_limits[env_id]["y_max"], float(np.nanmax(upper)))

    has_valid_limits = any(
        np.isfinite(lims["x_min"])
        and np.isfinite(lims["x_max"])
        and np.isfinite(lims["y_min"])
        and np.isfinite(lims["y_max"])
        for lims in row_limits.values()
    )
    if not has_valid_limits:
        raise RuntimeError("Matched runs did not contain valid metric series for plotting.")

    def _title_variant_path(path: str) -> str:
        p = Path(path)
        return str(p.with_name(f"{p.stem}_with_titles{p.suffix}"))

    def _layout_dims(nr: int, nc: int, layout: str) -> tuple[int, int]:
        total = nr * nc
        if total < 2 or total > 4:
            return nr, nc
        mode = layout.strip().lower()
        if mode not in {"horizontal", "vertical"}:
            raise ValueError("panel_layout must be 'horizontal' or 'vertical'.")
        if mode == "horizontal":
            return 1, total
        return total, 1

    def _render_single(output_path: str, *, include_titles: bool) -> None:
        base_rows = len(env_ids)
        base_cols = len(exp_keys)
        n_rows, n_cols = _layout_dims(base_rows, base_cols, panel_layout)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.8 * n_cols, 3.1 * n_rows),
            squeeze=False,
        )

        flat_positions = [(r, c) for r in range(base_rows) for c in range(base_cols)]
        for panel_i, (r, c) in enumerate(flat_positions):
            if n_rows == base_rows and n_cols == base_cols:
                ax = axes[r][c]
            elif n_rows == 1:
                ax = axes[0][panel_i]
            else:
                ax = axes[panel_i][0]
            env_id = env_ids[r]
            exp_key = exp_keys[c]

            plotted_any = False
            for idx, algo in enumerate(algo_tags_plot):
                computed = stats.get(exp_key, {}).get(env_id, {}).get(algo)
                if computed is None:
                    continue
                grid, mean, lower, upper = computed
                color = colors[idx % len(colors)]
                ax.plot(
                    grid,
                    mean,
                    color=color,
                    linewidth=2.0,
                    label=_label_algo(algo),
                )
                ax.fill_between(grid, lower, upper, color=color, alpha=0.2)
                plotted_any = True

            if not plotted_any:
                ax.text(0.5, 0.5, "No runs", ha="center", va="center", transform=ax.transAxes)

            if include_titles:
                tag_label = next(spec["tag_label"] for spec in exp_specs if spec["label"] == exp_key)
                ax.set_title(f"{_pretty_env_title(env_id)}\n{exp_key}\n{tag_label}", fontsize=8)
            else:
                ax.set_title(_pretty_env_title(env_id), fontsize=9)
            if c == 0 or n_cols == 1:
                ax.set_ylabel(_pretty_metric_label(metric))
            if r == base_rows - 1 or n_rows == 1:
                ax.set_xlabel(_pretty_step_label(step_metric))
            row_lims = row_limits[env_id]
            if np.isfinite(row_lims["x_min"]) and np.isfinite(row_lims["x_max"]):
                ax.set_xlim(row_lims["x_min"], row_lims["x_max"])
            if np.isfinite(row_lims["y_min"]) and np.isfinite(row_lims["y_max"]):
                row_y_bottom = min(0.0, row_lims["y_min"])
                span = max(float(row_lims["y_max"] - row_lims["y_min"]), 1e-6)
                pad = DEFAULT_CURVE_Y_MARGIN_FRAC * span
                row_y_top = float(row_lims["y_max"]) + pad
                ax.set_ylim(row_y_bottom, row_y_top)
            style_axes_wandb_curve(ax)
            if panel_i == len(flat_positions) - 1:
                ax.legend(fontsize=7, loc="best")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 1])
        png_path, pdf_path = save_figure_png_and_pdf(fig, output_path, dpi_png=150, dpi_pdf=300)
        plt.close(fig)
        print(f"Wrote {png_path}\n      {pdf_path}")

    _render_single(out, include_titles=display_titles)
    if not display_titles:
        _render_single(_title_variant_path(out), include_titles=True)


def plot_minatar_20m_exp1_mog_weighted_cf_vs_phi_td_mimic(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    baseline_experiment_tag: str = "MinAtar_20M_Td_Lambda_WeightedCF_MoG",
    phi_mimic_experiment_tag: str = "MinAtar_20M_PhiTD_MoG_Mimic_WeightedCF",
    out: str = "figures/minatar_20m_exp1_mog_weighted_cf_vs_phi_td_mimic.png",
    env_ids: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 600,
    max_runs_per_group: int = 5,
    smooth_window: int = 41,
    use_run_name_for_env: bool = True,
    multi_seed_tag: str = "multi_seed",
    display_titles: bool = False,
    panel_layout: str = "horizontal",
) -> None:
    """Four panels (one per MinAtar game): Exp 1 only — MoG-PQN weighted CF vs φTD-MoG mimic.

    **Baseline** runs must match the historical weighted-CF sweep (``slurm_minatar_lambda_aux_sensitivity_weighted_cf_mog.sh`` Exp 1): experiment tag ``baseline_experiment_tag``, tags ``MoG``, ``WEIGHTED_CF``, ``IS_DIVIDED_BY_OMEGA_SQUARED-True``, ``LAMBDA-0.0``, ``AUX_MEAN_LOSS_WEIGHT-0.0``, ``multi_seed``.

    **φTD mimic** runs use ``phi_mimic_experiment_tag`` and must carry exactly one of the algo tags in :func:`_algo_group` — default ``PhiTD-MoG`` from ``slurm/slurm_minatar_phi_td_mog_mimic_weighted_cf_exp1.sh``.
    """
    baseline_algo_key = "MoG-WeightedCF"
    phi_algo_key = "PhiTD-MoG"
    algo_tags_plot = [baseline_algo_key, phi_algo_key]

    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]

    baseline_required = [
        baseline_experiment_tag,
        "MoG",
        "WEIGHTED_CF",
        "IS_DIVIDED_BY_OMEGA_SQUARED-True",
        "LAMBDA-0.0",
        "AUX_MEAN_LOSS_WEIGHT-0.0",
        multi_seed_tag,
    ]
    phi_required = [
        phi_mimic_experiment_tag,
        phi_algo_key,
        "WEIGHTED_CF",
        "IS_DIVIDED_BY_OMEGA_SQUARED-True",
        "LAMBDA-0.0",
        "AUX_MEAN_LOSS_WEIGHT-0.0",
        "EXP1_PHI_TD_MIMIC",
        multi_seed_tag,
    ]

    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    run_jobs: list[tuple] = []

    for run in runs:
        env_id = _get_run_env_id(run, use_run_name=use_run_name_for_env)
        if env_id is None or env_id not in env_ids:
            continue

        algo: str | None = None
        if _run_matches_required(run, baseline_required):
            algo = baseline_algo_key
        elif _run_matches_required(run, phi_required):
            algo = phi_algo_key
        else:
            continue

        key = (env_id, algo)
        if group_counts[key] >= max_runs_per_group:
            continue
        group_counts[key] += 1
        run_jobs.append((run, env_id, algo))

    if not run_jobs:
        raise RuntimeError(
            "No runs matched for Exp1 MoG weighted CF vs φTD mimic. "
            "Check W&B tags (baseline vs mimic experiment tags) and env ids."
        )

    by_env_algo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    workers = min(12, len(run_jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                curves_from_wandb_run,
                run,
                metric=metric,
                step_metric=step_metric,
                multi_seed_tag=multi_seed_tag,
            ): (env_id, algo)
            for run, env_id, algo in run_jobs
        }
        for fut in as_completed(futures):
            env_id, algo = futures[fut]
            try:
                series_list = fut.result()
            except Exception:
                continue
            for series in series_list:
                by_env_algo[env_id][algo].append(series)

    if not by_env_algo:
        raise RuntimeError("Matched runs did not yield any metric series.")

    colors = _algo_colors(algo_tags_plot)
    label_map = {
        baseline_algo_key: "MoG-CQN (Weighted CF)",
        phi_algo_key: _legend_for_algo_tag(phi_algo_key),
    }

    row_limits: dict[str, dict[str, float]] = {
        env_id: {
            "x_min": np.inf,
            "x_max": -np.inf,
            "y_min": np.inf,
            "y_max": -np.inf,
        }
        for env_id in env_ids
    }

    stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(
        dict
    )
    for env_id in env_ids:
        for algo in algo_tags_plot:
            curves = by_env_algo.get(env_id, {}).get(algo, [])
            computed = _curve_mean_ci_on_grid(
                curves,
                grid_points=grid_points,
                smooth_window=smooth_window,
            )
            if computed is None:
                continue
            stats[env_id][algo] = computed
            grid, _, lower, upper = computed
            if np.isfinite(np.nanmin(grid)):
                row_limits[env_id]["x_min"] = min(row_limits[env_id]["x_min"], float(np.nanmin(grid)))
            if np.isfinite(np.nanmax(grid)):
                row_limits[env_id]["x_max"] = max(row_limits[env_id]["x_max"], float(np.nanmax(grid)))
            if np.isfinite(np.nanmin(lower)):
                row_limits[env_id]["y_min"] = min(row_limits[env_id]["y_min"], float(np.nanmin(lower)))
            if np.isfinite(np.nanmax(upper)):
                row_limits[env_id]["y_max"] = max(row_limits[env_id]["y_max"], float(np.nanmax(upper)))

    def _title_variant_path(path: str) -> str:
        p = Path(path)
        return str(p.with_name(f"{p.stem}_with_titles{p.suffix}"))

    def _layout_dims(nr: int, nc: int, layout: str) -> tuple[int, int]:
        total = nr * nc
        if total < 2 or total > 4:
            return nr, nc
        mode = layout.strip().lower()
        if mode not in {"horizontal", "vertical"}:
            raise ValueError("panel_layout must be 'horizontal' or 'vertical'.")
        if mode == "horizontal":
            return 1, total
        return total, 1

    def _render_single(output_path: str, *, include_titles: bool) -> None:
        base_rows = len(env_ids)
        base_cols = 1
        n_rows, n_cols = _layout_dims(base_rows, base_cols, panel_layout)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.8 * n_cols, 3.1 * n_rows),
            squeeze=False,
        )

        flat_positions = [(r, c) for r in range(base_rows) for c in range(base_cols)]
        for panel_i, (r, c) in enumerate(flat_positions):
            if n_rows == base_rows and n_cols == base_cols:
                ax = axes[r][c]
            elif n_rows == 1:
                ax = axes[0][panel_i]
            else:
                ax = axes[panel_i][0]
            env_id = env_ids[r]

            plotted_any = False
            for idx, algo in enumerate(algo_tags_plot):
                computed = stats.get(env_id, {}).get(algo)
                if computed is None:
                    continue
                grid, mean, lower, upper = computed
                color = colors[idx % len(colors)]
                ax.plot(grid, mean, color=color, linewidth=2.0, label=label_map.get(algo, algo))
                ax.fill_between(grid, lower, upper, color=color, alpha=0.2)
                plotted_any = True

            if not plotted_any:
                ax.text(0.5, 0.5, "No runs", ha="center", va="center", transform=ax.transAxes)

            title_env = _pretty_env_title(env_id)
            if include_titles:
                ax.set_title(
                    f"{title_env}\nExp 1: λ=0, aux=0\nMoG weighted CF vs φTD-MoG mimic",
                    fontsize=8,
                )
            else:
                ax.set_title(title_env, fontsize=9)
            if c == 0 or n_cols == 1:
                ax.set_ylabel(_pretty_metric_label(metric))
            if r == base_rows - 1 or n_rows == 1:
                ax.set_xlabel(_pretty_step_label(step_metric))
            row_lims = row_limits[env_id]
            if np.isfinite(row_lims["x_min"]) and np.isfinite(row_lims["x_max"]):
                ax.set_xlim(row_lims["x_min"], row_lims["x_max"])
            if np.isfinite(row_lims["y_min"]) and np.isfinite(row_lims["y_max"]):
                row_y_bottom = min(0.0, row_lims["y_min"])
                span = max(float(row_lims["y_max"] - row_lims["y_min"]), 1e-6)
                pad = DEFAULT_CURVE_Y_MARGIN_FRAC * span
                row_y_top = float(row_lims["y_max"]) + pad
                ax.set_ylim(row_y_bottom, row_y_top)
            style_axes_wandb_curve(ax)
            if panel_i == len(flat_positions) - 1:
                ax.legend(fontsize=7, loc="best")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 1])
        png_path, pdf_path = save_figure_png_and_pdf(fig, output_path, dpi_png=150, dpi_pdf=300)
        plt.close(fig)
        print(f"Wrote {png_path}\n      {pdf_path}")

    _render_single(out, include_titles=display_titles)
    if not display_titles:
        _render_single(_title_variant_path(out), include_titles=True)


def plot_minatar_10m_mog_cqn_pqn(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str = "MinAtar_10M_PQN",
    out: str = "figures/minatar_10m_episodic_return_mog_cqn_pqn.png",
    env_ids: list[str] | None = None,
    include_pong_misc: bool = False,
    use_run_name_for_env: bool = True,
    algo_tags: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 2000,
    smooth_window: int = 41,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
) -> None:
    """Four panels (one per MinAtar game): **MoG**, **PQN**, **CTD**, **QTD**, **IQN** only.

    Loads runs tagged ``experiment_tag`` (default ``MinAtar_10M_PQN``) plus exactly one of
    ``MoG`` / ``PQN`` / ``CTD`` / ``QTD`` / ``IQN`` -- matching ``slurm/slurm_minatar_30m_pqn_compare.sh``.
    """
    if algo_tags is None:
        algo_tags = ["MoG", "PQN", "CTD", "QTD", "IQN"]
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
        if include_pong_misc:
            env_ids = [*env_ids, "Pong-misc"]
    plot_episodic_return(
        project=project,
        entity=entity,
        experiment_tag=experiment_tag,
        env_ids=env_ids,
        use_run_name_for_env=use_run_name_for_env,
        algo_tags=algo_tags,
        metric=metric,
        step_metric=step_metric,
        out=out,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        multi_env_y_top_margin=multi_env_y_top_margin,
        multi_seed_tag=multi_seed_tag,
    )


# Back-compat alias (older name when the sweep was tagged ``mog_pqn_lambda``).
plot_minatar_10m_mog_lambda_cqn_pqn = plot_minatar_10m_mog_cqn_pqn


def plot_minatar_10m_benchmark_and_pqn_compare(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag_benchmark: str = "MinAtar_10M",
    experiment_tag_pqn: str = "MinAtar_10M_PQN",
    out_benchmark: str = "figures/minatar_10m_episodic_return_benchmark.png",
    out_pqn: str = "figures/minatar_10m_episodic_return_pqn.png",
    out_all_algos: str = "figures/minatar_10m_episodic_return_all_algos.png",
    plot_all_algos_combined: bool = True,
    env_ids: list[str] | None = None,
    include_pong_misc: bool = False,
    use_run_name_for_env: bool = True,
    algo_tags_benchmark: list[str] | None = None,
    algo_tags_pqn: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 2000,
    smooth_window: int = 41,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
) -> None:
    """Two or three multi-env figures from W&B.

    **Panel A (benchmark):** original sweep tagged ``MinAtar_10M`` + one of
    ``MoG`` / ``dqn`` / ``C51`` / ``QR-DQN`` / ``IQN`` (same as :func:`plot_minatar_10m_grid`).

    **Panel B (legacy four-way PQN sweep):** ``MinAtar_10M_PQN`` plus one of
    ``MoG-PQN-stab``, ``MoG-PQN``, ``CQN``, ``PQN``. For the canonical **MoG / CQN / PQN** sweep, use
    :func:`plot_minatar_10m_mog_cqn_pqn` instead (tag ``MoG``).

    **Panel C (optional, ``plot_all_algos_combined=True``):** benchmark + PQN curves **together**
    (default **9** algorithms) in one figure, four env columns — :func:`plot_minatar_10m_all_algos_combined`.

    All panels use the same ``metric`` / ``step_metric`` (purejaxql trainers log
    ``charts/episodic_return`` and ``global_step`` for MinAtar).
    """
    if algo_tags_benchmark is None:
        algo_tags_benchmark = ["MoG", "dqn", "C51", "QR-DQN", "IQN"]
    if algo_tags_pqn is None:
        algo_tags_pqn = ["MoG-PQN-stab", "MoG-PQN", "CQN", "PQN"]

    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
        if include_pong_misc:
            env_ids = [*env_ids, "Pong-misc"]

    plot_episodic_return(
        project=project,
        entity=entity,
        experiment_tag=experiment_tag_benchmark,
        env_ids=env_ids,
        use_run_name_for_env=use_run_name_for_env,
        algo_tags=algo_tags_benchmark,
        metric=metric,
        step_metric=step_metric,
        out=out_benchmark,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        multi_env_y_top_margin=multi_env_y_top_margin,
        multi_seed_tag=multi_seed_tag,
    )
    plot_episodic_return(
        project=project,
        entity=entity,
        experiment_tag=experiment_tag_pqn,
        env_ids=env_ids,
        use_run_name_for_env=use_run_name_for_env,
        algo_tags=algo_tags_pqn,
        metric=metric,
        step_metric=step_metric,
        out=out_pqn,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        multi_env_y_top_margin=multi_env_y_top_margin,
        multi_seed_tag=multi_seed_tag,
    )
    if plot_all_algos_combined:
        plot_minatar_10m_all_algos_combined(
            project=project,
            entity=entity,
            experiment_tag_benchmark=experiment_tag_benchmark,
            experiment_tag_pqn=experiment_tag_pqn,
            out=out_all_algos,
            env_ids=env_ids,
            use_run_name_for_env=use_run_name_for_env,
            algo_tags_benchmark=algo_tags_benchmark,
            algo_tags_pqn=algo_tags_pqn,
            metric=metric,
            step_metric=step_metric,
            grid_points=grid_points,
            max_runs=max_runs,
            smooth_window=smooth_window,
            multi_env_y_top_margin=multi_env_y_top_margin,
            multi_seed_tag=multi_seed_tag,
        )


def plot_minatar_10m_all_algos_combined(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag_benchmark: str = "MinAtar_10M",
    experiment_tag_pqn: str = "MinAtar_10M_PQN",
    out: str = "figures/minatar_10m_episodic_return_all_algos.png",
    env_ids: list[str] | None = None,
    include_pong_misc: bool = False,
    use_run_name_for_env: bool = True,
    algo_tags_benchmark: list[str] | None = None,
    algo_tags_pqn: list[str] | None = None,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 2000,
    smooth_window: int = 41,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
) -> None:
    """One figure: **benchmark + PQN-sweep** algorithms together (default **5 + 4 = 9** curves per env).

    Loads runs tagged ``MinAtar_10M`` for ``MoG`` / ``dqn`` / ``C51`` / ``QR-DQN`` / ``IQN`` and
    runs tagged ``MinAtar_10M_PQN`` for ``MoG-PQN-stab`` / ``MoG-PQN`` / ``CQN`` / ``PQN``.
    Four columns (one per MinAtar game), independent y-scales per panel.
    """
    if algo_tags_benchmark is None:
        algo_tags_benchmark = ["MoG", "dqn", "C51", "QR-DQN", "IQN"]
    if algo_tags_pqn is None:
        algo_tags_pqn = ["MoG-PQN-stab", "MoG-PQN", "CQN", "PQN"]

    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
        if include_pong_misc:
            env_ids = [*env_ids, "Pong-misc"]

    algo_tags_all = list(algo_tags_benchmark) + list(algo_tags_pqn)
    exp_for_algo: dict[str, str] = {
        **{t: experiment_tag_benchmark for t in algo_tags_benchmark},
        **{t: experiment_tag_pqn for t in algo_tags_pqn},
    }
    colors = _algo_colors(algo_tags_all)

    api = wandb.Api()
    path = _wandb_path(entity, project)
    runs = list(islice(api.runs(path, order="-created_at"), max_runs))

    by_env_algo: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        g = _algo_group(run, algo_tags_all)
        if g is None:
            continue
        et = exp_for_algo[g]
        if et not in set(run.tags or []):
            continue
        eid = _get_run_env_id(run, use_run_name=use_run_name_for_env)
        if eid is None or eid not in env_ids:
            continue
        for series in curves_from_wandb_run(
            run,
            metric=metric,
            step_metric=step_metric,
            multi_seed_tag=multi_seed_tag,
        ):
            by_env_algo[eid][g].append(series)

    if not by_env_algo:
        raise RuntimeError(
            "No runs matched for combined plot. Check experiment tags, env_ids, and algo tags on W&B."
        )

    n = len(env_ids)
    fig_w = max(18, 3.5 * n)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.8), sharey=False, squeeze=False)
    ax_flat = axes[0]

    for j, eid in enumerate(env_ids):
        ax = ax_flat[j]
        by_algo = by_env_algo.get(eid, {})
        if not any(len(by_algo.get(t, [])) > 0 for t in algo_tags_all):
            ax.text(0.5, 0.5, "No runs", ha="center", va="center", transform=ax.transAxes)
        else:
            _draw_algo_curves_on_ax(
                ax,
                dict(by_algo),
                algo_tags_all,
                colors,
                grid_points=grid_points,
                smooth_window=smooth_window,
                step_metric=step_metric,
                metric=metric,
                show_ylabel=(j == 0),
                autoscale_y=True,
                y_top_margin=multi_env_y_top_margin,
                y_bottom=0.0,
            )
        ax.set_title(_pretty_env_title(eid), fontsize=10)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 1])
    png_path, pdf_path = save_figure_png_and_pdf(fig, out, dpi_png=150, dpi_pdf=300)
    plt.close(fig)
    print(f"Wrote {png_path}\n      {pdf_path}")


def plot_minatar_10m_grid(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    out: str = "figures/minatar_10m_episodic_return.png",
    experiment_tag: str = "MinAtar_10M",
    env_ids: list[str] | None = None,
    include_pong_misc: bool = False,
    use_run_name_for_env: bool = True,
    **kwargs,
) -> None:
    """Convenience wrapper for the MinAtar 10M multi-env figure.

    By default uses **four** MinAtar games (Asterix, Breakout, Freeway, Space Invaders). Set
    ``include_pong_misc=True`` to add ``Pong-misc`` as a fifth panel (Slurm stand-in for Seaquest).
    If ``env_ids`` is passed explicitly, ``include_pong_misc`` is ignored.
    """
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
        if include_pong_misc:
            env_ids = [*env_ids, "Pong-misc"]
    plot_episodic_return(
        project=project,
        entity=entity,
        experiment_tag=experiment_tag,
        env_ids=env_ids,
        out=out,
        use_run_name_for_env=use_run_name_for_env,
        **kwargs,
    )


def plot_minatar_sampling_distribution_ablation(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str = "MinAtar_MoG_Sampling_Distribution",
    out: str = "figures/minatar_sampling_distribution_ablation.png",
    env_ids: list[str] | None = None,
    use_run_name_for_env: bool = True,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 3000,
    smooth_window: int = 41,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
    required_tag: list[str] | None = None,
    algo_tags: list[str] | None = None,
) -> None:
    """Plot MinAtar ω sampling-distribution ablation from W&B tags.

    **MoG-PQN** (legacy ``slurm_minatar_sampling_distribution.sh`` before Phi-TD): experiment tag
    ``MinAtar_MoG_Sampling_Distribution``, required tag ``MoG``, distribution tags
    ``HALF_LAPLACIAN`` / ``UNIFORM`` / ``HALF_GAUSSIAN``.

    **Phi-TD MoG** (current script): ``MinAtar_PhiTD_MoG_Sampling_Distribution``,
    ``required_tag=["PhiTD-MoG"]``, and ``algo_tags`` including ``PARETO_1`` for the truncated Pareto
    (α=1 on ``[0.01, 1]``) variant.
    """
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
    if required_tag is None:
        required_tag = ["MoG"]
    if algo_tags is None:
        algo_tags = ["HALF_LAPLACIAN", "UNIFORM", "HALF_GAUSSIAN"]

    plot_episodic_return(
        project=project,
        entity=entity,
        required_tag=required_tag,
        algo_tags=algo_tags,
        metric=metric,
        step_metric=step_metric,
        out=out,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        experiment_tag=experiment_tag,
        env_ids=env_ids,
        use_run_name_for_env=use_run_name_for_env,
        multi_env_y_top_margin=multi_env_y_top_margin,
        multi_seed_tag=multi_seed_tag,
    )


def plot_minatar_10m_phi_td_families(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str = "MinAtar_10M_PhiTD_Families",
    out: str = "figures/minatar_10m_phi_td_families.png",
    env_ids: list[str] | None = None,
    algo_tags: list[str] | None = None,
    use_run_name_for_env: bool = True,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 3000,
    smooth_window: int = 41,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
) -> None:
    """CTD / QTD / :math:`\\varphi`TD-Dirac / Categorical / Quantile from ``slurm/slurm_minatar_phi_families.sh``.

    Legends append ``$(F_m)$``, ``$(F_{C,m})$``, ``$(F_{Q,m})$`` only here; W&B tags stay
    ``PhiTD-Fm``, ``PhiTD-FCm``, ``PhiTD-FQm``.

    Runs must include tags: ``experiment_tag``, ``multi_seed``, ``phi_family_compare``, and exactly
    one of ``CTD``, ``QTD``, ``PhiTD-Fm``, ``PhiTD-FCm``, ``PhiTD-FQm``.
    """
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
    if algo_tags is None:
        algo_tags = ["CTD", "QTD", "PhiTD-Fm", "PhiTD-FCm", "PhiTD-FQm"]

    plot_episodic_return(
        project=project,
        entity=entity,
        required_tag=["phi_family_compare"],
        algo_tags=algo_tags,
        metric=metric,
        step_metric=step_metric,
        out=out,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        experiment_tag=experiment_tag,
        env_ids=env_ids,
        use_run_name_for_env=use_run_name_for_env,
        multi_env_y_top_margin=multi_env_y_top_margin,
        multi_seed_tag=multi_seed_tag,
        phi_td_families_compare_legend=True,
    )


def plot_minatar_10m_phi_td_mog_gamma_laplace_logistic(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str = "MinAtar_10M_MoG_PhiTD_Gamma_Laplace_Logistic",
    out: str = "figures/minatar_10m_phi_td_mog_gamma_laplace_logistic.png",
    env_ids: list[str] | None = None,
    algo_tags: list[str] | None = None,
    use_run_name_for_env: bool = True,
    metric: str = "charts/episodic_return",
    step_metric: str = "global_step",
    grid_points: int = 800,
    max_runs: int = 3000,
    smooth_window: int = 41,
    multi_env_y_top_margin: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    multi_seed_tag: str = "multi_seed",
    pqn_baseline_20m_exp1: bool = True,
    pqn_baseline_experiment_tag: str = "MinAtar_20M_Td_Lambda",
    pqn_baseline_max_steps: float | None = 10_000_000,
) -> None:
    """Phi-TD mixture CFs from ``slurm/slurm_minatar_phi_mog_gamma_laplace_logistic.sh`` (default curves
    omit ``MoG-PQN`` and ``PhiTD-Logistic``; pass ``algo_tags`` to include either).

    Runs must include tags: ``experiment_tag``, ``multi_seed``, ``phi_mog_gamma_laplace_logistic``,
    and exactly one of ``PhiTD-MoG``, ``PhiTD-MoGamma``, ``PhiTD-Laplace``, ``PhiTD-Logistic`` (or
    ``MoG-PQN`` if added via ``algo_tags``).

    When ``pqn_baseline_20m_exp1`` (default), also plots **PQN** from
    ``slurm/slurm_minatar_lambda_aux_sensitivity.sh`` experiment 1: same tag set as
    :func:`plot_minatar_20m_td_lambda_aux_3exp` for Exp 1 (``pqn_baseline_experiment_tag``,
    ``LAMBDA-0.0``, ``AUX_MEAN_LOSS_WEIGHT-0.0``, ``PQN``) — canonical dark blue in
    :mod:`plot_colors`. Those runs train for 20M steps; ``pqn_baseline_max_steps`` (default 10M,
    in ``step_metric`` units) trims the PQN curve so it matches the 10M φ-TD runs.
    """
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
    phi_algo_tags = [
        "PhiTD-MoG",
        "PhiTD-MoGamma",
        "PhiTD-Laplace",
    ]
    if algo_tags is None:
        algo_tags = (["PQN", *phi_algo_tags] if pqn_baseline_20m_exp1 else phi_algo_tags)
    elif pqn_baseline_20m_exp1 and "PQN" not in algo_tags:
        algo_tags = ["PQN", *algo_tags]

    extra_curve_specs: list[tuple[list[str], str] | tuple[list[str], str, float | None]] | None = None
    if pqn_baseline_20m_exp1:
        pqn_spec: tuple[list[str], str] | tuple[list[str], str, float | None] = (
            (
                [
                    pqn_baseline_experiment_tag,
                    "LAMBDA-0.0",
                    "AUX_MEAN_LOSS_WEIGHT-0.0",
                    "PQN",
                ],
                "PQN",
            )
        )
        if pqn_baseline_max_steps is not None:
            pqn_spec = (*pqn_spec, float(pqn_baseline_max_steps))
        extra_curve_specs = [pqn_spec]

    plot_episodic_return(
        project=project,
        entity=entity,
        required_tag=["phi_mog_gamma_laplace_logistic"],
        algo_tags=algo_tags,
        metric=metric,
        step_metric=step_metric,
        out=out,
        grid_points=grid_points,
        max_runs=max_runs,
        smooth_window=smooth_window,
        experiment_tag=experiment_tag,
        env_ids=env_ids,
        use_run_name_for_env=use_run_name_for_env,
        multi_env_y_top_margin=multi_env_y_top_margin,
        multi_seed_tag=multi_seed_tag,
        extra_curve_specs=extra_curve_specs,
    )


def plot_minatar_gradient_alignment_and_volatility(
    *,
    project: str = "Deep-CVI-Experiments",
    entity: str | None = None,
    experiment_tag: str = "MinAtar_10M_Cosine_Similarity",
    env_ids: list[str] | None = None,
    use_run_name_for_env: bool = True,
    step_metric: str = "global_step",
    algo_tags: list[str] | None = None,
    max_runs: int = 2000,
    smooth_window: int = 31,
    grid_points: int = 800,
    multi_seed_tag: str = "multi_seed",
    out_prefix: str = "figures/minatar_grad_alignment",
) -> None:
    """Render one figure per diagnostic metric for MoG/CTD/QTD."""
    if env_ids is None:
        env_ids = [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ]
    if algo_tags is None:
        algo_tags = ["MoG", "CTD", "QTD"]
    metrics = [
        "trunk_grad_cosine_similarity",
        # "trunk_td_lambda_grad_norm",
        # "trunk_dist_grad_norm",
        "centered_shape_volatility",
    ]
    for m in metrics:
        safe_name = m.replace("/", "_")
        plot_episodic_return(
            project=project,
            entity=entity,
            experiment_tag=experiment_tag,
            env_ids=env_ids,
            use_run_name_for_env=use_run_name_for_env,
            algo_tags=algo_tags,
            metric=m,
            step_metric=step_metric,
            out=f"{out_prefix}_{safe_name}.png",
            grid_points=grid_points,
            max_runs=max_runs,
            smooth_window=smooth_window,
            multi_seed_tag=multi_seed_tag,
        )


if __name__ == "__main__":
    # Comment out the experiment you are *not* plotting; leave exactly one ``plot_episodic_return`` call active.
    # -------------------------------------------------------------------------
    
    # plot_minatar_gradient_alignment_and_volatility(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     experiment_tag="MinAtar_10M_Cosine_Similarity",
    #     out_prefix="figures/minatar_10m_cosine_similarity",
    #     grid_points=800,
    #     max_runs=300,
    #     smooth_window=31,
    #     multi_seed_tag="multi_seed",
    # )
    
    #! without the new mog weighted cf
    # plot_minatar_20m_td_lambda_aux_3exp(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     experiment_tag="MinAtar_20M_Td_Lambda",
    #     out="figures/minatar_20m_td_lambda_aux_3exp.png",
    # )
    
    #? UNDER THIS ARE ALL OF THE PLOTS NEEDED FOR THE FINAL PAPER
    
    # plot_minatar_sampling_distribution_ablation(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     experiment_tag="MinAtar_MoG_Sampling_Distribution",
    #     out="figures/minatar_sampling_distribution_ablation.png",
    # )
    
    # plot_minatar_sampling_distribution_ablation(
    #     experiment_tag="MinAtar_PhiTD_MoG_Sampling_Distribution",
    #     required_tag=["PhiTD-MoG"],
    #     algo_tags=["HALF_LAPLACIAN", "UNIFORM", "HALF_GAUSSIAN", "PARETO_1"],
    #     out="figures/minatar_phitd_mog_sampling_distribution.png",
    # )
    
    # #! 4 envs, 3 experiments, 6 algos
    # plot_minatar_20m_td_lambda_aux_3exp(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     experiment_tag=["MinAtar_20M_Td_Lambda", "MinAtar_20M_Td_Lambda_WeightedCF_MoG"],
    #     max_runs=300,
    #     max_runs_per_group=5,
    #     weighted_cf_as_true_mog=True,
    #     out="figures/minatar_20m_td_lambda_aux_3exp_weighted_cf_mog_episodic_return.png",
    #     display_titles=False,
    # )

    # #! 4 envs, exp 1 only (weighted CF MoG replaces MoG)
    # plot_minatar_20m_td_lambda_aux_3exp(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     experiment_tag=["MinAtar_20M_Td_Lambda", "MinAtar_20M_Td_Lambda_WeightedCF_MoG"],
    #     max_runs=300,
    #     max_runs_per_group=5,
    #     weighted_cf_as_true_mog=True,
    #     showing_exp=[0],
    #     showing_env=[0, 1, 2, 3],
    #     out="figures/minatar_20m_td_lambda_aux_exp1_weighted_cf_mog_episodic_return.png",
    #     display_titles=False,
    #     panel_layout="horizontal",
    # )

    # #! exp 3, envs: Asterix + Breakout (weighted CF MoG replaces MoG)
    # plot_minatar_20m_td_lambda_aux_3exp(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     experiment_tag=["MinAtar_20M_Td_Lambda", "MinAtar_20M_Td_Lambda_WeightedCF_MoG"],
    #     max_runs=300,
    #     max_runs_per_group=5,
    #     weighted_cf_as_true_mog=True,
    #     showing_exp=[2],
    #     showing_env=[0, 1],
    #     out="figures/minatar_20m_td_lambda_aux_exp3_asterix_breakout_weighted_cf_mog_episodic_return.png",
    #     display_titles=False,
    #     panel_layout="vertical",
    # )
    
    # #! CTD, QTD, Phi-TD (F_m / F_C,m / F_Q,m) — slurm_minatar_phi_families.sh
    # plot_minatar_10m_phi_td_families(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     out="figures/minatar_10m_phi_td_families.png",
    # )
    # #! Phi-TD MoG / Gamma / Laplace / Logistic (+ optional PQN baseline) — slurm_minatar_phi_mog_gamma_laplace_logistic.sh
    plot_minatar_10m_phi_td_mog_gamma_laplace_logistic(
        project="Deep-CVI-Experiments",
        entity="fatty_data",
        out="figures/minatar_10m_phi_td_mog_gamma_laplace_logistic.png",
    )
    
    #m#! CTD vs F_C,m - QTD vs F_Q,m - Fm (Pareto distribution, no weighted CF)
    # plot_minatar_10m_phi_td_families(
    #     project="Deep-CVI-Experiments",
    #     experiment_tag="MinAtar_10M_PhiTD_Families_v2",
    #     entity="fatty_data",
    #     out="figures/minatar_10m_phi_td_families_v2.png",
    # )
    # #! Gaussian, Laplace, Gamme (Pareto distribution, no weighted CF)
    # plot_minatar_10m_phi_td_mog_gamma_laplace_logistic(
    #     project="Deep-CVI-Experiments",
    #     experiment_tag="MinAtar_10M_MoG_PhiTD_Gamma_Laplace_Logistic_v2",
    #     entity="fatty_data",
    #     out="figures/minatar_10m_phi_td_mog_gamma_laplace_logistic_v2.png",
    # )
    
    # Exp 1 (λ=0, aux=0): baseline ``MinAtar_20M_Td_Lambda_WeightedCF_MoG`` vs φTD-MoG mimic
    # (``slurm/slurm_minatar_phi_td_mog_mimic_weighted_cf_exp1.sh``).
    # plot_minatar_20m_exp1_mog_weighted_cf_vs_phi_td_mimic(
    #     project="Deep-CVI-Experiments",
    #     entity="fatty_data",
    #     metric="charts/episodic_return",
    #     step_metric="global_step",
    #     baseline_experiment_tag="MinAtar_20M_Td_Lambda_WeightedCF_MoG",
    #     phi_mimic_experiment_tag="MinAtar_20M_PhiTD_MoG_Mimic_WeightedCF",
    #     out="figures/minatar_20m_exp1_mog_weighted_cf_vs_phi_td_mimic.png",
    #     max_runs=600,
    #     max_runs_per_group=5,
    #     display_titles=False,
    # )

