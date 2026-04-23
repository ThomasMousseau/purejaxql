#!/usr/bin/env python3
"""Fetch distribution-analysis payloads from W&B and render aggregate plots."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

PAYLOAD_FILENAME = "distribution_analysis_payload.json"
DA_ORDER = ("iqn", "qr_dqn", "c51", "mog_cqn")
DA_TITLES = {"iqn": "IQN", "qr_dqn": "QR-DQN", "c51": "C51", "mog_cqn": "MoG-CQN"}
ROW_KEYS = ("phi_mse", "ks", "w1", "cvar_err", "er_err", "mean_err", "var_err", "mv_err")
ROW_LBL = (
    r"$\varphi$-MSE $\downarrow$",
    "KS $\\downarrow$",
    r"$W_1$ $\downarrow$",
    r"CVaR err $\downarrow$",
    "Entropic err $\\downarrow$",
    "Mean err $\\downarrow$",
    "Var err $\\downarrow$",
    "M-V err $\\downarrow$",
)
WIN_BG = "#D2F4D9"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
COLORS = {"truth": "#0A0A0A", "iqn": "#56B4E9", "qr_dqn": "#009E73", "c51": "#F4E4A6", "mog_cqn": "#E41A1C"}
LW = {"truth": 2.15, "iqn": 1.05, "qr_dqn": 1.08, "c51": 1.12, "mog_cqn": 1.2}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    short: str
    plot_x_min: float
    plot_x_max: float


ACTION_SPECS = (
    ActionSpec("Gaussian (baseline)", "Gauss", -6.0, 6.0),
    ActionSpec("Cauchy mixture (heavy-tailed)", "Cauchy", -12.0, 12.0),
    ActionSpec("MoG (sharp / bimodal)", "MoG", -6.0, 6.0),
    ActionSpec("Log-Normal (skewed)", "LogN", -1.0, 8.0),
)


def minimal_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", color="#E0E0E0", alpha=0.7)
    ax.set_axisbelow(True)


def _save_fig(fig: plt.Figure, pdf_path: str, *, png_dpi: int = 300) -> None:
    stem = os.path.splitext(pdf_path)[0]
    fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{stem}.png", format="png", bbox_inches="tight", pad_inches=0.08, dpi=png_dpi)


def _fmt_disp(x: float) -> str:
    return "-" if np.isnan(float(x)) else f"{float(x):.3g}"


def _smooth_curve(y: np.ndarray, passes: int = 2) -> np.ndarray:
    """Mildly smooth 1D curves to suppress inversion ringing artifacts."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 5 or passes <= 0:
        return arr
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    out = arr
    for _ in range(passes):
        padded = np.pad(out, (2, 2), mode="edge")
        out = np.convolve(padded, kernel, mode="valid")
    return out


def _plot_metric_tables(stderr: dict[str, tuple[np.ndarray, np.ndarray]], path: str) -> None:
    algos = [a for a in DA_ORDER if a in stderr]
    if not algos:
        raise ValueError("No metric data found for plotting.")
    means = np.stack([stderr[a][0] for a in algos])
    ses = np.stack([stderr[a][1] for a in algos])
    inf = np.where(np.isfinite(means), means, np.inf)
    best = np.argmin(inf, 0)
    best = np.where(~np.isfinite(means).any(0), -1, best)

    nfig = len(algos)
    fig, axes = plt.subplots(nfig, 1, figsize=(8.3, 2.2 * nfig + 0.7))
    if nfig == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.34)
    cols = [d.name for d in ACTION_SPECS]

    for i, (ax, algo) in enumerate(zip(axes, algos)):
        ax.axis("off")
        cells = []
        for r in range(len(ROW_KEYS)):
            row = []
            for c in range(len(ACTION_SPECS)):
                mv = means[i, r, c]
                sv = ses[i, r, c]
                if np.isfinite(mv) and sv > 0:
                    row.append(f"{mv:.4g}\n({sv:.3g})")
                else:
                    row.append(_fmt_disp(mv))
            cells.append(row)
        tbl = ax.table(cellText=cells, rowLabels=list(ROW_LBL), colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.08, 1.42)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col < 0:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
            if row >= 1 and col >= 0 and best[row - 1, col] == i and best[row - 1, col] >= 0:
                cell.set_facecolor(WIN_BG)
        ax.text(0.5, 1.02, DA_TITLES[algo], transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Mean ± SE over seeds", fontsize=11, y=0.985)
    _save_fig(fig, path)
    plt.close(fig)


def _plot_panels(mean_p: dict[str, np.ndarray], se_p: dict[str, np.ndarray], te: np.ndarray, xe: np.ndarray, cf_hw: float, path: str) -> None:
    cf_series = (
        ("panel_phi_true", "truth", "Truth"),
        ("panel_phi_iqn", "iqn", "IQN"),
        ("panel_phi_qr_dqn", "qr_dqn", "QR-DQN"),
        ("panel_phi_c51", "c51", "C51"),
        ("panel_phi_mog_cqn", "mog_cqn", "MoG-CQN"),
    )
    pdf_series = (
        ("panel_pdf_true", "truth", "Truth"),
        ("panel_pdf_iqn", "iqn", "IQN"),
        ("panel_pdf_qr_dqn", "qr_dqn", "QR-DQN"),
        ("panel_pdf_c51", "c51", "C51"),
        ("panel_pdf_mog_cqn", "mog_cqn", "MoG-CQN"),
    )
    cdf_series = (
        ("panel_F_true", "truth", "Truth"),
        ("panel_F_iqn", "iqn", "IQN"),
        ("panel_F_qr_dqn", "qr_dqn", "QR-DQN"),
        ("panel_F_c51", "c51", "C51"),
        ("panel_F_mog_cqn", "mog_cqn", "MoG-CQN"),
    )

    fig, axes = plt.subplots(len(ACTION_SPECS), 4, figsize=(13.5, 2.45 * len(ACTION_SPECS)))
    legend = None
    for row, dist in enumerate(ACTION_SPECS):
        axd, axcf, axp, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        hist = mean_p.get("panel_hist_samples")
        if hist is not None:
            axd.hist(
                hist[row],
                bins=72,
                range=(dist.plot_x_min, dist.plot_x_max),
                density=True,
                color="#E8E8E8",
                edgecolor="#CCC",
                lw=0.2,
            )
        if "panel_pdf_true" in mean_p:
            axd.plot(xe[xmask], mean_p["panel_pdf_true"][row][xmask], color=COLORS["truth"], lw=LW["truth"])
        minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axd.set_title("Density")

        tmask = np.abs(te) <= cf_hw
        for pkey, ckey, label in cf_series:
            if pkey not in mean_p:
                continue
            y = np.real(mean_p[pkey][row])
            ys = np.real(se_p[pkey][row])
            axcf.fill_between(te[tmask], (y - ys)[tmask], (y + ys)[tmask], color=COLORS[ckey], alpha=0.18, lw=0)
            axcf.plot(te[tmask], y[tmask], color=COLORS[ckey], lw=LW[ckey], label=label)
        minimal_style(axcf)
        axcf.set_xlim(-cf_hw, cf_hw)
        if row == 0:
            axcf.set_title(r"Re $\varphi$")

        for pkey, ckey, label in pdf_series:
            if pkey not in mean_p:
                continue
            y = _smooth_curve(mean_p[pkey][row])
            ys = _smooth_curve(se_p[pkey][row], passes=1)
            axp.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=COLORS[ckey], alpha=0.18, lw=0)
            axp.plot(xe[xmask], y[xmask], color=COLORS[ckey], lw=LW[ckey], label=label)
        minimal_style(axp)
        axp.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axp.set_title("PDF")

        for pkey, ckey, label in cdf_series:
            if pkey not in mean_p:
                continue
            y = mean_p[pkey][row]
            ys = se_p[pkey][row]
            axc.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=COLORS[ckey], alpha=0.18, lw=0)
            axc.plot(xe[xmask], y[xmask], color=COLORS[ckey], lw=LW[ckey], label=label)
        minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axc.set_title("CDF")

        axd.text(-0.34, 0.5, dist.name, transform=axd.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
        if legend is None:
            legend = axc.get_legend_handles_labels()

    if legend:
        fig.legend(*legend, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Distribution panels", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.09, wspace=0.3, hspace=0.42)
    _save_fig(fig, path)
    plt.close(fig)


def _download_run_payload(run: wandb.apis.public.Run, root_dir: str) -> dict | None:
    try:
        remote_file = run.file(PAYLOAD_FILENAME)
        downloaded = remote_file.download(root=root_dir, replace=True)
    except Exception:
        return None

    local_path = Path(getattr(downloaded, "name", Path(root_dir) / PAYLOAD_FILENAME))
    if not local_path.exists():
        fallback = Path(root_dir) / PAYLOAD_FILENAME
        if fallback.exists():
            local_path = fallback
        else:
            return None
    with local_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_payloads(payloads: list[dict]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    if not payloads:
        raise ValueError("No payloads to aggregate.")

    metric_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for algo in DA_ORDER:
        mats = []
        for payload in payloads:
            arr = payload.get("metric_means", {}).get(algo)
            if arr is not None:
                mats.append(np.asarray(arr, dtype=np.float64))
        if not mats:
            continue
        stack = np.stack(mats, axis=0)
        mean = np.nanmean(stack, axis=0)
        se = np.zeros_like(mean) if stack.shape[0] < 2 else np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
        metric_stats[algo] = (mean, se)

    t_eval = np.asarray(payloads[0]["t_eval"], dtype=np.float64)
    x_eval = np.asarray(payloads[0]["x_eval"], dtype=np.float64)
    omega_max = float(payloads[0]["omega_max"])

    panel_keys = sorted(payloads[0].get("panel", {}).keys())
    panel_mean: dict[str, np.ndarray] = {}
    panel_se: dict[str, np.ndarray] = {}
    for key in panel_keys:
        vals = []
        for payload in payloads:
            if key in payload.get("panel", {}):
                vals.append(np.asarray(payload["panel"][key], dtype=np.float64))
        if not vals:
            continue
        stack = np.stack(vals, axis=0)
        panel_mean[key] = np.nanmean(stack, axis=0)
        panel_se[key] = np.zeros_like(panel_mean[key]) if stack.shape[0] < 2 else np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])

    return metric_stats, panel_mean, panel_se, t_eval, x_eval, omega_max


def _write_metrics_csv(path: str, metric_stats: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["algorithm", "metric"] + [f"{x.short}_mean" for x in ACTION_SPECS] + [f"{x.short}_se" for x in ACTION_SPECS]
        writer.writerow(header)
        for algo in DA_ORDER:
            if algo not in metric_stats:
                continue
            mm, sm = metric_stats[algo]
            for row_idx, row_key in enumerate(ROW_KEYS):
                writer.writerow([algo, row_key, *[float(x) for x in mm[row_idx]], *[float(x) for x in sm[row_idx]]])


def _sanitize_tag(tag: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")
    return out or "distribution_analysis"


def plot_distribution_analysis(experiment_tag: str, *, project: str, entity: str | None, figures_dir: str) -> dict[str, str]:
    api = wandb.Api()
    final_entity = entity or os.environ.get("WANDB_ENTITY") or getattr(api, "default_entity", None)
    if not final_entity:
        raise ValueError("Missing W&B entity. Set --entity or WANDB_ENTITY.")

    runs = api.runs(
        f"{final_entity}/{project}",
        filters={"tags": {"$in": [experiment_tag]}, "state": {"$in": ["finished", "crashed", "failed"]}},
    )
    payloads = []
    with tempfile.TemporaryDirectory(prefix="dist_analysis_payloads_") as tmp_dir:
        for run in runs:
            payload = _download_run_payload(run, tmp_dir)
            if payload is not None and payload.get("experiment_tag") == experiment_tag:
                payloads.append(payload)

    if not payloads:
        raise FileNotFoundError(
            f"No runs found with payload file '{PAYLOAD_FILENAME}' for experiment tag '{experiment_tag}'."
        )

    metric_stats, panel_mean, panel_se, t_eval, x_eval, omega_max = _aggregate_payloads(payloads)
    os.makedirs(figures_dir, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{_sanitize_tag(experiment_tag)}_{timestamp}"
    metrics_pdf = os.path.join(figures_dir, f"distribution_analysis_metrics_combined_{base}.pdf")
    panels_pdf = os.path.join(figures_dir, f"distribution_analysis_panels_{base}.pdf")
    metrics_csv = os.path.join(figures_dir, f"distribution_analysis_metrics_combined_{base}.csv")

    _plot_metric_tables(metric_stats, metrics_pdf)
    _plot_panels(panel_mean, panel_se, t_eval, x_eval, omega_max, panels_pdf)
    _write_metrics_csv(metrics_csv, metric_stats)

    return {
        "num_runs": str(len(payloads)),
        "metrics_plot": os.path.splitext(metrics_pdf)[0] + ".pdf/.png",
        "panels_plot": os.path.splitext(panels_pdf)[0] + ".pdf/.png",
        "metrics_csv": metrics_csv,
    }


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-tag", default="DistAnalysis")
    p.add_argument("--project", default="Deep-CVI-Experiments")
    p.add_argument("--entity", default="fatty_data")
    p.add_argument("--figures-dir", default="figures/distribution_analysis")
    return p.parse_args(args)


def main(args=None):
    parsed = parse_args(args)
    out = plot_distribution_analysis(
        parsed.experiment_tag,
        project=parsed.project,
        entity=parsed.entity,
        figures_dir=parsed.figures_dir,
    )
    print(f"Runs aggregated: {out['num_runs']}")
    print(f"Metrics plot: {out['metrics_plot']}")
    print(f"Panels plot: {out['panels_plot']}")
    print(f"Metrics CSV: {out['metrics_csv']}")


if __name__ == "__main__":
    main()
