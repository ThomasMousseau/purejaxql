"""Shared Matplotlib style and PNG+PDF export for publication figures (NeurIPS-style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Fraction of the data y-span used as headroom (and footroom when needed). Keeps curves visible without a large empty band.
DEFAULT_CURVE_Y_MARGIN_FRAC: float = 0.03

# Align with ``plot_wandb_minatar`` episodic / CI curves (no spine trimming).
GRID_ALPHA_WANDB: float = 0.3

# Align multi-panel analysis figures (distribution + Cauchy): hidden top/right + dashed grid.
GRID_COLOR_PANEL: str = "#e1e1e1"
GRID_ALPHA_PANEL: float = 0.3


def style_axes_wandb_curve(ax: Any) -> None:
    """Grid + ``axisbelow`` — same as ``plot_wandb_minatar._draw_algo_curves_on_ax`` / Craftax RNN curves."""
    ax.grid(True, alpha=GRID_ALPHA_WANDB)
    ax.set_axisbelow(True)


def style_axes_panel(ax: Any) -> None:
    """Spines trimmed + dashed grid — distribution panels & Cauchy comparison panels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", color=GRID_COLOR_PANEL, alpha=GRID_ALPHA_PANEL)
    ax.set_axisbelow(True)


def configure_matplotlib() -> None:
    """Sans-serif, editable PDF text (type 42), consistent sizing."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans", "sans-serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def pdf_path_for_png_stem(png_stem: Path) -> Path:
    """Map ``.../figures/.../name`` → ``.../pdf/.../name.pdf``; same filename if no ``figures`` segment."""
    stem = Path(png_stem)
    parts = list(stem.parts)
    for i, seg in enumerate(parts):
        if seg == "figures":
            parts[i] = "pdf"
            pdf = Path(*parts).with_suffix(".pdf")
            pdf.parent.mkdir(parents=True, exist_ok=True)
            return pdf
    pdf = Path("pdf") / stem.with_suffix(".pdf").name
    pdf.parent.mkdir(parents=True, exist_ok=True)
    return pdf


def save_figure_png_and_pdf(
    fig: Any,
    path: str | Path,
    *,
    dpi_png: int = 150,
    dpi_pdf: int = 300,
    bbox_inches: str = "tight",
    pad_inches: float = 0.08,
    **kwargs: Any,
) -> tuple[str, str]:
    """Write raster under ``figures/...`` (or given path) and **vector PDF under ``pdf/...``** mirroring the relative path."""
    p = Path(path)
    stem = p.with_suffix("") if p.suffix.lower() in {".png", ".pdf", ".svg"} else p
    png_path = stem.with_suffix(".png")
    pdf_path = pdf_path_for_png_stem(stem)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    extra = {"bbox_inches": bbox_inches, **kwargs}
    if pad_inches is not None:
        extra["pad_inches"] = pad_inches

    fig.savefig(png_path, format="png", dpi=dpi_png, **extra)
    fig.savefig(pdf_path, format="pdf", dpi=dpi_pdf, **extra)
    return str(png_path), str(pdf_path)


def set_tight_curve_ylim(
    ax: Any,
    ymax_track: float,
    ymin_track: float,
    *,
    y_margin_frac: float = DEFAULT_CURVE_Y_MARGIN_FRAC,
    clip_bottom_at_zero: bool = True,
) -> None:
    """Small symmetric padding from span so the shaded CI and lines stay visible without excess headroom."""
    import numpy as np

    if not (np.isfinite(ymax_track) and np.isfinite(ymin_track)):
        return
    span = max(float(ymax_track - ymin_track), 1e-6)
    pad = y_margin_frac * span
    top = float(ymax_track) + pad
    if clip_bottom_at_zero and ymin_track >= 0:
        bot = max(0.0, float(ymin_track) - pad)
    else:
        bot = float(ymin_track) - pad
    ax.set_ylim(bottom=bot, top=top)


def ylabel_for_metric(metric: str) -> str:
    """Short axis label for logged W&B / analysis metrics."""
    m = {
        "charts/episodic_return": "Episode Return",
        "charts/episode_return": "Episode Return",
        "episodic_return": "Episode Return",
        "returned_episode_returns": "Episode Return",
        "charts/grad_norm": "Gradient Norm",
        "grad_norm": "Gradient Norm",
        "trunk_grad_cosine_similarity": "Cosine Similarity",
        "centered_shape_volatility": "Shape Volatility",
    }
    if metric in m:
        return m[metric]
    tail = metric.split("/")[-1]
    return tail.replace("_", " ").strip().title()


def xlabel_for_step_metric(step_metric: str) -> str:
    """Short x-axis label."""
    s = {
        "global_step": "Environment Steps",
        "env_step": "Environment Steps",
        "update_steps": "Update Steps",
        "_step": "Logging Step",
    }
    if step_metric in s:
        return s[step_metric]
    return step_metric.replace("_", " ").strip().title()
