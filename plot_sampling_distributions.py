#!/usr/bin/env python3
"""Plot ω sampling densities for MoG/φTD CF losses.

Half-Laplacian has support ``[0, \\omega_{\\max}]`` and positive density at ``\\omega=0``.
Truncated Pareto (``\\alpha=1``) has support ``[x_m, B]`` only: **no** samples in
``[0, x_m)``, with a singularity at ``x_m``. That structural mismatch (not just tail
shape) can change which CF frequencies drive gradients—relevant when comparing runs
(e.g. Craftax) under different ``OMEGA_MIN`` / ``OMEGA_MAX`` settings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_colors import (
    COLOR_DARK_BLUE,
    COLOR_GREEN,
    COLOR_PHITD_LAPLACE,
    COLOR_PHITD_LOGISTIC,
    COLOR_PHITD_MOG,
    COLOR_PHITD_MOGAMMA,
    COLOR_PURPLE,
)


def half_laplacian_pdf_trunc(x: np.ndarray, b: float, omega_max: float) -> np.ndarray:
    # Truncated exponential on [0, omega_max], matching sampling in mog_cf.py.
    z = 1.0 - np.exp(-omega_max / b)
    return np.exp(-x / b) / (b * z)


def uniform_pdf(x: np.ndarray, omega_max: float) -> np.ndarray:
    return np.where((x >= 0.0) & (x <= omega_max), 1.0 / omega_max, 0.0)


def half_gaussian_pdf_trunc(x: np.ndarray, sigma: float, omega_max: float) -> np.ndarray:
    # Truncated half-normal on [0, omega_max], default is sigma=1.
    coeff = np.sqrt(2.0 / np.pi) / sigma
    base = coeff * np.exp(-0.5 * (x / sigma) ** 2)
    z = math.erf(omega_max / (sigma * np.sqrt(2.0)))
    return base / z


def pareto1_pdf_trunc(
    x: np.ndarray, omega_min: float, omega_max: float
) -> np.ndarray:
    # Type I Pareto with α=1 and scale x_m = omega_min on [omega_min, omega_max];
    # matches ``sample_truncated_pareto_alpha_1`` in ``purejaxql.utils.cf_pareto``.
    z = 1.0 - omega_min / omega_max
    unscaled = omega_min / np.maximum(x, 1e-15) ** 2
    density = np.where((x >= omega_min) & (x <= omega_max), unscaled / z, 0.0)
    return density


def plot_pareto_truncations_vs_half_laplacian(
    output: Path,
    *,
    omega_min_pareto: float = 0.01,
    omega_max_laplacian: float = 1.0,
    pareto_bs: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
    x_right: float = 12.0,
) -> None:
    """Illustrate how raising Pareto upper bound B spreads mass toward larger ω.

    Half-Laplacian is fixed on ``[0, omega_max_laplacian]`` (same as MoG default).
    Truncated Pareto (α=1) uses ``[omega_min_pareto, B]``; larger ``B`` increases
    the fraction of draws with ω in the mid/large range (less extreme pile-up at
    ``x_m`` alone). Log-y helps compare the spike with smoother densities.
    """
    b = omega_max_laplacian / 3.0
    x = np.linspace(0.0, x_right, 2400)
    y_lap = half_laplacian_pdf_trunc(x, b=b, omega_max=omega_max_laplacian)

    pareto_colors = (
        COLOR_PHITD_MOG,
        COLOR_PHITD_MOGAMMA,
        COLOR_PHITD_LAPLACE,
        COLOR_PHITD_LOGISTIC,
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(
        x,
        np.maximum(y_lap, 1e-12),
        color=COLOR_GREEN,
        linewidth=2.4,
        label=rf"Half-Laplacian($b$={b:.2f}, trunc=${omega_max_laplacian:g}$)",
    )

    for i, B in enumerate(pareto_bs):
        if B <= omega_min_pareto:
            continue
        y_p = pareto1_pdf_trunc(x, omega_min=omega_min_pareto, omega_max=B)
        c = pareto_colors[i % len(pareto_colors)]
        ax.semilogy(
            x,
            np.maximum(y_p, 1e-12),
            color=c,
            linewidth=2.0,
            linestyle="-",
            label=rf"Pareto $\alpha$=1: $x_m$={omega_min_pareto:g}, $B$={B:g}",
        )

    ax.axvline(
        omega_max_laplacian,
        color="#888888",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9,
    )

    ax.set_xlim(0.0, x_right)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Density (log scale)")
    ax.grid(True, alpha=0.25, linewidth=0.8, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(
        r"Larger Pareto $B=\omega_{\max}$ extends support $[x_m,B]$ and shifts mass rightward"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved Pareto vs Half-Laplacian comparison to: {output}")


def _pareto_gap_shade_label() -> str:
    return r"Pareto: no mass on $[0, x_m)$"


def plot_four_sampling_distributions(
    output: Path,
    *,
    omega_max: float,
    omega_min_pareto: float,
    sigma: float = 0.5,
    n_points: int = 1200,
) -> None:
    """Two-row figure: full range (log y) + zoom near 0 to show the Pareto ``[0, x_m)`` gap."""
    b = omega_max / 3.0
    x = np.linspace(0.0, omega_max, n_points)
    y_half_lap = half_laplacian_pdf_trunc(x, b=b, omega_max=omega_max)
    y_uniform = uniform_pdf(x, omega_max=omega_max)
    y_half_gauss = half_gaussian_pdf_trunc(x, sigma=sigma, omega_max=omega_max)
    y_pareto = pareto1_pdf_trunc(
        x, omega_min=omega_min_pareto, omega_max=omega_max
    )

    zoom_right = float(
        min(
            max(0.08, 24.0 * omega_min_pareto),
            0.5 * omega_max,
            omega_max,
        )
    )
    x_z = np.linspace(0.0, zoom_right, 1000)
    y_lap_z = half_laplacian_pdf_trunc(x_z, b=b, omega_max=omega_max)
    y_uni_z = uniform_pdf(x_z, omega_max=omega_max)
    y_hg_z = half_gaussian_pdf_trunc(x_z, sigma=sigma, omega_max=omega_max)
    y_par_z = pareto1_pdf_trunc(
        x_z, omega_min=omega_min_pareto, omega_max=omega_max
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, (ax_full, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.8),
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.35},
        layout="constrained",
    )

    def shade_gap(ax, *, show_label: bool) -> None:
        ax.axvspan(
            0.0,
            omega_min_pareto,
            facecolor="#b8b8b8",
            alpha=0.38,
            zorder=0,
            label=_pareto_gap_shade_label() if show_label else None,
        )

    shade_gap(ax_full, show_label=True)
    eps_plot = 1e-15
    ax_full.semilogy(x, np.maximum(y_half_lap, eps_plot), color=COLOR_GREEN, lw=2.2, label=rf"Half-Laplacian($\mu$=0, $b$={b:.2f})")
    ax_full.semilogy(x, np.maximum(y_uniform, eps_plot), color=COLOR_DARK_BLUE, lw=2.2, label=rf"Uniform($a$=0, $b$={omega_max:g})")
    ax_full.semilogy(x, np.maximum(y_half_gauss, eps_plot), color=COLOR_PURPLE, lw=2.2, label=rf"Half-Gaussian($\mu$=0,$\sigma$={sigma:g})")
    ax_full.semilogy(x, np.maximum(y_pareto, eps_plot), color=COLOR_PHITD_MOG, lw=2.2, label=rf"Pareto($\alpha$=1, $x_m$={omega_min_pareto:g}, $B$={omega_max:g})")
    ax_full.set_xlim(0.0, omega_max)
    ax_full.set_ylabel("Density (log)")
    ax_full.set_title(r"Full support (log scale; Laplacian nonzero at $\omega\to 0^+$)")
    ax_full.grid(True, alpha=0.22, linewidth=0.8, which="both")
    ax_full.legend(frameon=False, loc="upper right", fontsize=8)

    shade_gap(ax_zoom, show_label=False)
    ax_zoom.semilogy(x_z, np.maximum(y_lap_z, eps_plot), color=COLOR_GREEN, lw=2.2, label="Half-Laplacian")
    ax_zoom.semilogy(x_z, np.maximum(y_uni_z, eps_plot), color=COLOR_DARK_BLUE, lw=2.2, label="Uniform")
    ax_zoom.semilogy(x_z, np.maximum(y_hg_z, eps_plot), color=COLOR_PURPLE, lw=2.2, label="Half-Gaussian")
    ax_zoom.semilogy(x_z, np.maximum(y_par_z, eps_plot), color=COLOR_PHITD_MOG, lw=2.2, label="Pareto")
    ax_zoom.set_xlim(0.0, zoom_right)
    ax_zoom.set_xlabel(r"$\omega$")
    ax_zoom.set_ylabel("Density (log)")
    ax_zoom.set_title(
        rf"Zoom: shaded region is unsupported by Pareto; Half-Laplacian still samples $\omega\in[0,x_m)$"
    )
    ax_zoom.grid(True, alpha=0.22, linewidth=0.8, which="both")
    ax_zoom.legend(frameon=False, loc="upper right", fontsize=7)

    fig.align_ylabels([ax_full, ax_zoom])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved plot to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot omega sampling distributions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/sampling_distributions.png"),
    )
    parser.add_argument(
        "--pareto-compare-output",
        type=Path,
        default=Path("figures/pareto_truncation_vs_half_laplacian.png"),
        help="Second figure: Pareto(α=1) with several B vs Half-Laplacian (log density).",
    )
    parser.add_argument(
        "--no-pareto-compare",
        action="store_true",
        help="Skip writing the Pareto truncation comparison figure.",
    )
    parser.add_argument(
        "--omega-max",
        type=float,
        default=1.0,
        help="Upper truncation B (Pareto) and ω_max for other laws; Craftax φTD MoG default is 1.0.",
    )
    parser.add_argument(
        "--omega-min",
        type=float,
        default=0.01,
        help="Pareto scale x_m (OMEGA_MIN); Half-Laplacian still uses [0, omega_max].",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Half-Gaussian σ.",
    )
    args = parser.parse_args()

    plot_four_sampling_distributions(
        args.output,
        omega_max=args.omega_max,
        omega_min_pareto=args.omega_min,
        sigma=args.sigma,
    )

    if not args.no_pareto_compare:
        plot_pareto_truncations_vs_half_laplacian(args.pareto_compare_output)


if __name__ == "__main__":
    main()
