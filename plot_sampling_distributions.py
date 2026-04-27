#!/usr/bin/env python3
"""Minimal plot for omega sampling distributions used in MoG-CF."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_colors import COLOR_DARK_BLUE, COLOR_GREEN, COLOR_PURPLE


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot omega sampling distributions.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/sampling_distributions.png"),
    )
    args = parser.parse_args()
    
    omega_max = 1.0
    b = omega_max / 3.0
    sigma = 0.5

    x = np.linspace(0.0, omega_max, 800)

    y_half_lap = half_laplacian_pdf_trunc(x, b=b, omega_max=omega_max)
    y_uniform = uniform_pdf(x, omega_max=omega_max)
    y_half_gauss = half_gaussian_pdf_trunc(x, sigma=sigma, omega_max=omega_max)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    plt.figure(figsize=(6.0, 3.8))
    plt.plot(
        x,
        y_half_lap,
        color=COLOR_GREEN,
        linewidth=2.4,
        label=rf"Half-Laplacian($\mu$=0, $b$={b:.2f})",
    )
    plt.plot(
        x,
        y_uniform,
        color=COLOR_DARK_BLUE,
        linewidth=2.4,
        label=rf"Uniform($a$=0, $b$={omega_max:.2f})",
    )
    plt.plot(
        x,
        y_half_gauss,
        color=COLOR_PURPLE,
        linewidth=2.4,
        label=rf"Half-Gaussian($\mu$=0,$\sigma$={sigma:.1f})",
    )

    plt.xlim(0.0, omega_max)
    ymax = max(np.max(y_half_lap), np.max(y_uniform), np.max(y_half_gauss))
    plt.ylim(0.0, ymax * 1.08)
    plt.xlabel(r"$\omega$")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.2, linewidth=0.8)
    plt.legend(frameon=False)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
