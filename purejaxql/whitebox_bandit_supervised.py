"""
White-box contextual-bandit benchmark: distributional algorithms on the same offline data.

Algorithms (shared trunk width/depth; fair defaults: ``num_atoms == num_tau`` for C51 vs IQN):
**IQN** (random implicit quantile levels), **QR-DQN** (fixed uniform τ grid, Dabney et al.),
**C51** (categorical atoms + Bellman projection, Bellemare et al.), optional **CQN**
(parametric CF in frequency), **MoG-CQN** (Gaussian mixture + analytic CF/CDF).

Continuous state ``x ~ N(0, I_d)`` (default ``d=5``); four discrete actions with varied
reward geometries (Gaussian, Cauchy mixture, MoG, Log-Normal). Training uses a white-box
1-step Bellman target ``R + gamma * Z_next`` with exact next-state sampling.

Mean / variance / mean–variance risk errors use **exact** moments where defined; Cauchy
mixture classical moments are **undefined** (``—`` / ``nan``). CVaR / entropic rows use the
wide risk grid.

**Default training export** is minimal: ``purejaxql/whitebox_bandit_run{tag}.npz`` +
msgpack params only. Use ``--full-export`` for per-seed CSVs and figures. After multi-seed
runs, aggregate tables + one LaTeX-ready CSV + PDF/PNG from NPZ::

    python -m purejaxql.whitebox_bandit_supervised --aggregate-metrics-dir figures/whitebox_bandit \\
        --aggregate-tag-stem _omega_15 --aggregate-npz-dir purejaxql

Disable CQN::

    python -m purejaxql.whitebox_bandit_supervised --no-cqn

Weights & Biases::

    WANDB_EXPERIMENT_TAG=MySweep python -m purejaxql.whitebox_bandit_supervised --wandb --append-seed-to-tag --seed 0

Cluster: ``sbatch slurm/slurm_whitebox_bandit.sh``.

Replot from NPZ (no training)::

    python -m purejaxql.whitebox_bandit_supervised --replot purejaxql/whitebox_bandit_run_TAG.npz

To tweak action distributions, edit the ``ACTION DISTRIBUTIONS`` section.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import flax.linen as nn
import flax.serialization
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training import train_state
from numpy.polynomial.hermite import hermgauss

from purejaxql.utils.mog_cf import sample_frequencies as sample_frequencies_half_laplacian

# ---------------------------------------------------------------------------
# Publication style (mirrors purejaxql/cauchy_mixture_supervised.py)
# ---------------------------------------------------------------------------
NEURIPS_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "system-ui",
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "Roboto",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
        "sans-serif",
    ],
    "mathtext.fontset": "dejavusans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": "#2E2E2E",
    "axes.labelcolor": "#2E2E2E",
    "axes.titlecolor": "#2E2E2E",
    "xtick.color": "#2E2E2E",
    "ytick.color": "#2E2E2E",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
plt.rcParams.update(NEURIPS_STYLE)

COLORS = {
    "truth": "#0A0A0A",
    "cqn": "#7C3AED",
    "iqn": "#56B4E9",
    "mog_cqn": "#E41A1C",
    "qr_dqn": "#009E73",
    "c51": "#F4E4A6",
}
LW_GT, LW_CQN, LW_IQN, LW_MOG = 2.15, 1.41, 1.05, 1.20
LW_QR, LW_C51 = 1.08, 1.12


def minimal_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", color="#E0E0E0", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save_figure_pdf_png(fig: plt.Figure, pdf_path: str, *, png_dpi: int = 300) -> None:
    stem = os.path.splitext(pdf_path)[0]
    fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{stem}.png", format="png", bbox_inches="tight", pad_inches=0.08, dpi=png_dpi)


# ---------------------------------------------------------------------------
# Numerical primitives (lifted from cauchy_mixture_supervised.py)
# ---------------------------------------------------------------------------
def _huber(u: jax.Array, kappa: float) -> jax.Array:
    abs_u = jnp.abs(u)
    quad = 0.5 * u**2
    lin = kappa * (abs_u - 0.5 * kappa)
    return jnp.where(abs_u <= kappa, quad, lin)


def quantile_huber_loss_pairwise(
    q_preds: jax.Array,  # (B, M)
    rewards: jax.Array,  # (B,)  — single sample target per row (gamma=0)
    taus: jax.Array,     # (B, M)
    kappa: float,
) -> jax.Array:
    """Per-row pairwise IQN quantile-Huber loss; mean over batch and quantile samples."""
    u = rewards[:, None] - q_preds  # (B, M)
    hub = _huber(u, kappa)
    ind = (u < 0).astype(q_preds.dtype)
    rho = jnp.abs(taus - ind) * hub / kappa
    return jnp.mean(rho)


def gil_pelaez_cdf(
    phi_positive: jax.Array,   # (T_pos,) complex
    t_pos: jax.Array,          # (T_pos,) strictly positive, increasing
    x_grid: jax.Array,         # (X,) real
) -> jax.Array:
    """
    F(x) = 1/2 - (1/pi) * integral_delta^inf Im(exp(-i t x) phi(t)) / t dt
    (small offset on Im(phi(0)) suppresses the t->0 singularity)
    """
    phi_positive = phi_positive - 1j * jnp.imag(phi_positive[0])
    x = x_grid[:, None]   # (X, 1)
    t = t_pos[None, :]    # (1, T_pos)
    integrand = jnp.imag(jnp.exp(-1j * t * x) * phi_positive[None, :]) / t
    integral = jnp.trapezoid(integrand, t, axis=1)
    return 0.5 - integral / jnp.pi


def cf_mse_vs_truth(phi_hat: jax.Array, phi_t: jax.Array) -> jax.Array:
    d = phi_hat - phi_t
    return jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)


def ks_on_grid(f_hat: jax.Array, f_true: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(f_hat - f_true))


def w1_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    """W1 between two distributions on R via CDF L1 (trapezoidal)."""
    return jnp.trapezoid(jnp.abs(f_hat - f_true), x_grid)


def empirical_cdf(sorted_samples: jax.Array, x_grid: jax.Array) -> jax.Array:
    n = sorted_samples.shape[0]
    idx = jnp.searchsorted(sorted_samples, x_grid, side="right")
    return idx.astype(jnp.float64) / jnp.maximum(n, 1)


def empirical_cf_from_samples(samples: jax.Array, t_grid: jax.Array) -> jax.Array:
    return jnp.mean(jnp.exp(1j * t_grid[:, None] * samples[None, :]), axis=1)


def cvar_from_cdf(F: jax.Array, x_grid: jax.Array, alpha: float = 0.05) -> jax.Array:
    """CVaR_alpha = (1/alpha) * integral_0^alpha F^{-1}(u) du, via inverse-CDF interp."""
    F_mono = jnp.clip(jax.lax.cummax(F.astype(jnp.float64)), 0.0, 1.0)
    u = jnp.linspace(alpha / 200.0, alpha, 200, dtype=jnp.float64)
    Q = jnp.interp(u, F_mono, x_grid.astype(jnp.float64))
    return jnp.mean(Q)


def entropic_risk_from_cdf(F: jax.Array, x_grid: jax.Array, beta: float) -> jax.Array:
    """Entropic risk (1/beta) log E[exp(beta R)] from a CDF on x_grid (Riemann-Stieltjes)."""
    F64 = F.astype(jnp.float64)
    x64 = x_grid.astype(jnp.float64)
    dF = jnp.clip(jnp.diff(F64), 0.0, None)
    x_mid = 0.5 * (x64[:-1] + x64[1:])
    expected = jnp.sum(jnp.exp(beta * x_mid) * dF)
    expected = jnp.maximum(expected, 1e-300)
    return jnp.log(expected) / beta


def entropic_risk_mog_cqn(pi: jax.Array, mu: jax.Array, sigma: jax.Array, beta: float) -> jax.Array:
    """
    Closed-form entropic risk for a Gaussian mixture.

    ER_beta(X) = (1 / beta) * log E[exp(beta X)],
    and for each component N(mu_k, sigma_k^2):
    E[exp(beta X_k)] = exp(beta * mu_k + 0.5 * beta^2 * sigma_k^2).
    """
    dtype = jnp.result_type(pi, mu, sigma, jnp.float32)
    beta_arr = jnp.asarray(beta, dtype=dtype)
    component_mgf = jnp.exp(beta_arr * mu + 0.5 * (beta_arr ** 2) * jnp.square(sigma))
    expected_exp = jnp.sum(pi * component_mgf, axis=-1)
    expected_exp = jnp.maximum(expected_exp, jnp.finfo(expected_exp.dtype).tiny)
    mean = jnp.sum(pi * mu, axis=-1)
    return jnp.where(jnp.abs(beta_arr) < 1e-8, mean, jnp.log(expected_exp) / beta_arr)

def exact_mog_cdf(x_grid: jax.Array, pi: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    """Analytical CDF for a Mixture of Gaussians."""
    # x_grid: (X,), pi/mu/sigma: (K,)
    x = x_grid[:, None]
    # z-score for each component
    z = (x - mu[None, :]) / (sigma[None, :] * jnp.sqrt(2.0))
    # Gaussian CDF for each component
    cdf_components = 0.5 * (1.0 + jax.scipy.special.erf(z))
    # Weighted sum
    return jnp.sum(pi[None, :] * cdf_components, axis=-1)


#! THIS IS HOW THE CONTEXT (x ~ 5D Gaussian) AFFECTS THE ACTION DISTRIBUTION

# ---------------------------------------------------------------------------
# ACTION DISTRIBUTIONS — tweak these constants to reshape the benchmark.
# Each distribution exposes four pure JAX callables:
#   sample(rng, x_state[B, d]) -> (B,)           — one sample per row (dataset / hist)
#   phi   (t[T], x_state[d])    -> (T,) complex   — characteristic function (analytic GT)
#   cdf   (x_grid[X], x_state[d]) -> (X,)         — CDF on x_grid (analytic GT)
#   pdf   (x_grid[X], x_state[d]) -> (X,)         — PDF on x_grid (for plotting only)
# ---------------------------------------------------------------------------
NUM_ACTIONS = 4
STATE_DIM = 5

# ---- a=0: Gaussian baseline N(x[0], sigma) ----
GAUSS_SIGMA = 1.0

def _gauss_sample(rng, x_state):
    mu = x_state[..., 0]
    eps = jax.random.normal(rng, mu.shape, dtype=x_state.dtype)
    return mu + GAUSS_SIGMA * eps

def _gauss_phi(t, x_state):
    mu = x_state[0]
    return jnp.exp(1j * t * mu - 0.5 * (GAUSS_SIGMA ** 2) * t ** 2)

def _gauss_cdf(x_grid, x_state):
    mu = x_state[0]
    return 0.5 * (1.0 + jax.scipy.special.erf((x_grid - mu) / (GAUSS_SIGMA * jnp.sqrt(2.0))))

def _gauss_pdf(x_grid, x_state):
    mu = x_state[0]
    return jnp.exp(-0.5 * ((x_grid - mu) / GAUSS_SIGMA) ** 2) / (GAUSS_SIGMA * jnp.sqrt(2.0 * jnp.pi))


# ---- a=1: Cauchy mixture (heavy-tailed); peak distance grows with |x[1]| ----
CAUCHY_PEAK_BASE = 2.0
CAUCHY_PEAK_SCALE = 2.0
CAUCHY_GAMMA = 0.5

def _cauchy_d(x_state):
    return CAUCHY_PEAK_BASE + CAUCHY_PEAK_SCALE * jnp.abs(x_state[..., 1])

def _cauchy_sample(rng, x_state):
    rk1, rk2 = jax.random.split(rng)
    pick = jax.random.bernoulli(rk1, 0.5, x_state.shape[:-1])
    u = jax.random.uniform(rk2, x_state.shape[:-1], dtype=x_state.dtype)
    z = jnp.tan(jnp.pi * (u - 0.5))
    d = _cauchy_d(x_state)
    return jnp.where(pick, -d + CAUCHY_GAMMA * z, d + CAUCHY_GAMMA * z)

def _cauchy_phi(t, x_state):
    d = _cauchy_d(x_state)
    abs_t = jnp.abs(t)
    return 0.5 * (jnp.exp(-1j * t * d - CAUCHY_GAMMA * abs_t) + jnp.exp(1j * t * d - CAUCHY_GAMMA * abs_t))

def _cauchy_cdf(x_grid, x_state):
    d = _cauchy_d(x_state)
    f1 = jnp.arctan((x_grid - (-d)) / CAUCHY_GAMMA) / jnp.pi + 0.5
    f2 = jnp.arctan((x_grid - (+d)) / CAUCHY_GAMMA) / jnp.pi + 0.5
    return 0.5 * (f1 + f2)

def _cauchy_pdf(x_grid, x_state):
    d = _cauchy_d(x_state)
    p1 = 1.0 / (jnp.pi * CAUCHY_GAMMA * (1.0 + ((x_grid + d) / CAUCHY_GAMMA) ** 2))
    p2 = 1.0 / (jnp.pi * CAUCHY_GAMMA * (1.0 + ((x_grid - d) / CAUCHY_GAMMA) ** 2))
    return 0.5 * (p1 + p2)


# ---- a=2: Sharp / Bimodal MoG; mode variance shrinks with small |x[2]| ----
MOG_MU = 3.0
MOG_SIGMA_FLOOR = 0.05
MOG_SIGMA_SCALE = 0.4

def _mog_sigma(x_state):
    return MOG_SIGMA_FLOOR + MOG_SIGMA_SCALE * jnp.abs(x_state[..., 2])

def _mog_sample(rng, x_state):
    rk1, rk2 = jax.random.split(rng)
    pick = jax.random.bernoulli(rk1, 0.5, x_state.shape[:-1])
    eps = jax.random.normal(rk2, x_state.shape[:-1], dtype=x_state.dtype)
    sigma = _mog_sigma(x_state)
    mu = jnp.where(pick, -MOG_MU, MOG_MU)
    return mu + sigma * eps

def _mog_phi(t, x_state):
    sigma = _mog_sigma(x_state)
    envelope = jnp.exp(-0.5 * (sigma ** 2) * t ** 2)
    return (envelope * jnp.cos(MOG_MU * t)).astype(jnp.complex128)

def _mog_cdf(x_grid, x_state):
    sigma = _mog_sigma(x_state)
    s2 = sigma * jnp.sqrt(2.0)
    f1 = 0.5 * (1.0 + jax.scipy.special.erf((x_grid + MOG_MU) / s2))
    f2 = 0.5 * (1.0 + jax.scipy.special.erf((x_grid - MOG_MU) / s2))
    return 0.5 * (f1 + f2)

def _mog_pdf(x_grid, x_state):
    sigma = _mog_sigma(x_state)
    p1 = jnp.exp(-0.5 * ((x_grid + MOG_MU) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))
    p2 = jnp.exp(-0.5 * ((x_grid - MOG_MU) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))
    return 0.5 * (p1 + p2)


# ---- a=3: Asymmetric Log-Normal; log-mean shifted by x[3] ----
LN_MU_BASE = 0.0
LN_MU_SCALE = 0.3
LN_SIGMA = 0.6

# Gauss-Hermite nodes / weights for the Log-Normal CF (no closed form):
# E[g(Z)] = (1/sqrt(pi)) * sum_k w_k g(sqrt(2) * xi_k),  Z ~ N(0, 1)
_LN_GH_N = 64
_gh_xi_np, _gh_w_np = hermgauss(_LN_GH_N)
LN_GH_NODES = jnp.asarray(np.sqrt(2.0) * _gh_xi_np, dtype=jnp.float64)        # (K,)
LN_GH_WEIGHTS = jnp.asarray(_gh_w_np / np.sqrt(np.pi), dtype=jnp.float64)     # (K,)

def _ln_mu(x_state):
    return LN_MU_BASE + LN_MU_SCALE * x_state[..., 3]

def _ln_sample(rng, x_state):
    eps = jax.random.normal(rng, x_state.shape[:-1], dtype=x_state.dtype)
    return jnp.exp(_ln_mu(x_state) + LN_SIGMA * eps)

def _ln_phi(t, x_state):
    mu = _ln_mu(x_state)
    y = jnp.exp(mu + LN_SIGMA * LN_GH_NODES)        # (K,)
    weights = LN_GH_WEIGHTS                         # (K,)
    return jnp.sum(weights[None, :] * jnp.exp(1j * t[:, None] * y[None, :]), axis=1)

def _ln_cdf(x_grid, x_state):
    mu = _ln_mu(x_state)
    safe_x = jnp.where(x_grid > 0, x_grid, 1.0)
    z = (jnp.log(safe_x) - mu) / (LN_SIGMA * jnp.sqrt(2.0))
    return jnp.where(x_grid > 0, 0.5 * (1.0 + jax.scipy.special.erf(z)), 0.0)

def _ln_pdf(x_grid, x_state):
    mu = _ln_mu(x_state)
    safe_x = jnp.where(x_grid > 0, x_grid, 1.0)
    log_dens = -0.5 * ((jnp.log(safe_x) - mu) / LN_SIGMA) ** 2
    pdf = jnp.exp(log_dens) / (safe_x * LN_SIGMA * jnp.sqrt(2.0 * jnp.pi))
    return jnp.where(x_grid > 0, pdf, 0.0)


@dataclass(frozen=True)
class ActionDist:
    name: str
    short: str
    sample: Callable
    phi: Callable
    cdf: Callable
    pdf: Callable
    plot_x_min: float
    plot_x_max: float
    plot_t_max: float


ACTION_DISTS: tuple[ActionDist, ...] = (
    ActionDist(
        "Gaussian (baseline)", "Gauss",
        _gauss_sample, _gauss_phi, _gauss_cdf, _gauss_pdf,
        plot_x_min=-6.0, plot_x_max=6.0, plot_t_max=6.0,
    ),
    ActionDist(
        "Cauchy mixture (heavy-tailed)", "Cauchy",
        _cauchy_sample, _cauchy_phi, _cauchy_cdf, _cauchy_pdf,
        plot_x_min=-12.0, plot_x_max=12.0, plot_t_max=10.0,
    ),
    ActionDist(
        "MoG (sharp / bimodal)", "MoG",
        _mog_sample, _mog_phi, _mog_cdf, _mog_pdf,
        plot_x_min=-6.0, plot_x_max=6.0, plot_t_max=12.0,
    ),
    ActionDist(
        "Log-Normal (skewed)", "LogN",
        _ln_sample, _ln_phi, _ln_cdf, _ln_pdf,
        plot_x_min=-1.0, plot_x_max=8.0, plot_t_max=10.0,
    ),
)
assert len(ACTION_DISTS) == NUM_ACTIONS


def analytic_bellman_mean_var_mv(
    k: int,
    x_state: jax.Array,
    gamma64: jax.Array,
    reward_std64: jax.Array,
    mv_lambda64: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Exact mean / variance / mean-variance functional of the Bellman target
    ``Z = R + gamma * Z_next`` with ``R ~ N(0, immediate_reward_std^2)`` independent of ``Z_next``.

    Action 1 (Cauchy mixture): classical mean and variance of ``Z_next`` do not exist (heavy tails);
    return NaNs so tables mark those cells as undefined (not finite-grid artifacts).

    Gaussian: ``Z_next ~ N(x[0], GAUSS_SIGMA^2)``.
    MoG: 0.5 ``N(-MOG_MU, sigma(x)^2)`` + 0.5 ``N(+MOG_MU, sigma(x)^2)`` with ``sigma = _mog_sigma(x)``.
    Log-Normal: ``log Z_next ~ N(_ln_mu(x), LN_SIGMA^2)``.
    """
    xs = x_state.astype(jnp.float64)
    g = gamma64
    sr = reward_std64
    lam = mv_lambda64

    if k == 0:
        mean_next = xs[0]
        var_next = jnp.float64(GAUSS_SIGMA ** 2)
        mean_z = g * mean_next
        var_z = sr ** 2 + g ** 2 * var_next
    elif k == 1:
        nan = jnp.float64(jnp.nan)
        return nan, nan, nan
    elif k == 2:
        sigma_m = _mog_sigma(xs)
        var_next = sigma_m ** 2 + jnp.float64(MOG_MU ** 2)
        mean_z = g * jnp.float64(0.0)
        var_z = sr ** 2 + g ** 2 * var_next
    else:
        mu_ln = _ln_mu(xs)
        sig = jnp.float64(LN_SIGMA)
        mean_next = jnp.exp(mu_ln + 0.5 * sig ** 2)
        var_next = (jnp.exp(sig ** 2) - 1.0) * jnp.exp(2.0 * mu_ln + sig ** 2)
        mean_z = g * mean_next
        var_z = sr ** 2 + g ** 2 * var_next

    mv_z = mean_z - lam * var_z
    return mean_z, var_z, mv_z


# ---------------------------------------------------------------------------
# Networks
#
# Both nets share the same (x, a)-trunk shape: concat(state, one_hot(action)) -> MLP.
# IQN head: cosine tau embedding ⊙ trunk -> scalar quantile (standard IQN paper merge).
# CQN head: cosine omega embedding ⊙ trunk -> (mean, scale) -> parametric CF, exactly as
#   in purejaxql/cqn_minatar.py (CharacteristicQNetwork). Specifically:
#     cf(ω) = exp(-½ · κ(ω) · scale + i · ω · mean), κ(ω) = ω² / (1 + ω²)
#   This guarantees |cf| ≤ 1 smoothly and biases the network toward Gaussian-like CFs
#   while still being expressive (mean and scale are conditioned on (x, a, ω)).
# ---------------------------------------------------------------------------
class FrequencyEmbedding(nn.Module):
    """cos(π · ω · {1..embed_dim}) — same Fourier basis used by cqn_minatar."""

    embed_dim: int

    @nn.compact
    def __call__(self, omega: jax.Array) -> jax.Array:
        omega = jnp.asarray(omega, dtype=jnp.float32)
        basis = jnp.arange(1, self.embed_dim + 1, dtype=omega.dtype)
        return jnp.cos(jnp.pi * omega[..., None] * basis)


class IQNNet(nn.Module):
    """trunk((x, one_hot(a))) ⊙ ReLU(Linear(cos(pi i tau))) -> small MLP -> scalar quantile."""

    n_actions: int
    embed_dim: int
    hidden_dim: int
    trunk_depth: int

    @nn.compact
    def __call__(self, x: jax.Array, a_onehot: jax.Array, tau: jax.Array) -> jax.Array:
        # x: (B, d); a_onehot: (B, A); tau: (B, M) in (0, 1)
        h = jnp.concatenate([x, a_onehot], axis=-1)
        for i in range(self.trunk_depth):
            h = nn.relu(nn.Dense(self.hidden_dim, name=f"trunk_{i}")(h))
        # h: (B, hidden)
        tau_clipped = jnp.clip(tau, 1e-6, 1.0 - 1e-6)
        idx = jnp.arange(1, self.embed_dim + 1, dtype=tau.dtype)
        emb = jnp.cos(jnp.pi * tau_clipped[..., None] * idx[None, None, :])  # (B, M, E)
        emb = nn.relu(nn.Dense(self.hidden_dim, name="tau_proj")(emb))       # (B, M, H)
        merged = h[:, None, :] * emb                                         # (B, M, H)
        merged = nn.relu(nn.Dense(self.hidden_dim, name="post_merge")(merged))
        q = nn.Dense(1, name="head")(merged).squeeze(-1)                     # (B, M)
        return q


class CQNNet(nn.Module):
    """
    Parametric CF network mirroring purejaxql/cqn_minatar.py::CharacteristicQNetwork.

    Forward returns ``(mean, scale)`` with shape ``(B, K)`` for ``K`` (typically positive)
    frequencies. The CF is reconstructed at the call site via :func:`compose_cf` to allow
    enforcing exact conjugate symmetry ``cf(-ω) = conj(cf(ω))`` at evaluation time.
    """

    n_actions: int
    embed_dim: int
    hidden_dim: int
    trunk_depth: int
    sigma_min: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, a_onehot: jax.Array, omega: jax.Array) -> tuple[jax.Array, jax.Array]:
        # x: (B, d); a_onehot: (B, A); omega: (B, K) real (caller passes |ω| at eval)
        h = jnp.concatenate([x, a_onehot], axis=-1)
        for i in range(self.trunk_depth):
            h = nn.relu(nn.Dense(self.hidden_dim, name=f"trunk_{i}")(h))
        emb = FrequencyEmbedding(self.embed_dim)(omega)                       # (B, K, embed_dim)
        emb = nn.Dense(self.hidden_dim, name="freq_proj")(emb)                # (B, K, H)
        joint = h[:, None, :] * emb                                           # (B, K, H)
        joint = nn.relu(nn.Dense(self.hidden_dim, name="post_merge")(joint))
        mean = nn.Dense(1, name="mean_head", kernel_init=nn.initializers.zeros)(joint).squeeze(-1)
        raw_scale = nn.Dense(1, name="scale_head", kernel_init=nn.initializers.zeros)(joint).squeeze(-1)
        scale = nn.softplus(raw_scale) + self.sigma_min
        return mean, scale


class MoGCQNNet(nn.Module):
    """
    State-conditioned MoG head with analytic CF.

    This intentionally follows the ``mog_pqn_minatar.py`` parameterization (pi, mu, sigma)
    while dropping all PQN-specific mechanisms. The trunk is the same (x, a) MLP used by
    IQN/CQN so the comparison remains apples-to-apples.
    """

    n_actions: int
    hidden_dim: int
    trunk_depth: int
    num_components: int
    sigma_min: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array, a_onehot: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = jnp.concatenate([x, a_onehot], axis=-1)
        for i in range(self.trunk_depth):
            h = nn.relu(nn.Dense(self.hidden_dim, name=f"trunk_{i}")(h))
        pi_logits = nn.Dense(self.num_components, name="pi_head")(h)
        pi = jax.nn.softmax(pi_logits, axis=-1)
        mu = nn.Dense(self.num_components, name="mu_head")(h)
        raw_sigma = nn.Dense(self.num_components, name="sigma_head")(h)
        sigma = nn.softplus(raw_sigma) + self.sigma_min
        return pi, mu, sigma


def compose_cf(
    omega_signed: jax.Array,
    mean_abs: jax.Array,
    scale_abs: jax.Array,
    dtype: jnp.dtype = jnp.complex128,
) -> jax.Array:
    """
    Build cf(ω) = exp(-½·κ(|ω|)·scale + i·ω·mean) from network outputs evaluated at |ω|.
    Phase carries the sign of ω so that cf(-ω) = conj(cf(ω)) by construction. κ is even
    in ω so the magnitude is automatically even.

    ``dtype`` controls the complex precision (default complex128 for eval). Training should
    pass ``jnp.complex64`` to keep the loss in fp32 and avoid promoting the entire backward
    pass through the trunk to fp64.
    """
    omega_abs = jnp.abs(omega_signed)
    kappa = jnp.square(omega_abs) / (1.0 + jnp.square(omega_abs))
    log_mag = -0.5 * kappa * scale_abs
    phase = omega_signed * mean_abs
    return jnp.exp(log_mag.astype(dtype) + 1j * phase.astype(dtype))


def compose_mog_cf(
    omega_signed: jax.Array,
    pi: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
    dtype: jnp.dtype = jnp.complex128,
) -> jax.Array:
    """Analytic CF for a Gaussian mixture: sum_k pi_k exp(i w mu_k - 0.5 w^2 sigma_k^2)."""
    omega = omega_signed[..., None]

    # Align mixture params to broadcast with omega:
    # - training: omega (B, K, 1), params (B, C) -> (B, 1, C)
    # - eval:     omega (T, 1),    params (C,)   -> (1, C)
    def _align_param(param: jax.Array) -> jax.Array:
        if param.ndim == omega_signed.ndim:
            return param[..., None, :]
        if param.ndim == omega_signed.ndim + 1:
            return param
        raise ValueError(
            "MoG parameter rank incompatible with omega rank: "
            f"param.ndim={param.ndim}, omega.ndim={omega_signed.ndim}"
        )

    pi = _align_param(pi)
    mu = _align_param(mu)
    sigma = _align_param(sigma)

    term = jnp.exp(
        (1j * omega * mu).astype(dtype)
        - 0.5 * jnp.square(omega).astype(dtype) * jnp.square(sigma).astype(dtype)
    )
    return jnp.sum(pi.astype(dtype) * term, axis=-1)


def project_delta_onto_c51_atoms(y: jax.Array, support: jax.Array) -> jax.Array:
    """Batch C51 projection of scalar Bellman targets onto a fixed atom grid (Bellemare et al., 2017)."""
    vmin = support[0]
    vmax = support[-1]
    n = support.shape[0]
    dz = (vmax - vmin) / jnp.float32(n - 1)
    y_c = jnp.clip(y, vmin, vmax)
    b = (y_c - vmin) / dz
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)
    lower = jnp.clip(lower, 0, n - 1)
    upper = jnp.clip(upper, 0, n - 1)
    lower_mass = upper.astype(jnp.float32) - b
    upper_mass = b - lower.astype(jnp.float32)
    same = lower == upper
    lower_mass = jnp.where(same, jnp.ones_like(lower_mass), lower_mass)
    upper_mass = jnp.where(same, jnp.zeros_like(upper_mass), upper_mass)
    batch = jnp.arange(y.shape[0], dtype=jnp.int32)
    mat = jnp.zeros((y.shape[0], n), dtype=jnp.float32)
    mat = mat.at[batch, lower].add(lower_mass)
    mat = mat.at[batch, upper].add(upper_mass)
    return mat


class C51Net(nn.Module):
    """C51 categorical head on the shared trunk design (same depth/width as IQN/CQN trunk)."""

    n_actions: int
    hidden_dim: int
    trunk_depth: int
    num_atoms: int

    @nn.compact
    def __call__(self, x: jax.Array, a_onehot: jax.Array) -> jax.Array:
        h = jnp.concatenate([x, a_onehot], axis=-1)
        for i in range(self.trunk_depth):
            h = nn.relu(nn.Dense(self.hidden_dim, name=f"trunk_{i}")(h))
        return nn.Dense(self.num_atoms, name="atom_logits")(h)


def discrete_return_cdf(probs: jax.Array, atoms: jax.Array, x_grid: jax.Array) -> jax.Array:
    """CDF of ``sum_k p_k δ_{z_k}`` on ``x_grid``."""
    return jnp.sum(
        probs[:, None] * (atoms[:, None] <= x_grid[None, :]).astype(jnp.float64),
        axis=0,
    )


def discrete_return_cf(probs: jax.Array, atoms: jax.Array, t: jax.Array) -> jax.Array:
    """``φ(t) = Σ_k p_k exp(i t z_k)``."""
    return jnp.sum(
        probs[:, None] * jnp.exp(1j * atoms[:, None] * t[None, :].astype(jnp.complex128)),
        axis=0,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIG_FILE_TAG = "_omega_15"

PANEL_STATE_SEED_OFFSET = 9_001
PANEL_HIST_SAMPLES = 50_000
DEFAULT_OMEGA_MAX = 30.0

@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    state_dim: int = STATE_DIM

    # Dataset
    dataset_size: int = 500_000
    dataset_seed_offset: int = 1_001

    # Algorithms (disable CQN to compare IQN vs MoG-CQN only; skips training, CSV, tables, NPZ panels)
    train_cqn: bool = True
    #: Train only MoG-CQN (no IQN / QR-DQN / C51 / CQN); eval/msgpack/W&B only expose MoG metrics.
    train_mog_cqn_only: bool = False

    # Training
    total_steps: int = 100_000
    log_every: int = 20_000        # also the size of one JIT'd lax.scan chunk
    batch_size: int = 1024 #! 512
    num_tau: int = 32
    num_omega: int = 32
    gamma: float = 0.99
    immediate_reward_std: float = 0.1
    omega_max: float = DEFAULT_OMEGA_MAX  # IQN-side reference; eval grid half-width
    huber_kappa: float = 1.0
    lr_iqn: float = 1e-3
    lr_cf: float = 1e-3
    max_grad_norm: float = 10.0   # global-norm clip (matches cqn_minatar MAX_GRAD_NORM)

    # CQN training ω: truncated exponential on (0, omega_clip] — purejaxql/utils/mog_cf.sample_frequencies
    omega_sampler: str = "half_laplacian"
    omega_clip: float = DEFAULT_OMEGA_MAX  # upper truncation (= omega_max in mog_cf.sample_frequencies)
    omega_laplace_scale: float = 1 # None → omega_clip / 3 (same default as mog_cf.py)
    sigma_min: float = 1e-6
    aux_mean_loss_weight: float = 0.1
    num_mog_components: int = 5
    #: C51 (fair vs IQN): same atom count as implicit quantiles by default (Bellemare et al., 2017).
    num_atoms: int = 32
    #: C51 atom support ``[v_min, v_max]``. Too wide wastes atoms on empty tails; too narrow clips
    #: targets and hurts metrics. ``±25`` matches the bulk of ``R + γ Z`` here while leaving margin
    #: for Cauchy/log-normal tails (same γ and reward noise as the rest of the benchmark).
    c51_v_min: float = -25.0
    c51_v_max: float = 25.0

    # Architecture
    embed_dim: int = 64
    hidden_dim: int = 256
    trunk_depth: int = 3

    # Eval
    n_test: int = 10_000
    test_seed_offset: int = 7_777
    eval_chunk_size: int = 50
    num_taus_iqn_eval: int = 2_000
    eval_iqn_seed: int = 42
    eval_t_max: float = DEFAULT_OMEGA_MAX  # CF grid half-width; keep <= omega_max (see DEFAULT_OMEGA_MAX)
    eval_num_t: int = 201
    # Primary visualization support (kept compact for readable 4x3 panels).
    eval_x_min: float = -15.0
    eval_x_max: float = 15.0
    eval_num_x: int = 301
    # Dual-grid "gold standard" risk support (tail-safe integration; prevents truncation masking).
    risk_x_min: float = -200.0
    risk_x_max: float = 200.0
    risk_num_x: int = 2001
    gp_t_delta: float = 1e-4
    gp_t_max: float = DEFAULT_OMEGA_MAX
    gp_num_t: int = 501
    cvar_alpha: float = 0.05
    er_beta: float = 0.1
    mv_lambda: float = 0.1

    # Output
    figures_dir: str = "figures/whitebox_bandit"
    figure_tag: str = FIG_FILE_TAG
    #: When True (default): each training run only writes ``.npz`` (+ msgpack params). Figures/CSVs come from aggregation.
    minimal_artifact_export: bool = True

    def npz_filename(self) -> str:
        return f"purejaxql/whitebox_bandit_run{self.figure_tag}.npz"

    def msgpack_filename(self) -> str:
        return f"purejaxql/whitebox_bandit_run{self.figure_tag}.msgpack"


def c51_atoms(cfg: TrainConfig) -> jax.Array:
    return jnp.linspace(
        jnp.float32(cfg.c51_v_min),
        jnp.float32(cfg.c51_v_max),
        cfg.num_atoms,
        dtype=jnp.float32,
    )


def fixed_qr_tau_grid(num_tau: int, dtype=jnp.float32) -> jax.Array:
    """Dabney et al. (2018) fixed quantile midpoints in (0, 1)."""
    idx = jnp.arange(1, num_tau + 1, dtype=dtype)
    return (idx - jnp.float32(0.5)) / jnp.float32(num_tau)


def panel_state_seed(cfg: TrainConfig) -> int:
    return int(cfg.seed + PANEL_STATE_SEED_OFFSET)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset(rng: jax.Array, cfg: TrainConfig) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """x ~ N(0, I_d), a ~ Uniform{0..A-1}, with white-box Bellman pieces (r_immediate, z_next)."""
    rng_x, rng_a, rng_r, rng_z = jax.random.split(rng, 4)
    n, d = cfg.dataset_size, cfg.state_dim
    x = jax.random.normal(rng_x, (n, d), dtype=jnp.float32)
    a = jax.random.randint(rng_a, (n,), 0, NUM_ACTIONS)

    # Immediate reward used for Bellman target: simple dense Gaussian noise.
    r_immediate = jax.random.normal(rng_r, (n,), dtype=jnp.float32) * jnp.float32(cfg.immediate_reward_std)

    # "Complex" distributions are injected as exact next-state samples z_next.
    sub_keys = jax.random.split(rng_z, NUM_ACTIONS)
    z_next_per_action = jnp.stack(
        [dist.sample(sub_keys[k], x).astype(jnp.float32) for k, dist in enumerate(ACTION_DISTS)],
        axis=0,
    )  # (A, N)
    z_next = z_next_per_action[a, jnp.arange(n)]
    return x, a, r_immediate, z_next


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _make_optimizer(lr: float, max_grad_norm: float) -> optax.GradientTransformation:
    """Adam + optional global-norm clip (Adam preferred for this supervised benchmark)."""
    return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr))


def create_train_states_mog_only(rng: jax.Array, cfg: TrainConfig) -> tuple[train_state.TrainState, MoGCQNNet]:
    """MoG-CQN head only (same trunk design as full benchmark)."""
    mog = MoGCQNNet(
        NUM_ACTIONS,
        cfg.hidden_dim,
        cfg.trunk_depth,
        num_components=cfg.num_mog_components,
        sigma_min=cfg.sigma_min,
    )
    x0 = jnp.zeros((2, cfg.state_dim), jnp.float32)
    a0 = jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    v_mog = mog.init(rng, x0, a0)
    state_m = train_state.TrainState.create(
        apply_fn=mog.apply,
        params=v_mog["params"],
        tx=_make_optimizer(cfg.lr_cf, cfg.max_grad_norm),
    )
    return state_m, mog


def create_train_states(rng: jax.Array, cfg: TrainConfig):
    """IQN + QR-DQN (same architecture, separate weights) + C51 + MoG-CQN; optional CQN."""
    rng_i, rng_q, rng_51, rng_c, rng_m = jax.random.split(rng, 5)
    iqn = IQNNet(NUM_ACTIONS, cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth)
    qr_dqn = IQNNet(NUM_ACTIONS, cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth)
    c51 = C51Net(NUM_ACTIONS, cfg.hidden_dim, cfg.trunk_depth, cfg.num_atoms)
    mog = MoGCQNNet(
        NUM_ACTIONS,
        cfg.hidden_dim,
        cfg.trunk_depth,
        num_components=cfg.num_mog_components,
        sigma_min=cfg.sigma_min,
    )

    x0 = jnp.zeros((2, cfg.state_dim), jnp.float32)
    a0 = jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    tau0 = jnp.full((2, cfg.num_tau), 0.5, jnp.float32)

    v_iqn = iqn.init(rng_i, x0, a0, tau0)
    v_qr = qr_dqn.init(rng_q, x0, a0, tau0)
    v_c51 = c51.init(rng_51, x0, a0)
    v_mog = mog.init(rng_m, x0, a0)

    tx_iqn = _make_optimizer(cfg.lr_iqn, cfg.max_grad_norm)
    state_i = train_state.TrainState.create(apply_fn=iqn.apply, params=v_iqn["params"], tx=tx_iqn)
    state_q = train_state.TrainState.create(apply_fn=qr_dqn.apply, params=v_qr["params"], tx=tx_iqn)
    state_c51 = train_state.TrainState.create(apply_fn=c51.apply, params=v_c51["params"], tx=tx_iqn)
    state_m = train_state.TrainState.create(
        apply_fn=mog.apply, params=v_mog["params"], tx=_make_optimizer(cfg.lr_cf, cfg.max_grad_norm),
    )

    if not cfg.train_cqn:
        return state_i, state_q, state_c51, None, state_m, iqn, qr_dqn, c51, None, mog

    cqn = CQNNet(NUM_ACTIONS, cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth, sigma_min=cfg.sigma_min)
    om0 = jnp.zeros((2, cfg.num_omega + 1), jnp.float32)
    v_cqn = cqn.init(rng_c, x0, a0, om0)
    state_c = train_state.TrainState.create(
        apply_fn=cqn.apply, params=v_cqn["params"], tx=_make_optimizer(cfg.lr_cf, cfg.max_grad_norm),
    )
    return state_i, state_q, state_c51, state_c, state_m, iqn, qr_dqn, c51, cqn, mog


def make_train_chunk(
    cfg: TrainConfig,
    iqn: IQNNet,
    qr_dqn: IQNNet,
    c51: C51Net,
    cqn: CQNNet,
    mog: MoGCQNNet,
    dataset: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
):
    """JIT-compiled scan over ``cfg.log_every`` steps. Returns (states, rng) -> (states, rng, mean_losses)."""
    x_all, a_all, r_all, z_next_all = dataset
    n = x_all.shape[0]
    aux_w = jnp.float32(cfg.aux_mean_loss_weight)
    gamma = jnp.float32(cfg.gamma)
    z_atoms = c51_atoms(cfg)

    def step(carry, _):
        state_i, state_q, state_c51, state_c, state_m, rng = carry
        rng, kb, kt, ko = jax.random.split(rng, 4)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        x_b = x_all[idx]
        a_b = a_all[idx]
        r_b = r_all[idx]
        z_next_b = z_next_all[idx]
        a_oh = jax.nn.one_hot(a_b, NUM_ACTIONS, dtype=jnp.float32)

        taus = jax.random.uniform(kt, (cfg.batch_size, cfg.num_tau), dtype=jnp.float32)
        tau_fixed = fixed_qr_tau_grid(cfg.num_tau)
        taus_qr = jnp.broadcast_to(tau_fixed[None, :], (cfg.batch_size, cfg.num_tau))
        # CQN: positive ω only — half-Laplacian / truncated exponential (purejaxql/utils/mog_cf.py).
        n_flat = cfg.batch_size * cfg.num_omega
        scale = cfg.omega_laplace_scale
        om_flat = sample_frequencies_half_laplacian(
            ko,
            num_samples=n_flat,
            omega_max=jnp.asarray(cfg.omega_clip, jnp.float32),
            scale=jnp.asarray(scale, jnp.float32) if scale is not None else None,
        ).astype(jnp.float32)
        omegas_pos = om_flat.reshape(cfg.batch_size, cfg.num_omega)
        omegas_with_zero = jnp.concatenate(
            [jnp.zeros((cfg.batch_size, 1), dtype=omegas_pos.dtype), omegas_pos], axis=1,
        )  # (B, K+1)

        # One-sample spatial Bellman targets for the batch.
        target_samples = r_b + gamma * z_next_b

        def loss_iqn(params):
            q = iqn.apply({"params": params}, x_b, a_oh, taus)  # (B, M)
            return quantile_huber_loss_pairwise(q, target_samples, taus, cfg.huber_kappa)

        def loss_qr_dqn(params):
            q = qr_dqn.apply({"params": params}, x_b, a_oh, taus_qr)
            return quantile_huber_loss_pairwise(q, target_samples, taus_qr, cfg.huber_kappa)

        def loss_c51(params):
            logits = c51.apply({"params": params}, x_b, a_oh)
            target_probs = project_delta_onto_c51_atoms(target_samples, z_atoms)
            log_p = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(target_probs * log_p, axis=-1))

        def loss_cqn(params):
            mean_pred, scale_pred = cqn.apply({"params": params}, x_b, a_oh, omegas_with_zero)
            # CF loss on the K positive ω; aux mean loss on the leading ω=0 column.
            # Use complex64 here: complex128 would force the entire backward pass through
            # the trunk to fp64 (fp32 inputs get promoted by the loss dtype) and roughly
            # halve training throughput on GPU.
            cf_pred = compose_cf(
                omegas_with_zero[:, 1:], mean_pred[:, 1:], scale_pred[:, 1:],
                dtype=jnp.complex64,
            )
            omega_pos = omegas_with_zero[:, 1:]
            gamma_omega = gamma * omega_pos

            # Analytical Bellman convolution in frequency domain:
            # phi(r + gamma * Z_next) = exp(i * omega * r) * phi_Z_next(gamma * omega).
            cf_r = jnp.exp(1j * omega_pos.astype(jnp.complex64) * r_b[:, None].astype(jnp.complex64))

            def _phi_one(a_i: jax.Array, w_i: jax.Array, x_i: jax.Array) -> jax.Array:
                return jax.lax.switch(a_i, [d.phi for d in ACTION_DISTS], w_i.astype(jnp.float64), x_i.astype(jnp.float64))

            cf_z_next = jax.vmap(_phi_one)(a_b, gamma_omega, x_b).astype(jnp.complex64)
            cf_target = cf_r * cf_z_next
            d = cf_pred - cf_target
            cf_loss = 0.5 * jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)
            # Aux mean loss is Huber (kappa=1) instead of MSE: Cauchy-mixture rewards have
            # infinite variance, so single-sample MSE produces unbounded gradients. Huber
            # gives the same E[r] minimum (when it exists) but is robust to outliers.
            mean_at_zero = mean_pred[:, 0]  # mean(x, a, ω=0)
            mean_loss = jnp.mean(_huber(mean_at_zero - target_samples, cfg.huber_kappa))
            return cf_loss + aux_w * mean_loss

        def loss_mog(params):
            pi, mu, sigma = mog.apply({"params": params}, x_b, a_oh)
            cf_pred = compose_mog_cf(omegas_pos, pi, mu, sigma, dtype=jnp.complex64)
            gamma_omega = gamma * omegas_pos

            cf_r = jnp.exp(1j * omegas_pos.astype(jnp.complex64) * r_b[:, None].astype(jnp.complex64))

            def _phi_one(a_i: jax.Array, w_i: jax.Array, x_i: jax.Array) -> jax.Array:
                return jax.lax.switch(a_i, [d.phi for d in ACTION_DISTS], w_i.astype(jnp.float64), x_i.astype(jnp.float64))

            cf_z_next = jax.vmap(_phi_one)(a_b, gamma_omega, x_b).astype(jnp.complex64)
            cf_target = cf_r * cf_z_next
            d = cf_pred - cf_target
            cf_loss = 0.5 * jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)
            mean_pred = jnp.sum(pi * mu, axis=-1)
            mean_loss = jnp.mean(_huber(mean_pred - target_samples, cfg.huber_kappa))
            return cf_loss + aux_w * mean_loss

        li, gi = jax.value_and_grad(loss_iqn)(state_i.params)
        lq, gq = jax.value_and_grad(loss_qr_dqn)(state_q.params)
        l51, g51 = jax.value_and_grad(loss_c51)(state_c51.params)
        lc, gc = jax.value_and_grad(loss_cqn)(state_c.params)
        lm, gm = jax.value_and_grad(loss_mog)(state_m.params)
        state_i = state_i.apply_gradients(grads=gi)
        state_q = state_q.apply_gradients(grads=gq)
        state_c51 = state_c51.apply_gradients(grads=g51)
        state_c = state_c.apply_gradients(grads=gc)
        state_m = state_m.apply_gradients(grads=gm)
        return (state_i, state_q, state_c51, state_c, state_m, rng), (li, lq, l51, lc, lm)

    @jax.jit
    def run_chunk(state_i, state_q, state_c51, state_c, state_m, rng):
        (state_i, state_q, state_c51, state_c, state_m, rng), losses = jax.lax.scan(
            step,
            (state_i, state_q, state_c51, state_c, state_m, rng),
            None,
            length=cfg.log_every,
        )
        return state_i, state_q, state_c51, state_c, state_m, rng, losses

    return run_chunk


def make_train_chunk_no_cqn(
    cfg: TrainConfig,
    iqn: IQNNet,
    qr_dqn: IQNNet,
    c51: C51Net,
    mog: MoGCQNNet,
    dataset: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
):
    """Same as ``make_train_chunk`` without CQN (IQN + QR-DQN + C51 + MoG-CQN)."""
    x_all, a_all, r_all, z_next_all = dataset
    n = x_all.shape[0]
    aux_w = jnp.float32(cfg.aux_mean_loss_weight)
    gamma = jnp.float32(cfg.gamma)
    z_atoms = c51_atoms(cfg)

    def step(carry, _):
        state_i, state_q, state_c51, state_m, rng = carry
        rng, kb, kt, ko = jax.random.split(rng, 4)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        x_b = x_all[idx]
        a_b = a_all[idx]
        r_b = r_all[idx]
        z_next_b = z_next_all[idx]
        a_oh = jax.nn.one_hot(a_b, NUM_ACTIONS, dtype=jnp.float32)

        taus = jax.random.uniform(kt, (cfg.batch_size, cfg.num_tau), dtype=jnp.float32)
        tau_fixed = fixed_qr_tau_grid(cfg.num_tau)
        taus_qr = jnp.broadcast_to(tau_fixed[None, :], (cfg.batch_size, cfg.num_tau))
        n_flat = cfg.batch_size * cfg.num_omega
        scale = cfg.omega_laplace_scale
        om_flat = sample_frequencies_half_laplacian(
            ko,
            num_samples=n_flat,
            omega_max=jnp.asarray(cfg.omega_clip, jnp.float32),
            scale=jnp.asarray(scale, jnp.float32) if scale is not None else None,
        ).astype(jnp.float32)
        omegas_pos = om_flat.reshape(cfg.batch_size, cfg.num_omega)

        target_samples = r_b + gamma * z_next_b

        def loss_iqn(params):
            q = iqn.apply({"params": params}, x_b, a_oh, taus)
            return quantile_huber_loss_pairwise(q, target_samples, taus, cfg.huber_kappa)

        def loss_qr_dqn(params):
            q = qr_dqn.apply({"params": params}, x_b, a_oh, taus_qr)
            return quantile_huber_loss_pairwise(q, target_samples, taus_qr, cfg.huber_kappa)

        def loss_c51(params):
            logits = c51.apply({"params": params}, x_b, a_oh)
            target_probs = project_delta_onto_c51_atoms(target_samples, z_atoms)
            log_p = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(target_probs * log_p, axis=-1))

        def loss_mog(params):
            pi, mu, sigma = mog.apply({"params": params}, x_b, a_oh)
            cf_pred = compose_mog_cf(omegas_pos, pi, mu, sigma, dtype=jnp.complex64)
            gamma_omega = gamma * omegas_pos

            cf_r = jnp.exp(1j * omegas_pos.astype(jnp.complex64) * r_b[:, None].astype(jnp.complex64))

            def _phi_one(a_i: jax.Array, w_i: jax.Array, x_i: jax.Array) -> jax.Array:
                return jax.lax.switch(a_i, [d.phi for d in ACTION_DISTS], w_i.astype(jnp.float64), x_i.astype(jnp.float64))

            cf_z_next = jax.vmap(_phi_one)(a_b, gamma_omega, x_b).astype(jnp.complex64)
            cf_target = cf_r * cf_z_next
            d = cf_pred - cf_target
            cf_loss = 0.5 * jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)
            mean_pred = jnp.sum(pi * mu, axis=-1)
            mean_loss = jnp.mean(_huber(mean_pred - target_samples, cfg.huber_kappa))
            return cf_loss + aux_w * mean_loss

        li, gi = jax.value_and_grad(loss_iqn)(state_i.params)
        lq, gq = jax.value_and_grad(loss_qr_dqn)(state_q.params)
        l51, g51 = jax.value_and_grad(loss_c51)(state_c51.params)
        lm, gm = jax.value_and_grad(loss_mog)(state_m.params)
        state_i = state_i.apply_gradients(grads=gi)
        state_q = state_q.apply_gradients(grads=gq)
        state_c51 = state_c51.apply_gradients(grads=g51)
        state_m = state_m.apply_gradients(grads=gm)
        return (state_i, state_q, state_c51, state_m, rng), (li, lq, l51, lm)

    @jax.jit
    def run_chunk(state_i, state_q, state_c51, state_m, rng):
        (state_i, state_q, state_c51, state_m, rng), losses = jax.lax.scan(
            step,
            (state_i, state_q, state_c51, state_m, rng),
            None,
            length=cfg.log_every,
        )
        return state_i, state_q, state_c51, state_m, rng, losses

    return run_chunk


def make_train_chunk_mog_only(
    cfg: TrainConfig,
    mog: MoGCQNNet,
    dataset: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
):
    """JIT scan: MoG-CQN CF loss only."""
    x_all, a_all, r_all, z_next_all = dataset
    n = x_all.shape[0]
    aux_w = jnp.float32(cfg.aux_mean_loss_weight)
    gamma = jnp.float32(cfg.gamma)

    def step(carry, _):
        state_m, rng = carry
        rng, kb, ko = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        x_b = x_all[idx]
        a_b = a_all[idx]
        r_b = r_all[idx]
        z_next_b = z_next_all[idx]
        a_oh = jax.nn.one_hot(a_b, NUM_ACTIONS, dtype=jnp.float32)
        n_flat = cfg.batch_size * cfg.num_omega
        scale = cfg.omega_laplace_scale
        om_flat = sample_frequencies_half_laplacian(
            ko,
            num_samples=n_flat,
            omega_max=jnp.asarray(cfg.omega_clip, jnp.float32),
            scale=jnp.asarray(scale, jnp.float32) if scale is not None else None,
        ).astype(jnp.float32)
        omegas_pos = om_flat.reshape(cfg.batch_size, cfg.num_omega)
        target_samples = r_b + gamma * z_next_b

        def loss_mog(params):
            pi, mu, sigma = mog.apply({"params": params}, x_b, a_oh)
            cf_pred = compose_mog_cf(omegas_pos, pi, mu, sigma, dtype=jnp.complex64)
            gamma_omega = gamma * omegas_pos
            cf_r = jnp.exp(1j * omegas_pos.astype(jnp.complex64) * r_b[:, None].astype(jnp.complex64))

            def _phi_one(a_i: jax.Array, w_i: jax.Array, x_i: jax.Array) -> jax.Array:
                return jax.lax.switch(a_i, [d.phi for d in ACTION_DISTS], w_i.astype(jnp.float64), x_i.astype(jnp.float64))

            cf_z_next = jax.vmap(_phi_one)(a_b, gamma_omega, x_b).astype(jnp.complex64)
            cf_target = cf_r * cf_z_next
            d = cf_pred - cf_target
            cf_loss = 0.5 * jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)
            mean_pred = jnp.sum(pi * mu, axis=-1)
            mean_loss = jnp.mean(_huber(mean_pred - target_samples, cfg.huber_kappa))
            return cf_loss + aux_w * mean_loss

        lm, gm = jax.value_and_grad(loss_mog)(state_m.params)
        state_m = state_m.apply_gradients(grads=gm)
        return (state_m, rng), lm

    @jax.jit
    def run_chunk(state_m, rng):
        (state_m, rng), losses = jax.lax.scan(step, (state_m, rng), None, length=cfg.log_every)
        return state_m, rng, losses

    return run_chunk


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _make_eval_one_state_with_cqn(
    cfg: TrainConfig,
    iqn: IQNNet,
    qr_dqn: IQNNet,
    c51: C51Net,
    cqn: CQNNet,
    mog: MoGCQNNet,
):
    """
    ``eval_one_state(params_iqn, params_qr_dqn, params_c51, params_cqn, params_mog, x_state)``
    → per-metric arrays stacked over actions (leading dim ``NUM_ACTIONS``).
    """
    t_eval = jnp.linspace(-cfg.eval_t_max, cfg.eval_t_max, cfg.eval_num_t, dtype=jnp.float64)
    x_eval = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, dtype=jnp.float64)
    x_risk = jnp.linspace(cfg.risk_x_min, cfg.risk_x_max, cfg.risk_num_x, dtype=jnp.float64)
    t_pos = jnp.linspace(cfg.gp_t_delta, cfg.gp_t_max, cfg.gp_num_t, dtype=jnp.float64)
    gamma64 = jnp.float64(cfg.gamma)
    reward_std64 = jnp.float64(cfg.immediate_reward_std)
    taus_eval = jax.random.uniform(
        jax.random.PRNGKey(cfg.eval_iqn_seed),
        (cfg.num_taus_iqn_eval,),
        dtype=jnp.float32,
    )
    atoms_c51 = c51_atoms(cfg)
    taus_qr_eval = jnp.linspace(
        1.0 / (2.0 * cfg.num_taus_iqn_eval),
        1.0 - 1.0 / (2.0 * cfg.num_taus_iqn_eval),
        cfg.num_taus_iqn_eval,
        dtype=jnp.float32,
    )

    def per_action(params_iqn, params_qr_dqn, params_c51, params_cqn, params_mog, x_state, k):
        a_oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1 = x_state[None, :].astype(jnp.float32)
        a1 = a_oh[None, :]

        # IQN: query 5k taus, sort -> empirical CDF / CF.
        q_samples = iqn.apply({"params": params_iqn}, x1, a1, taus_eval[None, :])[0].astype(jnp.float64)
        sorted_q = jnp.sort(q_samples)
        F_iqn = empirical_cdf(sorted_q, x_eval)
        F_iqn_risk = empirical_cdf(sorted_q, x_risk)
        phi_iqn = empirical_cf_from_samples(
            q_samples.astype(jnp.complex128), t_eval.astype(jnp.complex128)
        )

        q_qr = qr_dqn.apply({"params": params_qr_dqn}, x1, a1, taus_qr_eval[None, :])[0].astype(jnp.float64)
        sorted_qr = jnp.sort(q_qr)
        F_qr_dqn = empirical_cdf(sorted_qr, x_eval)
        F_qr_dqn_risk = empirical_cdf(sorted_qr, x_risk)
        phi_qr_dqn = empirical_cf_from_samples(
            q_qr.astype(jnp.complex128), t_eval.astype(jnp.complex128)
        )

        logits_c51 = c51.apply({"params": params_c51}, x1, a1)[0]
        probs_c51 = jax.nn.softmax(logits_c51)
        F_c51 = discrete_return_cdf(probs_c51, atoms_c51, x_eval)
        F_c51_risk = discrete_return_cdf(probs_c51, atoms_c51, x_risk)
        phi_c51 = discrete_return_cf(probs_c51, atoms_c51, t_eval)

        # CQN: query the network at |ω|, then compose the parametric CF with signed ω so
        # that cf(-ω) = conj(cf(ω)) by construction (matches CQN's Gaussian-like form).
        t_eval_abs = jnp.abs(t_eval).astype(jnp.float32)
        mean_eval, scale_eval = cqn.apply({"params": params_cqn}, x1, a1, t_eval_abs[None, :])
        phi_cqn = compose_cf(t_eval, mean_eval[0], scale_eval[0])
        # Gil-Pelaez grid is positive by construction (t_pos > 0).
        mean_gp, scale_gp = cqn.apply({"params": params_cqn}, x1, a1, t_pos[None, :].astype(jnp.float32))
        phi_cqn_gp = compose_cf(t_pos, mean_gp[0], scale_gp[0])
        F_cqn = gil_pelaez_cdf(phi_cqn_gp, t_pos, x_eval)
        F_cqn_risk = gil_pelaez_cdf(phi_cqn_gp, t_pos, x_risk)
        # O(1) CQN moment extraction at omega=0.
        mean_zero, scale_zero = cqn.apply(
            {"params": params_cqn},
            x1,
            a1,
            jnp.zeros((1, 1), dtype=jnp.float32),
        )
        mean_cqn = mean_zero[0, 0].astype(jnp.float64)
        var_cqn = scale_zero[0, 0].astype(jnp.float64)
        
        #! Exact MoG CDF:
        # MoG-CQN: state-conditioned mixture with closed-form CF and exact spatial distributions.
        pi_eval, mu_eval, sigma_eval = mog.apply({"params": params_mog}, x1, a1)
        phi_mog_cqn = compose_mog_cf(t_eval, pi_eval[0], mu_eval[0], sigma_eval[0])

        # Exact CDF calculation (NO Gil-Pelaez error!)
        F_mog_cqn = exact_mog_cdf(x_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        F_mog_cqn_risk = exact_mog_cdf(x_risk, pi_eval[0], mu_eval[0], sigma_eval[0])
        
        #! MoG-CQN: state-conditioned mixture with closed-form CF.
        # pi_eval, mu_eval, sigma_eval = mog.apply({"params": params_mog}, x1, a1)
        # phi_mog_cqn = compose_mog_cf(t_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        # phi_mog_cqn_gp = compose_mog_cf(t_pos, pi_eval[0], mu_eval[0], sigma_eval[0])
        # F_mog_cqn = gil_pelaez_cdf(phi_mog_cqn_gp, t_pos, x_eval)
        # F_mog_cqn_risk = gil_pelaez_cdf(phi_mog_cqn_gp, t_pos, x_risk)

        # Ground truth Bellman target:
        # Z_target = R + gamma * Z_next, with R ~ N(0, immediate_reward_std^2).
        # In frequency domain:
        # phi_target(w) = phi_R(w) * phi_Z_next(gamma * w).
        phi_r_eval = jnp.exp(-0.5 * (reward_std64 ** 2) * jnp.square(t_eval))
        phi_z_eval = jax.lax.switch(
            k, [d.phi for d in ACTION_DISTS], gamma64 * t_eval, x_state.astype(jnp.float64)
        )
        phi_true = phi_r_eval * phi_z_eval

        phi_r_gp = jnp.exp(-0.5 * (reward_std64 ** 2) * jnp.square(t_pos))
        phi_z_gp = jax.lax.switch(
            k, [d.phi for d in ACTION_DISTS], gamma64 * t_pos, x_state.astype(jnp.float64)
        )
        phi_true_gp = phi_r_gp * phi_z_gp
        F_true = gil_pelaez_cdf(phi_true_gp, t_pos, x_eval)
        F_true_risk = gil_pelaez_cdf(phi_true_gp, t_pos, x_risk)
        pdf_true = jnp.clip(jnp.gradient(F_true, x_eval), 0.0, None)
        pdf_iqn = jnp.clip(jnp.gradient(F_iqn, x_eval), 0.0, None)
        pdf_qr_dqn = jnp.clip(jnp.gradient(F_qr_dqn, x_eval), 0.0, None)
        pdf_c51 = jnp.clip(jnp.gradient(F_c51, x_eval), 0.0, None)
        pdf_cqn = jnp.clip(jnp.gradient(F_cqn, x_eval), 0.0, None)
        pdf_mog_cqn = jnp.clip(jnp.gradient(F_mog_cqn, x_eval), 0.0, None)

        phi_mse_iqn = cf_mse_vs_truth(phi_iqn, phi_true)
        phi_mse_qr_dqn = cf_mse_vs_truth(phi_qr_dqn, phi_true)
        phi_mse_c51 = cf_mse_vs_truth(phi_c51, phi_true)
        phi_mse_cqn = cf_mse_vs_truth(phi_cqn, phi_true)
        phi_mse_mog_cqn = cf_mse_vs_truth(phi_mog_cqn, phi_true)
        ks_iqn = ks_on_grid(F_iqn, F_true)
        ks_qr_dqn = ks_on_grid(F_qr_dqn, F_true)
        ks_c51 = ks_on_grid(F_c51, F_true)
        ks_cqn = ks_on_grid(F_cqn, F_true)
        ks_mog_cqn = ks_on_grid(F_mog_cqn, F_true)
        w1_iqn_v = w1_cdf(F_iqn, F_true, x_eval)
        w1_qr_dqn_v = w1_cdf(F_qr_dqn, F_true, x_eval)
        w1_c51_v = w1_cdf(F_c51, F_true, x_eval)
        w1_cqn_v = w1_cdf(F_cqn, F_true, x_eval)
        w1_mog_cqn_v = w1_cdf(F_mog_cqn, F_true, x_eval)

        cvar_true = cvar_from_cdf(F_true_risk, x_risk, cfg.cvar_alpha)
        cvar_iqn = cvar_from_cdf(F_iqn_risk, x_risk, cfg.cvar_alpha)
        cvar_qr_dqn = cvar_from_cdf(F_qr_dqn_risk, x_risk, cfg.cvar_alpha)
        cvar_c51 = cvar_from_cdf(F_c51_risk, x_risk, cfg.cvar_alpha)
        cvar_cqn = cvar_from_cdf(F_cqn_risk, x_risk, cfg.cvar_alpha)
        cvar_mog_cqn = cvar_from_cdf(F_mog_cqn_risk, x_risk, cfg.cvar_alpha)
        er_true = entropic_risk_from_cdf(F_true_risk, x_risk, cfg.er_beta)
        er_iqn = entropic_risk_from_cdf(F_iqn_risk, x_risk, cfg.er_beta)
        er_qr_dqn = entropic_risk_from_cdf(F_qr_dqn_risk, x_risk, cfg.er_beta)
        er_c51 = entropic_risk_from_cdf(F_c51_risk, x_risk, cfg.er_beta)
        er_cqn = entropic_risk_from_cdf(F_cqn_risk, x_risk, cfg.er_beta)
        er_mog_cqn = entropic_risk_mog_cqn(pi_eval[0], mu_eval[0], sigma_eval[0], cfg.er_beta)

        mv_l64 = jnp.float64(cfg.mv_lambda)
        mean_true, var_true, mv_true = analytic_bellman_mean_var_mv(
            k, x_state, gamma64, reward_std64, mv_l64,
        )

        mean_iqn = jnp.mean(q_samples)
        var_iqn = jnp.mean(jnp.square(q_samples - mean_iqn))
        mv_iqn = mean_iqn - cfg.mv_lambda * var_iqn

        mean_qr_dqn = jnp.mean(q_qr)
        var_qr_dqn = jnp.mean(jnp.square(q_qr - mean_qr_dqn))
        mv_qr_dqn = mean_qr_dqn - cfg.mv_lambda * var_qr_dqn

        mean_c51 = jnp.sum(probs_c51 * atoms_c51).astype(jnp.float64)
        second_c51 = jnp.sum(probs_c51 * jnp.square(atoms_c51)).astype(jnp.float64)
        var_c51 = second_c51 - jnp.square(mean_c51)
        mv_c51 = mean_c51 - cfg.mv_lambda * var_c51

        mean_mog_cqn = jnp.sum(pi_eval[0] * mu_eval[0]).astype(jnp.float64)
        second_mog_cqn = jnp.sum(
            pi_eval[0] * (jnp.square(sigma_eval[0]) + jnp.square(mu_eval[0]))
        ).astype(jnp.float64)
        var_mog_cqn = second_mog_cqn - jnp.square(mean_mog_cqn)
        mv_mog_cqn = mean_mog_cqn - cfg.mv_lambda * var_mog_cqn
        mv_cqn = mean_cqn - cfg.mv_lambda * var_cqn

        return dict(
            phi_iqn=phi_iqn,
            phi_qr_dqn=phi_qr_dqn,
            phi_c51=phi_c51,
            phi_cqn=phi_cqn,
            phi_mog_cqn=phi_mog_cqn,
            phi_true=phi_true,
            F_iqn=F_iqn,
            F_qr_dqn=F_qr_dqn,
            F_c51=F_c51,
            F_cqn=F_cqn,
            F_mog_cqn=F_mog_cqn,
            F_true=F_true,
            pdf_true=pdf_true,
            pdf_iqn=pdf_iqn,
            pdf_qr_dqn=pdf_qr_dqn,
            pdf_c51=pdf_c51,
            pdf_cqn=pdf_cqn,
            pdf_mog_cqn=pdf_mog_cqn,
            phi_mse_iqn=phi_mse_iqn,
            phi_mse_qr_dqn=phi_mse_qr_dqn,
            phi_mse_c51=phi_mse_c51,
            phi_mse_cqn=phi_mse_cqn,
            phi_mse_mog_cqn=phi_mse_mog_cqn,
            ks_iqn=ks_iqn,
            ks_qr_dqn=ks_qr_dqn,
            ks_c51=ks_c51,
            ks_cqn=ks_cqn,
            ks_mog_cqn=ks_mog_cqn,
            w1_iqn=w1_iqn_v,
            w1_qr_dqn=w1_qr_dqn_v,
            w1_c51=w1_c51_v,
            w1_cqn=w1_cqn_v,
            w1_mog_cqn=w1_mog_cqn_v,
            cvar_err_iqn=jnp.abs(cvar_iqn - cvar_true),
            cvar_err_qr_dqn=jnp.abs(cvar_qr_dqn - cvar_true),
            cvar_err_c51=jnp.abs(cvar_c51 - cvar_true),
            cvar_err_cqn=jnp.abs(cvar_cqn - cvar_true),
            cvar_err_mog_cqn=jnp.abs(cvar_mog_cqn - cvar_true),
            er_err_iqn=jnp.abs(er_iqn - er_true),
            er_err_qr_dqn=jnp.abs(er_qr_dqn - er_true),
            er_err_c51=jnp.abs(er_c51 - er_true),
            er_err_cqn=jnp.abs(er_cqn - er_true),
            er_err_mog_cqn=jnp.abs(er_mog_cqn - er_true),
            mean_err_iqn=jnp.abs(mean_iqn - mean_true),
            mean_err_qr_dqn=jnp.abs(mean_qr_dqn - mean_true),
            mean_err_c51=jnp.abs(mean_c51 - mean_true),
            mean_err_cqn=jnp.abs(mean_cqn - mean_true),
            mean_err_mog_cqn=jnp.abs(mean_mog_cqn - mean_true),
            var_err_iqn=jnp.abs(var_iqn - var_true),
            var_err_qr_dqn=jnp.abs(var_qr_dqn - var_true),
            var_err_c51=jnp.abs(var_c51 - var_true),
            var_err_cqn=jnp.abs(var_cqn - var_true),
            var_err_mog_cqn=jnp.abs(var_mog_cqn - var_true),
            mv_err_iqn=jnp.abs(mv_iqn - mv_true),
            mv_err_qr_dqn=jnp.abs(mv_qr_dqn - mv_true),
            mv_err_c51=jnp.abs(mv_c51 - mv_true),
            mv_err_cqn=jnp.abs(mv_cqn - mv_true),
            mv_err_mog_cqn=jnp.abs(mv_mog_cqn - mv_true),
        )

    @jax.jit
    def eval_one_state(params_iqn, params_qr_dqn, params_c51, params_cqn, params_mog, x_state):
        per = [
            per_action(params_iqn, params_qr_dqn, params_c51, params_cqn, params_mog, x_state, k)
            for k in range(NUM_ACTIONS)
        ]
        out = {key: jnp.stack([p[key] for p in per], axis=0) for key in per[0]}
        return out

    return eval_one_state, t_eval, x_eval


def _make_eval_one_state_without_cqn(
    cfg: TrainConfig,
    iqn: IQNNet,
    qr_dqn: IQNNet,
    c51: C51Net,
    mog: MoGCQNNet,
):
    """IQN + QR-DQN + C51 + MoG-CQN (no parametric CQN)."""
    t_eval = jnp.linspace(-cfg.eval_t_max, cfg.eval_t_max, cfg.eval_num_t, dtype=jnp.float64)
    x_eval = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, dtype=jnp.float64)
    x_risk = jnp.linspace(cfg.risk_x_min, cfg.risk_x_max, cfg.risk_num_x, dtype=jnp.float64)
    t_pos = jnp.linspace(cfg.gp_t_delta, cfg.gp_t_max, cfg.gp_num_t, dtype=jnp.float64)
    gamma64 = jnp.float64(cfg.gamma)
    reward_std64 = jnp.float64(cfg.immediate_reward_std)
    taus_eval = jax.random.uniform(
        jax.random.PRNGKey(cfg.eval_iqn_seed),
        (cfg.num_taus_iqn_eval,),
        dtype=jnp.float32,
    )
    atoms_c51 = c51_atoms(cfg)
    taus_qr_eval = jnp.linspace(
        1.0 / (2.0 * cfg.num_taus_iqn_eval),
        1.0 - 1.0 / (2.0 * cfg.num_taus_iqn_eval),
        cfg.num_taus_iqn_eval,
        dtype=jnp.float32,
    )

    def per_action(params_iqn, params_qr_dqn, params_c51, params_mog, x_state, k):
        a_oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1 = x_state[None, :].astype(jnp.float32)
        a1 = a_oh[None, :]

        q_samples = iqn.apply({"params": params_iqn}, x1, a1, taus_eval[None, :])[0].astype(jnp.float64)
        sorted_q = jnp.sort(q_samples)
        F_iqn = empirical_cdf(sorted_q, x_eval)
        F_iqn_risk = empirical_cdf(sorted_q, x_risk)
        phi_iqn = empirical_cf_from_samples(
            q_samples.astype(jnp.complex128), t_eval.astype(jnp.complex128)
        )

        q_qr = qr_dqn.apply({"params": params_qr_dqn}, x1, a1, taus_qr_eval[None, :])[0].astype(jnp.float64)
        sorted_qr = jnp.sort(q_qr)
        F_qr_dqn = empirical_cdf(sorted_qr, x_eval)
        F_qr_dqn_risk = empirical_cdf(sorted_qr, x_risk)
        phi_qr_dqn = empirical_cf_from_samples(
            q_qr.astype(jnp.complex128), t_eval.astype(jnp.complex128)
        )

        logits_c51 = c51.apply({"params": params_c51}, x1, a1)[0]
        probs_c51 = jax.nn.softmax(logits_c51)
        F_c51 = discrete_return_cdf(probs_c51, atoms_c51, x_eval)
        F_c51_risk = discrete_return_cdf(probs_c51, atoms_c51, x_risk)
        phi_c51 = discrete_return_cf(probs_c51, atoms_c51, t_eval)

        pi_eval, mu_eval, sigma_eval = mog.apply({"params": params_mog}, x1, a1)
        phi_mog_cqn = compose_mog_cf(t_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        F_mog_cqn = exact_mog_cdf(x_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        F_mog_cqn_risk = exact_mog_cdf(x_risk, pi_eval[0], mu_eval[0], sigma_eval[0])

        phi_r_eval = jnp.exp(-0.5 * (reward_std64 ** 2) * jnp.square(t_eval))
        phi_z_eval = jax.lax.switch(
            k, [d.phi for d in ACTION_DISTS], gamma64 * t_eval, x_state.astype(jnp.float64)
        )
        phi_true = phi_r_eval * phi_z_eval

        phi_r_gp = jnp.exp(-0.5 * (reward_std64 ** 2) * jnp.square(t_pos))
        phi_z_gp = jax.lax.switch(
            k, [d.phi for d in ACTION_DISTS], gamma64 * t_pos, x_state.astype(jnp.float64)
        )
        phi_true_gp = phi_r_gp * phi_z_gp
        F_true = gil_pelaez_cdf(phi_true_gp, t_pos, x_eval)
        F_true_risk = gil_pelaez_cdf(phi_true_gp, t_pos, x_risk)
        pdf_true = jnp.clip(jnp.gradient(F_true, x_eval), 0.0, None)
        pdf_iqn = jnp.clip(jnp.gradient(F_iqn, x_eval), 0.0, None)
        pdf_qr_dqn = jnp.clip(jnp.gradient(F_qr_dqn, x_eval), 0.0, None)
        pdf_c51 = jnp.clip(jnp.gradient(F_c51, x_eval), 0.0, None)
        pdf_mog_cqn = jnp.clip(jnp.gradient(F_mog_cqn, x_eval), 0.0, None)

        phi_mse_iqn = cf_mse_vs_truth(phi_iqn, phi_true)
        phi_mse_qr_dqn = cf_mse_vs_truth(phi_qr_dqn, phi_true)
        phi_mse_c51 = cf_mse_vs_truth(phi_c51, phi_true)
        phi_mse_mog_cqn = cf_mse_vs_truth(phi_mog_cqn, phi_true)
        ks_iqn = ks_on_grid(F_iqn, F_true)
        ks_qr_dqn = ks_on_grid(F_qr_dqn, F_true)
        ks_c51 = ks_on_grid(F_c51, F_true)
        ks_mog_cqn = ks_on_grid(F_mog_cqn, F_true)
        w1_iqn_v = w1_cdf(F_iqn, F_true, x_eval)
        w1_qr_dqn_v = w1_cdf(F_qr_dqn, F_true, x_eval)
        w1_c51_v = w1_cdf(F_c51, F_true, x_eval)
        w1_mog_cqn_v = w1_cdf(F_mog_cqn, F_true, x_eval)

        cvar_true = cvar_from_cdf(F_true_risk, x_risk, cfg.cvar_alpha)
        cvar_iqn = cvar_from_cdf(F_iqn_risk, x_risk, cfg.cvar_alpha)
        cvar_qr_dqn = cvar_from_cdf(F_qr_dqn_risk, x_risk, cfg.cvar_alpha)
        cvar_c51 = cvar_from_cdf(F_c51_risk, x_risk, cfg.cvar_alpha)
        cvar_mog_cqn = cvar_from_cdf(F_mog_cqn_risk, x_risk, cfg.cvar_alpha)
        er_true = entropic_risk_from_cdf(F_true_risk, x_risk, cfg.er_beta)
        er_iqn = entropic_risk_from_cdf(F_iqn_risk, x_risk, cfg.er_beta)
        er_qr_dqn = entropic_risk_from_cdf(F_qr_dqn_risk, x_risk, cfg.er_beta)
        er_c51 = entropic_risk_from_cdf(F_c51_risk, x_risk, cfg.er_beta)
        er_mog_cqn = entropic_risk_mog_cqn(pi_eval[0], mu_eval[0], sigma_eval[0], cfg.er_beta)

        mv_l64 = jnp.float64(cfg.mv_lambda)
        mean_true, var_true, mv_true = analytic_bellman_mean_var_mv(
            k, x_state, gamma64, reward_std64, mv_l64,
        )

        mean_iqn = jnp.mean(q_samples)
        var_iqn = jnp.mean(jnp.square(q_samples - mean_iqn))
        mv_iqn = mean_iqn - cfg.mv_lambda * var_iqn

        mean_qr_dqn = jnp.mean(q_qr)
        var_qr_dqn = jnp.mean(jnp.square(q_qr - mean_qr_dqn))
        mv_qr_dqn = mean_qr_dqn - cfg.mv_lambda * var_qr_dqn

        mean_c51 = jnp.sum(probs_c51 * atoms_c51).astype(jnp.float64)
        second_c51 = jnp.sum(probs_c51 * jnp.square(atoms_c51)).astype(jnp.float64)
        var_c51 = second_c51 - jnp.square(mean_c51)
        mv_c51 = mean_c51 - cfg.mv_lambda * var_c51

        mean_mog_cqn = jnp.sum(pi_eval[0] * mu_eval[0]).astype(jnp.float64)
        second_mog_cqn = jnp.sum(
            pi_eval[0] * (jnp.square(sigma_eval[0]) + jnp.square(mu_eval[0]))
        ).astype(jnp.float64)
        var_mog_cqn = second_mog_cqn - jnp.square(mean_mog_cqn)
        mv_mog_cqn = mean_mog_cqn - cfg.mv_lambda * var_mog_cqn

        return dict(
            phi_iqn=phi_iqn,
            phi_qr_dqn=phi_qr_dqn,
            phi_c51=phi_c51,
            phi_mog_cqn=phi_mog_cqn,
            phi_true=phi_true,
            F_iqn=F_iqn,
            F_qr_dqn=F_qr_dqn,
            F_c51=F_c51,
            F_mog_cqn=F_mog_cqn,
            F_true=F_true,
            pdf_true=pdf_true,
            pdf_iqn=pdf_iqn,
            pdf_qr_dqn=pdf_qr_dqn,
            pdf_c51=pdf_c51,
            pdf_mog_cqn=pdf_mog_cqn,
            phi_mse_iqn=phi_mse_iqn,
            phi_mse_qr_dqn=phi_mse_qr_dqn,
            phi_mse_c51=phi_mse_c51,
            phi_mse_mog_cqn=phi_mse_mog_cqn,
            ks_iqn=ks_iqn,
            ks_qr_dqn=ks_qr_dqn,
            ks_c51=ks_c51,
            ks_mog_cqn=ks_mog_cqn,
            w1_iqn=w1_iqn_v,
            w1_qr_dqn=w1_qr_dqn_v,
            w1_c51=w1_c51_v,
            w1_mog_cqn=w1_mog_cqn_v,
            cvar_err_iqn=jnp.abs(cvar_iqn - cvar_true),
            cvar_err_qr_dqn=jnp.abs(cvar_qr_dqn - cvar_true),
            cvar_err_c51=jnp.abs(cvar_c51 - cvar_true),
            cvar_err_mog_cqn=jnp.abs(cvar_mog_cqn - cvar_true),
            er_err_iqn=jnp.abs(er_iqn - er_true),
            er_err_qr_dqn=jnp.abs(er_qr_dqn - er_true),
            er_err_c51=jnp.abs(er_c51 - er_true),
            er_err_mog_cqn=jnp.abs(er_mog_cqn - er_true),
            mean_err_iqn=jnp.abs(mean_iqn - mean_true),
            mean_err_qr_dqn=jnp.abs(mean_qr_dqn - mean_true),
            mean_err_c51=jnp.abs(mean_c51 - mean_true),
            mean_err_mog_cqn=jnp.abs(mean_mog_cqn - mean_true),
            var_err_iqn=jnp.abs(var_iqn - var_true),
            var_err_qr_dqn=jnp.abs(var_qr_dqn - var_true),
            var_err_c51=jnp.abs(var_c51 - var_true),
            var_err_mog_cqn=jnp.abs(var_mog_cqn - var_true),
            mv_err_iqn=jnp.abs(mv_iqn - mv_true),
            mv_err_qr_dqn=jnp.abs(mv_qr_dqn - mv_true),
            mv_err_c51=jnp.abs(mv_c51 - mv_true),
            mv_err_mog_cqn=jnp.abs(mv_mog_cqn - mv_true),
        )

    @jax.jit
    def eval_one_state(params_iqn, params_qr_dqn, params_c51, params_mog, x_state):
        per = [
            per_action(params_iqn, params_qr_dqn, params_c51, params_mog, x_state, kk)
            for kk in range(NUM_ACTIONS)
        ]
        out = {key: jnp.stack([p[key] for p in per], axis=0) for key in per[0]}
        return out

    return eval_one_state, t_eval, x_eval


def _make_eval_one_state_mog_only(cfg: TrainConfig, mog: MoGCQNNet):
    """Ground truth + MoG-CQN only (same metrics as full run for MoG rows)."""
    t_eval = jnp.linspace(-cfg.eval_t_max, cfg.eval_t_max, cfg.eval_num_t, dtype=jnp.float64)
    x_eval = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, dtype=jnp.float64)
    x_risk = jnp.linspace(cfg.risk_x_min, cfg.risk_x_max, cfg.risk_num_x, dtype=jnp.float64)
    t_pos = jnp.linspace(cfg.gp_t_delta, cfg.gp_t_max, cfg.gp_num_t, dtype=jnp.float64)
    gamma64 = jnp.float64(cfg.gamma)
    reward_std64 = jnp.float64(cfg.immediate_reward_std)

    def per_action(params_mog, x_state, k):
        a_oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1 = x_state[None, :].astype(jnp.float32)
        a1 = a_oh[None, :]

        pi_eval, mu_eval, sigma_eval = mog.apply({"params": params_mog}, x1, a1)
        phi_mog_cqn = compose_mog_cf(t_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        F_mog_cqn = exact_mog_cdf(x_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        F_mog_cqn_risk = exact_mog_cdf(x_risk, pi_eval[0], mu_eval[0], sigma_eval[0])

        phi_r_eval = jnp.exp(-0.5 * (reward_std64 ** 2) * jnp.square(t_eval))
        phi_z_eval = jax.lax.switch(
            k, [d.phi for d in ACTION_DISTS], gamma64 * t_eval, x_state.astype(jnp.float64)
        )
        phi_true = phi_r_eval * phi_z_eval

        phi_r_gp = jnp.exp(-0.5 * (reward_std64 ** 2) * jnp.square(t_pos))
        phi_z_gp = jax.lax.switch(
            k, [d.phi for d in ACTION_DISTS], gamma64 * t_pos, x_state.astype(jnp.float64)
        )
        phi_true_gp = phi_r_gp * phi_z_gp
        F_true = gil_pelaez_cdf(phi_true_gp, t_pos, x_eval)
        F_true_risk = gil_pelaez_cdf(phi_true_gp, t_pos, x_risk)
        pdf_true = jnp.clip(jnp.gradient(F_true, x_eval), 0.0, None)
        pdf_mog_cqn = jnp.clip(jnp.gradient(F_mog_cqn, x_eval), 0.0, None)

        phi_mse_mog_cqn = cf_mse_vs_truth(phi_mog_cqn, phi_true)
        ks_mog_cqn = ks_on_grid(F_mog_cqn, F_true)
        w1_mog_cqn_v = w1_cdf(F_mog_cqn, F_true, x_eval)

        cvar_true = cvar_from_cdf(F_true_risk, x_risk, cfg.cvar_alpha)
        cvar_mog_cqn = cvar_from_cdf(F_mog_cqn_risk, x_risk, cfg.cvar_alpha)
        er_true = entropic_risk_from_cdf(F_true_risk, x_risk, cfg.er_beta)
        er_mog_cqn = entropic_risk_mog_cqn(pi_eval[0], mu_eval[0], sigma_eval[0], cfg.er_beta)

        mv_l64 = jnp.float64(cfg.mv_lambda)
        mean_true, var_true, mv_true = analytic_bellman_mean_var_mv(
            k, x_state, gamma64, reward_std64, mv_l64,
        )

        mean_mog_cqn = jnp.sum(pi_eval[0] * mu_eval[0]).astype(jnp.float64)
        second_mog_cqn = jnp.sum(
            pi_eval[0] * (jnp.square(sigma_eval[0]) + jnp.square(mu_eval[0]))
        ).astype(jnp.float64)
        var_mog_cqn = second_mog_cqn - jnp.square(mean_mog_cqn)
        mv_mog_cqn = mean_mog_cqn - cfg.mv_lambda * var_mog_cqn

        return dict(
            phi_mog_cqn=phi_mog_cqn,
            phi_true=phi_true,
            F_mog_cqn=F_mog_cqn,
            F_true=F_true,
            pdf_true=pdf_true,
            pdf_mog_cqn=pdf_mog_cqn,
            phi_mse_mog_cqn=phi_mse_mog_cqn,
            ks_mog_cqn=ks_mog_cqn,
            w1_mog_cqn=w1_mog_cqn_v,
            cvar_err_mog_cqn=jnp.abs(cvar_mog_cqn - cvar_true),
            er_err_mog_cqn=jnp.abs(er_mog_cqn - er_true),
            mean_err_mog_cqn=jnp.abs(mean_mog_cqn - mean_true),
            var_err_mog_cqn=jnp.abs(var_mog_cqn - var_true),
            mv_err_mog_cqn=jnp.abs(mv_mog_cqn - mv_true),
        )

    @jax.jit
    def eval_one_state(params_mog, x_state):
        per = [per_action(params_mog, x_state, kk) for kk in range(NUM_ACTIONS)]
        out = {key: jnp.stack([p[key] for p in per], axis=0) for key in per[0]}
        return out

    return eval_one_state, t_eval, x_eval


def make_eval_one_state(
    cfg: TrainConfig,
    iqn: IQNNet,
    qr_dqn: IQNNet,
    c51: C51Net,
    cqn: CQNNet | None,
    mog: MoGCQNNet,
):
    if cfg.train_mog_cqn_only:
        return _make_eval_one_state_mog_only(cfg, mog)
    if cfg.train_cqn:
        assert cqn is not None
        return _make_eval_one_state_with_cqn(cfg, iqn, qr_dqn, c51, cqn, mog)
    return _make_eval_one_state_without_cqn(cfg, iqn, qr_dqn, c51, mog)


def evaluate_test_set(
    cfg: TrainConfig,
    iqn: IQNNet,
    qr_dqn: IQNNet,
    c51: C51Net,
    cqn: CQNNet | None,
    mog: MoGCQNNet,
    params_iqn,
    params_qr_dqn,
    params_c51,
    params_cqn,
    params_mog,
    x_test: jax.Array,
):
    """Vmapped over a chunk of test states; iterated over chunks."""
    eval_one, t_eval, x_eval = make_eval_one_state(cfg, iqn, qr_dqn, c51, cqn, mog)
    if cfg.train_mog_cqn_only:
        eval_chunk = jax.jit(jax.vmap(eval_one, in_axes=(None, 0)))
    elif cfg.train_cqn:
        assert params_cqn is not None
        eval_chunk = jax.jit(
            jax.vmap(eval_one, in_axes=(None, None, None, None, None, 0))
        )
    else:
        eval_chunk = jax.jit(jax.vmap(eval_one, in_axes=(None, None, None, None, 0)))

    n = x_test.shape[0]
    cs = cfg.eval_chunk_size
    assert n % cs == 0, f"n_test={n} must be divisible by eval_chunk_size={cs}"
    n_chunks = n // cs
    x_chunks = x_test.reshape(n_chunks, cs, -1)

    chunks_out = []
    for c in range(n_chunks):
        if cfg.train_mog_cqn_only:
            chunks_out.append(eval_chunk(params_mog, x_chunks[c]))
        elif cfg.train_cqn:
            chunks_out.append(
                eval_chunk(params_iqn, params_qr_dqn, params_c51, params_cqn, params_mog, x_chunks[c])
            )
        else:
            chunks_out.append(
                eval_chunk(params_iqn, params_qr_dqn, params_c51, params_mog, x_chunks[c])
            )
    # Concatenate over leading axis
    out = {
        key: jnp.concatenate([co[key] for co in chunks_out], axis=0)
        for key in chunks_out[0]
    }
    # `out[key]` shape: (n_test, NUM_ACTIONS, ...)
    return out, np.asarray(t_eval), np.asarray(x_eval)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _draw_density_column(
    ax,
    pdf_true,
    hist_samples,
    x_grid,
    plot_x_min: float,
    plot_x_max: float,
    *,
    hist_label: str = "Histogram",
    truth_label: str = "Truth PDF",
) -> None:
    """Normalized histogram of Bellman-target samples + analytic truth PDF (black line)."""
    mask = (x_grid >= plot_x_min) & (x_grid <= plot_x_max)
    x_m = np.asarray(x_grid)[mask]
    ax.hist(
        np.asarray(hist_samples),
        bins=72,
        range=(plot_x_min, plot_x_max),
        density=True,
        color="#E8E8E8",
        edgecolor="#CCCCCC",
        linewidth=0.2,
        label=hist_label,
        zorder=1,
    )
    ax.plot(
        x_m,
        np.asarray(pdf_true)[mask],
        color=COLORS["truth"],
        lw=LW_GT,
        label=truth_label,
        zorder=3,
    )
    minimal_style(ax)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_xlabel(r"$x$")


def _draw_pdf_curves(ax, x_grid, pdf_true, plot_x_min, plot_x_max, **pdf_by_algo):
    """Plot analytic truth PDF plus model PDF curves (finite-difference from learned CDFs)."""
    mask = (x_grid >= plot_x_min) & (x_grid <= plot_x_max)
    x_m = np.asarray(x_grid)[mask]
    ax.plot(x_m, np.asarray(pdf_true)[mask], color=COLORS["truth"], lw=LW_GT, label="Truth", zorder=6)
    draw_order = (
        ("pdf_iqn", "IQN", COLORS["iqn"], LW_IQN),
        ("pdf_qr_dqn", "QR-DQN", COLORS["qr_dqn"], LW_QR),
        ("pdf_c51", "C51", COLORS["c51"], LW_C51),
        ("pdf_cqn", "CQN", COLORS["cqn"], LW_CQN),
        ("pdf_mog_cqn", "MoG-CQN", COLORS["mog_cqn"], LW_MOG),
    )
    for key, lbl, color, lw in draw_order:
        arr = pdf_by_algo.get(key)
        if arr is None:
            continue
        ax.plot(x_m, np.asarray(arr)[mask], color=color, lw=lw, alpha=0.92, label=lbl, zorder=2)
    minimal_style(ax)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_xlabel(r"$x$")


def _draw_phi_panel(ax, t_grid, phi_true, phi_iqn, cf_omega_half: float, *,
                    phi_qr_dqn=None, phi_c51=None, phi_cqn=None, phi_mog_cqn=None):
    """CF x-axis: ``[-cf_omega_half, +cf_omega_half]`` (same as ``TrainConfig.eval_t_max`` / training ω band)."""
    m = np.abs(t_grid) <= cf_omega_half
    ax.plot(t_grid[m], np.real(phi_true)[m], color=COLORS["truth"], lw=LW_GT, label="Truth", zorder=1)
    if phi_c51 is not None:
        ax.plot(t_grid[m], np.real(phi_c51)[m], color=COLORS["c51"], lw=LW_C51, alpha=0.95, label="C51", zorder=2)
    if phi_qr_dqn is not None:
        ax.plot(t_grid[m], np.real(phi_qr_dqn)[m], color=COLORS["qr_dqn"], lw=LW_QR, alpha=0.95, label="QR-DQN", zorder=3)
    ax.plot(t_grid[m], np.real(phi_iqn)[m], color=COLORS["iqn"], lw=LW_IQN, alpha=0.95, label="IQN", zorder=4)
    if phi_cqn is not None:
        ax.plot(t_grid[m], np.real(phi_cqn)[m], color=COLORS["cqn"], lw=LW_CQN, label="CQN", zorder=5)
    if phi_mog_cqn is not None:
        ax.plot(t_grid[m], np.real(phi_mog_cqn)[m], color=COLORS["mog_cqn"], lw=LW_MOG, alpha=0.95, label="MoG-CQN", zorder=6)
    minimal_style(ax)
    ax.set_xlim(-cf_omega_half, cf_omega_half)
    ax.set_xlabel(r"$\omega$")


def _draw_cdf_panel(ax, x_grid, F_true, F_iqn, plot_x_min, plot_x_max, *,
                    F_qr_dqn=None, F_c51=None, F_cqn=None, F_mog_cqn=None):
    m = (x_grid >= plot_x_min) & (x_grid <= plot_x_max)
    ax.plot(x_grid[m], F_true[m], color=COLORS["truth"], lw=LW_GT, label="Truth", zorder=1)
    if F_c51 is not None:
        ax.plot(x_grid[m], F_c51[m], color=COLORS["c51"], lw=LW_C51, alpha=0.95, label="C51", zorder=2)
    if F_qr_dqn is not None:
        ax.plot(x_grid[m], F_qr_dqn[m], color=COLORS["qr_dqn"], lw=LW_QR, alpha=0.95, label="QR-DQN", zorder=3)
    ax.plot(x_grid[m], F_iqn[m], color=COLORS["iqn"], lw=LW_IQN, alpha=0.95, label="IQN", zorder=4)
    if F_cqn is not None:
        ax.plot(x_grid[m], F_cqn[m], color=COLORS["cqn"], lw=LW_CQN, label="CQN", zorder=5)
    if F_mog_cqn is not None:
        ax.plot(x_grid[m], F_mog_cqn[m], color=COLORS["mog_cqn"], lw=LW_MOG, alpha=0.95, label="MoG-CQN", zorder=6)
    minimal_style(ax)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_xlabel(r"$x$")


def plot_panels_figure(
    panel_data: dict,
    t_eval: np.ndarray,
    x_eval: np.ndarray,
    out_path: str,
    *,
    cf_omega_half: float,
    png_dpi: int = 300,
) -> None:
    """``NUM_ACTIONS`` × 4 grid: Density (hist + truth PDF) | CF | PDF | CDF.

    ``cf_omega_half`` must match ``TrainConfig.eval_t_max`` so CF panels show ``[-ω_max, ω_max]``.
    """
    n_rows = NUM_ACTIONS
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.5, 2.45 * n_rows), sharey=False)
    legend_handles, legend_labels = None, None

    for row, dist in enumerate(ACTION_DISTS):
        ax_den, ax_phi, ax_pdf, ax_cdf = axes[row]
        pdf_true = panel_data["pdf_true"][row]
        hist = panel_data["hist_samples"][row]
        _draw_density_column(ax_den, pdf_true, hist, x_eval, dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            ax_den.set_title("Density", fontsize=10, color="#2E2E2E")
        ax_den.set_ylabel("density")

        phi_cqn_row = panel_data["phi_cqn"][row] if "phi_cqn" in panel_data else None
        _draw_phi_panel(
            ax_phi, t_eval,
            panel_data["phi_true"][row],
            panel_data["phi_iqn"][row],
            cf_omega_half,
            phi_qr_dqn=panel_data["phi_qr_dqn"][row] if "phi_qr_dqn" in panel_data else None,
            phi_c51=panel_data["phi_c51"][row] if "phi_c51" in panel_data else None,
            phi_cqn=phi_cqn_row,
            phi_mog_cqn=panel_data["phi_mog_cqn"][row] if "phi_mog_cqn" in panel_data else None,
        )
        if row == 0:
            ax_phi.set_title("CF", fontsize=10, color="#2E2E2E")
        ax_phi.set_ylabel(r"$\mathrm{Re}\,\varphi(\omega)$")

        kw_pdf = {"pdf_true": pdf_true}
        for key in ("pdf_iqn", "pdf_qr_dqn", "pdf_c51", "pdf_cqn", "pdf_mog_cqn"):
            if key in panel_data:
                kw_pdf[key] = panel_data[key][row]
        _draw_pdf_curves(ax_pdf, x_eval, pdf_true, dist.plot_x_min, dist.plot_x_max, **kw_pdf)
        if row == 0:
            ax_pdf.set_title("PDF", fontsize=10, color="#2E2E2E")
        ax_pdf.set_ylabel("density")

        F_cqn_row = panel_data["F_cqn"][row] if "F_cqn" in panel_data else None
        _draw_cdf_panel(
            ax_cdf, x_eval,
            panel_data["F_true"][row],
            panel_data["F_iqn"][row],
            dist.plot_x_min, dist.plot_x_max,
            F_qr_dqn=panel_data["F_qr_dqn"][row] if "F_qr_dqn" in panel_data else None,
            F_c51=panel_data["F_c51"][row] if "F_c51" in panel_data else None,
            F_cqn=F_cqn_row,
            F_mog_cqn=panel_data["F_mog_cqn"][row] if "F_mog_cqn" in panel_data else None,
        )
        if row == 0:
            ax_cdf.set_title("CDF", fontsize=10, color="#2E2E2E")
        ax_cdf.set_ylabel(r"$F(x)$")

        ax_den.text(
            -0.34, 0.5, dist.name,
            transform=ax_den.transAxes,
            rotation=90, ha="center", va="center",
            fontsize=10, color="#2E2E2E", fontweight="bold",
        )

        if legend_handles is None:
            handles, labels = ax_cdf.get_legend_handles_labels()
            legend_handles, legend_labels = handles, labels

    if legend_handles is not None:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.035),
            bbox_transform=fig.transFigure,
            ncol=min(8, len(legend_labels)),
            frameon=False, fontsize=8, labelcolor="#2E2E2E",
        )
    fig.suptitle("White-box bandit: per-action distributional fits (one test state)",
                 fontsize=11, y=0.995, color="#2E2E2E")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.09, wspace=0.30, hspace=0.42)
    _save_figure_pdf_png(fig, out_path, png_dpi=png_dpi)
    plt.close(fig)


def _fmt_metric_display(v: float) -> str:
    """Human-readable table cell (undefined Cauchy moments → em dash)."""
    xf = float(v)
    if np.isnan(xf):
        return "—"
    return f"{xf:.3g}"


def _fmt_metric_csv(v: float) -> str:
    """CSV cell; use ``nan`` so aggregate scripts can ``float(...)`` round-trip."""
    xf = float(v)
    if np.isnan(xf):
        return "nan"
    return f"{xf:.3g}"


ALGO_TITLES = {
    "iqn": "IQN (implicit quantiles)",
    "qr_dqn": "QR-DQN (fixed τ)",
    "c51": "C51 (categorical)",
    "cqn": "CQN (frequency)",
    "mog_cqn": "MoG-CQN (frequency)",
}

ALGO_ORDER = ("iqn", "qr_dqn", "c51", "cqn", "mog_cqn")

_METRIC_KEYS = ("phi_mse", "ks", "w1", "cvar_err", "er_err", "mean_err", "var_err", "mv_err")
_METRIC_ROW_LABELS = (
    r"Frequency — $\varphi$-MSE ($\downarrow$)",
    r"Spatial — KS distance ($\downarrow$)",
    r"Spatial — $W_1$ distance ($\downarrow$)",
    rf"Risk-grid — CVaR$_{{0.05}}$ error ($\downarrow$)",
    r"Risk-grid — Entropic-risk error ($\downarrow$)",
    r"Risk-grid — Mean error ($\downarrow$)",
    r"Risk-grid — Variance error ($\downarrow$)",
    r"Risk-grid — M-V risk error ($\downarrow$)",
)
_WIN_FACECOLOR = "#D2F4D9"  # soft green for the better cell (lower = better)


def _per_action_means(per_state_metrics: dict, algo: str) -> np.ndarray:
    """Return a (n_metrics, n_actions) array of test-set means for one algorithm."""
    return np.stack(
        [np.mean(np.asarray(per_state_metrics[f"{mk}_{algo}"]), axis=0) for mk in _METRIC_KEYS],
        axis=0,
    )


def _available_algos(per_state_metrics: dict) -> list[str]:
    out: list[str] = []
    for algo in ALGO_ORDER:
        if algo not in ALGO_TITLES:
            continue
        if all(f"{mk}_{algo}" in per_state_metrics for mk in _METRIC_KEYS):
            out.append(algo)
    return out


def wandb_log_whitebox_csv_metrics(per_state_metrics: dict, cfg: TrainConfig, experiment_tag: str) -> None:
    """Log the same per-cell scalar means as ``save_metrics_csv`` (mean over test states per metric/action)."""
    import wandb

    payload: dict[str, float] = {}
    for algo in _available_algos(per_state_metrics):
        means = _per_action_means(per_state_metrics, algo)
        for mi, mk in enumerate(_METRIC_KEYS):
            for ai, dist in enumerate(ACTION_DISTS):
                v = float(means[mi, ai])
                if not np.isnan(v):
                    payload[f"metrics/{algo}/{mk}/{dist.short}"] = v
    wandb.log(payload)
    wandb.summary["wandb_experiment_tag"] = experiment_tag
    wandb.summary["figure_tag"] = cfg.figure_tag


def wandb_maybe_init(cfg: TrainConfig, experiment_tag: str) -> None:
    """Group runs by ``experiment_tag`` (filters in W&B UI); run name includes seed."""
    import wandb

    project = os.environ.get("WANDB_PROJECT", "Deep-CVI-Experiments")
    entity = os.environ.get("WANDB_ENTITY")
    mode = os.environ.get("WANDB_MODE", "online")
    kwargs = dict(
        project=project,
        tags=[experiment_tag, "whitebox_bandit_supervised"],
        group=experiment_tag,
        name=f"{experiment_tag}_seed{cfg.seed}",
        mode=mode,
        config=asdict(cfg),
    )
    if entity:
        kwargs["entity"] = entity
    wandb.init(**kwargs)


def plot_metrics_table_figure_combined(per_state_metrics: dict, out_path: str) -> None:
    """
    Combined comparison figure with one table per algorithm stacked vertically.
    Best value per (metric, action) across all algorithms is highlighted in green.
    """
    available_algos = _available_algos(per_state_metrics)
    if not available_algos:
        return

    all_means = np.stack([_per_action_means(per_state_metrics, algo) for algo in available_algos], axis=0)
    # (n_algos, n_metrics, n_actions) -> global argmin over algo axis (ignore NaNs; no winner if all NaN)
    inf_rep = np.where(np.isfinite(all_means), all_means, np.inf)
    best_algo_idx = np.argmin(inf_rep, axis=0)
    no_winner = ~np.isfinite(all_means).any(axis=0)
    best_algo_idx = np.where(no_winner, -1, best_algo_idx)

    n_algos = len(available_algos)
    fig_h = 2.2 * n_algos + 0.7
    fig, axes = plt.subplots(n_algos, 1, figsize=(8.3, fig_h))
    if n_algos == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.34)

    row_labels = list(_METRIC_ROW_LABELS)
    col_labels = [d.name for d in ACTION_DISTS]

    for i, (ax, algo) in enumerate(zip(axes, available_algos)):
        ax.axis("off")
        means = all_means[i]
        cells = [[_fmt_metric_display(float(v)) for v in row] for row in means]
        tbl = ax.table(
            cellText=cells, rowLabels=row_labels, colLabels=col_labels,
            loc="center", cellLoc="center", colLoc="center", rowLoc="right",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.05, 1.35)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col < 0:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.6)
            if (
                row >= 1 and col >= 0
                and best_algo_idx[row - 1, col] == i
                and best_algo_idx[row - 1, col] >= 0
            ):
                cell.set_facecolor(_WIN_FACECOLOR)
                cell.set_text_props(fontweight="bold", color="#0A0A0A")

        title = ALGO_TITLES.get(algo, algo)
        ax.text(
            0.5, 1.02,
            f"{title}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9, color="#2E2E2E", fontweight="bold",
        )

    fig.suptitle(
        "Per-action distributional & risk metrics (mean over test states)",
        fontsize=11, y=0.985, color="#2E2E2E",
    )
    short = " / ".join(ALGO_TITLES[a].split(" (")[0] for a in available_algos)
    fig.text(
        0.5, 0.02,
        f"Green cells: best across {short} for that (metric, action) pair (lower is better). "
        "Plots use x in [-15, 15]; risk metrics use the dual-grid x in [-200, 200].",
        ha="center", fontsize=7, color="#5A5A5A",
    )
    _save_figure_pdf_png(fig, out_path)
    plt.close(fig)


def load_metrics_csv_matrix(path: str) -> np.ndarray:
    """Return ``(n_metrics, n_actions)`` floats (same row order as ``save_metrics_csv``)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        raise ValueError(f"Empty or header-only metrics CSV: {path}")
    body = rows[1:]
    return np.asarray([[float(c) for c in r[1:]] for r in body], dtype=np.float64)


def discover_seed_metric_csvs(csv_dir: str, tag_stem: str) -> dict[str, list[tuple[int, str]]]:
    """
    Find ``whitebox_bandit_metrics_<algo>{tag_stem}_seed<N>.csv`` files (same naming as training).
    ``tag_stem`` is the figure-tag prefix before ``_seed``, e.g. ``_omega_15``.
    Returns mapping algo -> sorted list of (seed_int, path).
    """
    rx = re.compile(
        r"^whitebox_bandit_metrics_(iqn|qr_dqn|c51|cqn|mog_cqn)(.+)\.csv$"
    )
    seed_suffix_re = re.compile(r"^" + re.escape(tag_stem) + r"_seed(\d+)$")
    found: dict[str, list[tuple[int, str]]] = {}
    for name in os.listdir(csv_dir):
        m = rx.match(name)
        if not m:
            continue
        algo, rest = m.group(1), m.group(2)
        ms = seed_suffix_re.match(rest)
        if not ms:
            continue
        seed_n = int(ms.group(1))
        path = os.path.join(csv_dir, name)
        found.setdefault(algo, []).append((seed_n, path))
    for algo in found:
        found[algo].sort(key=lambda t: t[0])
    return found


def aggregate_metrics_mean_stderr_over_seeds(
    csv_dir: str, tag_stem: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """
    For each algorithm, stack per-seed CSV matrices and return per-cell mean and standard error
    (sample std / sqrt(n_seeds)).
    """
    grouped = discover_seed_metric_csvs(csv_dir, tag_stem)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for algo, pairs in grouped.items():
        mats = [load_metrics_csv_matrix(p) for _, p in pairs]
        if not mats:
            continue
        stack = np.stack(mats, axis=0)
        n = stack.shape[0]
        mean = np.nanmean(stack, axis=0)
        if n <= 1:
            stderr = np.zeros_like(mean)
        else:
            stderr = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)
        out[algo] = (mean, stderr)
        counts[algo] = n
    return out, counts


def metrics_mean_matrix_from_npz_optional(z: np.lib.npyio.NpzFile, algo: str) -> np.ndarray | None:
    """Mean over test states → ``(n_metrics, n_actions)`` if all metric keys exist (else ``None``)."""
    rows = []
    for mk in _METRIC_KEYS:
        key = f"metric_{mk}_{algo}"
        if key not in z.files:
            return None
        rows.append(np.mean(np.asarray(z[key]), axis=0))
    return np.stack(rows, axis=0)


def discover_seed_npz_paths(npz_dir: str, tag_stem: str) -> list[tuple[int, str]]:
    """Return sorted ``(seed, path)`` for ``purejaxql/whitebox_bandit_run{stem}_seed<N>.npz``."""
    rx = re.compile(rf"^whitebox_bandit_run{re.escape(tag_stem)}_seed(\d+)\.npz$")
    found: list[tuple[int, str]] = []
    if not os.path.isdir(npz_dir):
        return found
    for name in os.listdir(npz_dir):
        m = rx.match(name)
        if not m:
            continue
        found.append((int(m.group(1)), os.path.join(npz_dir, name)))
    found.sort(key=lambda t: t[0])
    return found


def aggregate_metrics_mean_stderr_over_seeds_npz(
    npz_dir: str,
    tag_stem: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, int]]:
    """Aggregate per-seed metrics from NPZ archives (preferred over legacy per-algo CSVs)."""
    paths = discover_seed_npz_paths(npz_dir, tag_stem)
    if not paths:
        return {}, {}
    z0 = np.load(paths[0][1], allow_pickle=False)
    algos = []
    for algo in ALGO_ORDER:
        if f"metric_phi_mse_{algo}" in z0.files:
            algos.append(algo)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for algo in algos:
        mats = []
        for _, p in paths:
            mat = metrics_mean_matrix_from_npz_optional(np.load(p, allow_pickle=False), algo)
            if mat is None:
                mats = []
                break
            mats.append(mat)
        if not mats:
            continue
        stack = np.stack(mats, axis=0)
        n = stack.shape[0]
        mean = np.nanmean(stack, axis=0)
        stderr = np.zeros_like(mean) if n <= 1 else np.nanstd(stack, axis=0, ddof=1) / np.sqrt(n)
        out[algo] = (mean, stderr)
        counts[algo] = n
    return out, counts


def _fmt_mean_se_cell(m: float, s: float) -> str:
    """Readable cell: ``mean (SE)`` with compact SE (avoids ``0.000±0.000`` noise)."""
    if np.isnan(m):
        return "—"
    ms = f"{m:.4g}"
    if np.isnan(s) or s <= 0.0:
        return ms
    ss = f"{s:.3g}"
    return f"{ms}\n({ss})"


def plot_metrics_table_figure_combined_aggregate(
    mean_stderr_by_algo: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
    *,
    n_seeds_by_algo: dict[str, int] | None = None,
    png_dpi: int = 300,
) -> None:
    """Same layout as ``plot_metrics_table_figure_combined``, cells show mean with SE on second line."""
    available_algos = [a for a in ALGO_ORDER if a in mean_stderr_by_algo]
    if not available_algos:
        return

    all_means = np.stack([mean_stderr_by_algo[a][0] for a in available_algos], axis=0)
    best_algo_idx = np.argmin(all_means, axis=0)

    n_algos = len(available_algos)
    fig_h = 2.2 * n_algos + 0.7
    fig, axes = plt.subplots(n_algos, 1, figsize=(8.3, fig_h))
    if n_algos == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.34)

    row_labels = list(_METRIC_ROW_LABELS)
    col_labels = [d.name for d in ACTION_DISTS]

    for i, (ax, algo) in enumerate(zip(axes, available_algos)):
        ax.axis("off")
        mean_m, se_m = mean_stderr_by_algo[algo]
        cells = []
        for r in range(mean_m.shape[0]):
            row_cells = []
            for c in range(mean_m.shape[1]):
                m, s = float(mean_m[r, c]), float(se_m[r, c])
                row_cells.append(_fmt_mean_se_cell(m, s))
            cells.append(row_cells)
        tbl = ax.table(
            cellText=cells, rowLabels=row_labels, colLabels=col_labels,
            loc="center", cellLoc="center", colLoc="center", rowLoc="right",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.08, 1.42)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col < 0:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
            cell.set_linewidth(0.6)
            if row >= 1 and col >= 0 and best_algo_idx[row - 1, col] == i:
                cell.set_facecolor(_WIN_FACECOLOR)
                cell.set_text_props(fontweight="bold", color="#0A0A0A")

        title = ALGO_TITLES.get(algo, algo)
        ns = (n_seeds_by_algo or {}).get(algo)
        sub = f" ({ns} seeds)" if ns else ""
        ax.text(
            0.5, 1.02,
            f"{title}{sub}",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9, color="#2E2E2E", fontweight="bold",
        )

    ns_note = ""
    if n_seeds_by_algo:
        ns_set = set(n_seeds_by_algo.values())
        if len(ns_set) == 1:
            ns_note = f" Across {next(iter(ns_set))} random seeds per algorithm."
        else:
            ns_note = " Seed counts may differ per algorithm if some CSVs are missing."

    fig.suptitle(
        "Per-action distributional & risk metrics (mean ± SE over seeds; within each seed: mean over test states)",
        fontsize=11, y=0.985, color="#2E2E2E",
    )
    short = " / ".join(ALGO_TITLES[a].split(" (")[0] for a in available_algos)
    fig.text(
        0.5, 0.02,
        f"Green cells: best mean across {short} for that (metric, action) pair (lower is better).{ns_note}",
        ha="center", fontsize=7, color="#5A5A5A",
    )
    _save_figure_pdf_png(fig, out_path, png_dpi=png_dpi)
    plt.close(fig)


def save_metrics_csv(per_state_metrics: dict, algo: str, out_path: str) -> None:
    means = _per_action_means(per_state_metrics, algo)
    col_labels = [d.name for d in ACTION_DISTS]
    label_for_csv = [
        "phi_mse",
        "ks",
        "w1",
        "cvar_0p05_err",
        "entropic_risk_err",
        "mean_err",
        "variance_err",
        "mean_variance_risk_err",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", *col_labels])
        for lbl, row in zip(label_for_csv, means):
            w.writerow([lbl, *[_fmt_metric_csv(float(x)) for x in row]])


# ---------------------------------------------------------------------------
# Archive (NPZ + msgpack) and replot
# ---------------------------------------------------------------------------
def _build_panel_data(rng: jax.Array, panel_eval: dict, x_eval: np.ndarray,
                      panel_state_x: np.ndarray, cfg: TrainConfig) -> dict:
    """Re-sample a histogram per action for the panel (deterministic via rng)."""
    sub_keys = jax.random.split(rng, NUM_ACTIONS)
    hist = []
    for k, dist in enumerate(ACTION_DISTS):
        x_tile = jnp.broadcast_to(jnp.asarray(panel_state_x), (PANEL_HIST_SAMPLES, panel_state_x.shape[0]))
        rk_r, rk_z = jax.random.split(sub_keys[k])
        z_next = dist.sample(rk_z, x_tile)
        r_immediate = (
            jax.random.normal(rk_r, (PANEL_HIST_SAMPLES,), dtype=z_next.dtype)
            * jnp.asarray(cfg.immediate_reward_std, dtype=z_next.dtype)
        )
        samples = r_immediate + jnp.asarray(cfg.gamma, dtype=z_next.dtype) * z_next
        hist.append(np.asarray(samples))

    out = {
        "pdf_true": np.asarray(panel_eval["pdf_true"]),
        "phi_true": np.asarray(panel_eval["phi_true"]),
        "phi_iqn": np.asarray(panel_eval["phi_iqn"]),
        "F_true": np.asarray(panel_eval["F_true"]),
        "F_iqn": np.asarray(panel_eval["F_iqn"]),
        "hist_samples": np.stack(hist, axis=0),
    }
    for key in (
        "phi_qr_dqn",
        "phi_c51",
        "phi_cqn",
        "phi_mog_cqn",
        "F_qr_dqn",
        "F_c51",
        "F_cqn",
        "F_mog_cqn",
        "pdf_iqn",
        "pdf_qr_dqn",
        "pdf_c51",
        "pdf_cqn",
        "pdf_mog_cqn",
    ):
        if key in panel_eval:
            out[key] = np.asarray(panel_eval[key])
    return out


def export_artifacts(cfg: TrainConfig, eval_results: dict, t_eval: np.ndarray, x_eval: np.ndarray,
                     panel_state_x: np.ndarray, panel_state_index: int,
                     panel_rng_seed: int) -> dict:
    os.makedirs(cfg.figures_dir, exist_ok=True)

    metric_keys_export = tuple(
        f"{mk}_{algo}"
        for algo in ALGO_ORDER
        for mk in _METRIC_KEYS
    )
    per_state_metrics = {
        k: np.asarray(eval_results[k])
        for k in metric_keys_export
        if k in eval_results
    }

    panel_metric_keys = (
        "pdf_true",
        "phi_true",
        "phi_iqn",
        "phi_qr_dqn",
        "phi_c51",
        "phi_cqn",
        "phi_mog_cqn",
        "F_true",
        "F_iqn",
        "F_qr_dqn",
        "F_c51",
        "F_cqn",
        "F_mog_cqn",
        "pdf_iqn",
        "pdf_qr_dqn",
        "pdf_c51",
        "pdf_cqn",
        "pdf_mog_cqn",
    )
    panel_eval = {
        key: np.asarray(eval_results[key])[panel_state_index]
        for key in panel_metric_keys
        if key in eval_results
    }
    panel_data = _build_panel_data(
        jax.random.PRNGKey(panel_rng_seed),
        panel_eval,
        x_eval,
        panel_state_x,
        cfg,
    )

    tag = cfg.figure_tag

    if not cfg.minimal_artifact_export:
        panels_path = os.path.join(cfg.figures_dir, f"whitebox_bandit_panels{tag}.pdf")
        plot_panels_figure(
            panel_data, t_eval, x_eval, panels_path, cf_omega_half=float(cfg.eval_t_max),
        )
        print(f"Saved {panels_path} (+ .png)")

        for algo in _available_algos(per_state_metrics):
            csv_path = os.path.join(cfg.figures_dir, f"whitebox_bandit_metrics_{algo}{tag}.csv")
            save_metrics_csv(per_state_metrics, algo, csv_path)
            print(f"Saved {csv_path}")
        metrics_pdf_path = os.path.join(cfg.figures_dir, f"whitebox_bandit_metrics_combined{tag}.pdf")
        plot_metrics_table_figure_combined(per_state_metrics, metrics_pdf_path)
        print(f"Saved {metrics_pdf_path} (+ .png)")

    # NPZ archive (everything needed for --replot, no params)
    save_dict = {
        "eval_t_max": np.array(cfg.eval_t_max, dtype=np.float64),
        "t_eval": t_eval,
        "x_eval": x_eval,
        "action_names": np.array([d.name for d in ACTION_DISTS]),
        "action_shorts": np.array([d.short for d in ACTION_DISTS]),
        "plot_x_min": np.array([d.plot_x_min for d in ACTION_DISTS]),
        "plot_x_max": np.array([d.plot_x_max for d in ACTION_DISTS]),
        "plot_t_max": np.array([d.plot_t_max for d in ACTION_DISTS]),
        "panel_state": np.asarray(panel_state_x),
        "panel_state_index": np.array(panel_state_index, dtype=np.int64),
        "panel_rng_seed": np.array(panel_rng_seed, dtype=np.int64),
        "n_test": np.array(cfg.n_test, dtype=np.int64),
        "cvar_alpha": np.array(cfg.cvar_alpha),
        "er_beta": np.array(cfg.er_beta),
        "mv_lambda": np.array(cfg.mv_lambda),
        "risk_x_min": np.array(cfg.risk_x_min),
        "risk_x_max": np.array(cfg.risk_x_max),
        "risk_num_x": np.array(cfg.risk_num_x, dtype=np.int64),
    }
    for k, v in panel_data.items():
        save_dict[f"panel_{k}"] = v
    for k, v in per_state_metrics.items():
        save_dict[f"metric_{k}"] = v
    npz_out = cfg.npz_filename()
    np.savez(npz_out, **save_dict)
    print(f"Saved {npz_out}")
    return per_state_metrics


def save_params_msgpack(
    cfg: TrainConfig,
    params_iqn,
    params_qr_dqn,
    params_c51,
    params_cqn,
    params_mog_cqn,
) -> None:
    if cfg.train_mog_cqn_only:
        payload = {"mog_cqn": params_mog_cqn}
    else:
        payload = {
            "iqn": params_iqn,
            "qr_dqn": params_qr_dqn,
            "c51": params_c51,
            "mog_cqn": params_mog_cqn,
        }
        if cfg.train_cqn and params_cqn is not None:
            payload["cqn"] = params_cqn
    blob = flax.serialization.to_bytes(payload)
    out = cfg.msgpack_filename()
    with open(out, "wb") as f:
        f.write(blob)
    print(f"Saved {out}")


def figure_tag_from_npz_filename(npz_path: str) -> str | None:
    base = os.path.basename(npz_path)
    prefix, suffix = "whitebox_bandit_run", ".npz"
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix): -len(suffix)]
    return None


def replot_from_npz(npz_path: str, cfg: TrainConfig | None = None,
                    *, infer_figure_tag: bool = True) -> None:
    z = np.load(npz_path, allow_pickle=False)
    c = cfg or TrainConfig()
    if infer_figure_tag:
        inferred = figure_tag_from_npz_filename(npz_path)
        if inferred is not None:
            c = replace(c, figure_tag=inferred)
    os.makedirs(c.figures_dir, exist_ok=True)

    panel_data = {
        key.removeprefix("panel_"): np.asarray(z[key])
        for key in z.files
        if key.startswith("panel_") and key not in (
            "panel_state", "panel_state_index", "panel_rng_seed",
        )
    }
    per_state_metrics = {
        key.removeprefix("metric_"): np.asarray(z[key])
        for key in z.files if key.startswith("metric_")
    }
    t_eval = np.asarray(z["t_eval"])
    x_eval = np.asarray(z["x_eval"])

    cf_hw = float(np.asarray(z["eval_t_max"])) if "eval_t_max" in z.files else float(c.eval_t_max)
    panels_path = os.path.join(c.figures_dir, f"whitebox_bandit_panels{c.figure_tag}.pdf")
    plot_panels_figure(panel_data, t_eval, x_eval, panels_path, cf_omega_half=cf_hw)
    print(f"Saved {panels_path} (+ .png)")

    for algo in _available_algos(per_state_metrics):
        csv_path = os.path.join(c.figures_dir, f"whitebox_bandit_metrics_{algo}{c.figure_tag}.csv")
        save_metrics_csv(per_state_metrics, algo, csv_path)
        print(f"Saved {csv_path}")
    metrics_pdf_path = os.path.join(c.figures_dir, f"whitebox_bandit_metrics_combined{c.figure_tag}.pdf")
    plot_metrics_table_figure_combined(per_state_metrics, metrics_pdf_path)
    print(f"Saved {metrics_pdf_path} (+ .png)")
    print(f"Replotted from {npz_path} -> {c.figures_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    cfg: TrainConfig | None = None,
    *,
    wandb_enabled: bool = False,
    wandb_experiment_tag: str | None = None,
) -> None:
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.figures_dir, exist_ok=True)

    exp_tag = wandb_experiment_tag or os.environ.get("WANDB_EXPERIMENT_TAG", "WhiteboxBandit")
    if wandb_enabled:
        wandb_maybe_init(cfg, exp_tag)

    try:
        _main_train_and_export(cfg, exp_tag, wandb_enabled)
    finally:
        if wandb_enabled:
            import wandb

            if wandb.run is not None:
                wandb.finish()


def _main_train_and_export(cfg: TrainConfig, exp_tag: str, wandb_enabled: bool) -> None:
    rng = jax.random.PRNGKey(cfg.seed)
    rng_init, rng_data, rng_test, rng_train = jax.random.split(rng, 4)

    print(f"Building offline dataset: N={cfg.dataset_size:,}, d={cfg.state_dim}, |A|={NUM_ACTIONS}", flush=True)
    t_data = time.perf_counter()
    dataset = build_dataset(jax.random.fold_in(rng_data, cfg.dataset_seed_offset), cfg)
    jax.tree.map(lambda x: x.block_until_ready(), dataset)
    print(f"  dataset built in {time.perf_counter() - t_data:.1f}s", flush=True)

    if cfg.train_mog_cqn_only:
        state_m, mog = create_train_states_mog_only(rng_init, cfg)
        run_chunk = make_train_chunk_mog_only(cfg, mog, dataset)
        state_i = state_q = state_c51 = state_c = None
        iqn = qr_dqn = c51 = cqn = None
    else:
        pack = create_train_states(rng_init, cfg)
        if cfg.train_cqn:
            state_i, state_q, state_c51, state_c, state_m, iqn, qr_dqn, c51, cqn, mog = pack
            assert cqn is not None and state_c is not None
            run_chunk = make_train_chunk(cfg, iqn, qr_dqn, c51, cqn, mog, dataset)
        else:
            state_i, state_q, state_c51, state_c, state_m, iqn, qr_dqn, c51, cqn, mog = pack
            run_chunk = make_train_chunk_no_cqn(cfg, iqn, qr_dqn, c51, mog, dataset)

    n_chunks = cfg.total_steps // cfg.log_every
    print(f"Training: {cfg.total_steps:,} steps in {n_chunks} chunks of {cfg.log_every:,}", flush=True)

    t0 = time.perf_counter()
    rng_local = rng_train
    for c in range(n_chunks):
        t_c = time.perf_counter()
        if cfg.train_mog_cqn_only:
            state_m, rng_local, losses = run_chunk(state_m, rng_local)
            lm = losses
            jax.tree.map(lambda x: x.block_until_ready(), losses)
            dt_c = time.perf_counter() - t_c
            cur_steps = (c + 1) * cfg.log_every
            sps = cur_steps / (time.perf_counter() - t0)
            print(
                f"  step={cur_steps:>6,}  loss_mog={float(lm[-1]):.5f}  "
                f"chunk_dt={dt_c:.1f}s  sps={sps:.1f}",
                flush=True,
            )
        elif cfg.train_cqn:
            state_i, state_q, state_c51, state_c, state_m, rng_local, losses = run_chunk(
                state_i, state_q, state_c51, state_c, state_m, rng_local,
            )
            li, lq, l51, lc, lm = losses
            jax.tree.map(lambda x: x.block_until_ready(), losses)
            dt_c = time.perf_counter() - t_c
            cur_steps = (c + 1) * cfg.log_every
            sps = cur_steps / (time.perf_counter() - t0)
            print(
                f"  step={cur_steps:>6,}  loss_iqn={float(li[-1]):.5f}  loss_qr={float(lq[-1]):.5f}  "
                f"loss_c51={float(l51[-1]):.5f}  loss_cqn={float(lc[-1]):.5f}  "
                f"loss_mog={float(lm[-1]):.5f}  chunk_dt={dt_c:.1f}s  sps={sps:.1f}",
                flush=True,
            )
        else:
            state_i, state_q, state_c51, state_m, rng_local, losses = run_chunk(
                state_i, state_q, state_c51, state_m, rng_local,
            )
            li, lq, l51, lm = losses
            jax.tree.map(lambda x: x.block_until_ready(), losses)
            dt_c = time.perf_counter() - t_c
            cur_steps = (c + 1) * cfg.log_every
            sps = cur_steps / (time.perf_counter() - t0)
            print(
                f"  step={cur_steps:>6,}  loss_iqn={float(li[-1]):.5f}  loss_qr={float(lq[-1]):.5f}  "
                f"loss_c51={float(l51[-1]):.5f}  loss_mog={float(lm[-1]):.5f}  "
                f"chunk_dt={dt_c:.1f}s  sps={sps:.1f}",
                flush=True,
            )
    train_time = time.perf_counter() - t0
    print(f"Training done in {train_time:.1f}s ({cfg.total_steps/train_time:.1f} steps/s)", flush=True)

    # Test set
    rng_test = jax.random.fold_in(rng_test, cfg.test_seed_offset)
    x_test = jax.random.normal(rng_test, (cfg.n_test, cfg.state_dim), dtype=jnp.float32)

    # FIX: Force index 0 to be perfectly identical across ALL seeds for the aggregate plots!
    fixed_rng = jax.random.PRNGKey(9999)
    fixed_panel_state = jax.random.normal(fixed_rng, (cfg.state_dim,), dtype=jnp.float32)
    x_test = x_test.at[0].set(fixed_panel_state)

    panel_seed = panel_state_seed(cfg)
    panel_idx = 0
    panel_state_x = np.asarray(x_test[panel_idx])

    print(f"Evaluating on {cfg.n_test:,} test states (chunk size {cfg.eval_chunk_size}) ...", flush=True)
    t_eval0 = time.perf_counter()
    eval_results, t_eval, x_eval = evaluate_test_set(
        cfg,
        iqn,
        qr_dqn,
        c51,
        cqn,
        mog,
        state_i.params if state_i is not None else None,
        state_q.params if state_q is not None else None,
        state_c51.params if state_c51 is not None else None,
        state_c.params if cfg.train_cqn and state_c is not None else None,
        state_m.params,
        x_test,
    )
    jax.tree.map(lambda x: x.block_until_ready(), eval_results)
    print(f"  eval done in {time.perf_counter() - t_eval0:.1f}s", flush=True)

    per_state_metrics = export_artifacts(
        cfg, eval_results, t_eval, x_eval,
        panel_state_x=panel_state_x, panel_state_index=panel_idx,
        panel_rng_seed=panel_seed,
    )
    save_params_msgpack(
        cfg,
        state_i.params if state_i is not None else None,
        state_q.params if state_q is not None else None,
        state_c51.params if state_c51 is not None else None,
        state_c.params if cfg.train_cqn and state_c is not None else None,
        state_m.params,
    )
    if wandb_enabled:
        wandb_log_whitebox_csv_metrics(per_state_metrics, cfg, exp_tag)


def save_aggregate_combined_csv(
    mean_stderr_by_algo: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
) -> None:
    """Single wide CSV for LaTeX: per algorithm × metric, action columns for mean and SE."""
    shorts = [d.short for d in ACTION_DISTS]
    rows_csv = [
        "phi_mse",
        "ks",
        "w1",
        "cvar_0p05_err",
        "entropic_risk_err",
        "mean_err",
        "variance_err",
        "mean_variance_risk_err",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = ["algorithm", "metric"] + [f"{s}_mean" for s in shorts] + [f"{s}_se" for s in shorts]
        w.writerow(hdr)
        for algo in ALGO_ORDER:
            if algo not in mean_stderr_by_algo:
                continue
            mm, sm = mean_stderr_by_algo[algo]
            for ri, ml in enumerate(rows_csv):
                row = [algo, ml]
                row.extend(float(x) for x in mm[ri])
                row.extend(float(x) for x in sm[ri])
                w.writerow(row)
    print(f"Saved {out_path}")


def plot_panels_figure_aggregate(npz_paths: list[str], out_path: str, *, png_dpi: int = 300) -> None:
    """Four-column panel figure with mean curves and ±1 SE ribbons (real part for CF)."""
    if not npz_paths:
        return
    z0 = np.load(npz_paths[0], allow_pickle=False)
    t_eval = np.asarray(z0["t_eval"])
    x_eval = np.asarray(z0["x_eval"])
    cf_hw = float(np.asarray(z0["eval_t_max"])) if "eval_t_max" in z0.files else float(DEFAULT_OMEGA_MAX)
    panel_keys = [
        k
        for k in z0.files
        if k.startswith("panel_")
        and k
        not in (
            "panel_state",
            "panel_state_index",
            "panel_rng_seed",
        )
    ]
    stacks = {k: [] for k in panel_keys}
    for p in npz_paths:
        z = np.load(p, allow_pickle=False)
        for k in panel_keys:
            stacks[k].append(np.asarray(z[k]))
    mean_p = {k: np.mean(np.stack(stacks[k], axis=0), axis=0) for k in panel_keys}
    se_p = {
        k: (
            np.std(np.stack(stacks[k], axis=0), axis=0, ddof=1) / np.sqrt(len(npz_paths))
            if len(npz_paths) > 1
            else np.zeros_like(mean_p[k])
        )
        for k in panel_keys
    }

    n_rows = NUM_ACTIONS
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.5, 2.45 * n_rows), sharey=False)
    legend_handles, legend_labels = None, None
    hist0 = mean_p.get("panel_hist_samples") if "panel_hist_samples" in mean_p else None

    cf_series = (
        ("phi_true", COLORS["truth"], LW_GT, "Truth", False),
        ("phi_iqn", COLORS["iqn"], LW_IQN, "IQN", True),
        ("phi_qr_dqn", COLORS["qr_dqn"], LW_QR, "QR-DQN", True),
        ("phi_c51", COLORS["c51"], LW_C51, "C51", True),
        ("phi_cqn", COLORS["cqn"], LW_CQN, "CQN", True),
        ("phi_mog_cqn", COLORS["mog_cqn"], LW_MOG, "MoG-CQN", True),
    )
    pdf_series = (
        ("pdf_true", COLORS["truth"], LW_GT, "Truth", False),
        ("pdf_iqn", COLORS["iqn"], LW_IQN, "IQN", True),
        ("pdf_qr_dqn", COLORS["qr_dqn"], LW_QR, "QR-DQN", True),
        ("pdf_c51", COLORS["c51"], LW_C51, "C51", True),
        ("pdf_cqn", COLORS["cqn"], LW_CQN, "CQN", True),
        ("pdf_mog_cqn", COLORS["mog_cqn"], LW_MOG, "MoG-CQN", True),
    )
    cdf_series = (
        ("F_true", COLORS["truth"], LW_GT, "Truth", False),
        ("F_iqn", COLORS["iqn"], LW_IQN, "IQN", True),
        ("F_qr_dqn", COLORS["qr_dqn"], LW_QR, "QR-DQN", True),
        ("F_c51", COLORS["c51"], LW_C51, "C51", True),
        ("F_cqn", COLORS["cqn"], LW_CQN, "CQN", True),
        ("F_mog_cqn", COLORS["mog_cqn"], LW_MOG, "MoG-CQN", True),
    )

    for row, dist in enumerate(ACTION_DISTS):
        ax_den, ax_cf, ax_pdf, ax_cdf = axes[row]
        if hist0 is not None:
            pdf_t = mean_p["panel_pdf_true"][row]
            _draw_density_column(
                ax_den,
                pdf_t,
                hist0[row],
                x_eval,
                dist.plot_x_min,
                dist.plot_x_max,
            )
        if row == 0:
            ax_den.set_title("Density", fontsize=10, color="#2E2E2E")
        ax_den.set_ylabel("density")

        for name, color, lw, lab, use_se in cf_series:
            pk = "panel_" + name
            if pk not in mean_p:
                continue
            y = np.real(mean_p[pk][row])
            y_se = np.real(se_p[pk][row]) if use_se else np.zeros_like(y)
            m = np.abs(t_eval) <= cf_hw
            if use_se:
                ax_cf.fill_between(
                    t_eval[m],
                    (y - y_se)[m],
                    (y + y_se)[m],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            ax_cf.plot(t_eval[m], y[m], color=color, lw=lw, label=lab, alpha=0.95)
        minimal_style(ax_cf)
        ax_cf.set_xlim(-cf_hw, cf_hw)
        if row == 0:
            ax_cf.set_title("CF", fontsize=10, color="#2E2E2E")
        ax_cf.set_ylabel(r"$\mathrm{Re}\,\varphi$")

        xm = (x_eval >= dist.plot_x_min) & (x_eval <= dist.plot_x_max)
        for name, color, lw, lab, use_se in pdf_series:
            pk = "panel_" + name
            if pk not in mean_p:
                continue
            y = mean_p[pk][row]
            y_se = se_p[pk][row] if use_se else np.zeros_like(y)
            if use_se:
                ax_pdf.fill_between(
                    x_eval[xm],
                    (y - y_se)[xm],
                    (y + y_se)[xm],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            ax_pdf.plot(x_eval[xm], y[xm], color=color, lw=lw, label=lab, alpha=0.95)
        minimal_style(ax_pdf)
        ax_pdf.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            ax_pdf.set_title("PDF", fontsize=10, color="#2E2E2E")
        ax_pdf.set_ylabel("density")

        for name, color, lw, lab, use_se in cdf_series:
            pk = "panel_" + name
            if pk not in mean_p:
                continue
            y = mean_p[pk][row]
            y_se = se_p[pk][row] if use_se else np.zeros_like(y)
            if use_se:
                ax_cdf.fill_between(
                    x_eval[xm],
                    (y - y_se)[xm],
                    (y + y_se)[xm],
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            ax_cdf.plot(x_eval[xm], y[xm], color=color, lw=lw, label=lab, alpha=0.95)
        minimal_style(ax_cdf)
        ax_cdf.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            ax_cdf.set_title("CDF", fontsize=10, color="#2E2E2E")
        ax_cdf.set_ylabel(r"$F(x)$")

        ax_den.text(
            -0.34,
            0.5,
            dist.name,
            transform=ax_den.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            color="#2E2E2E",
            fontweight="bold",
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax_cdf.get_legend_handles_labels()

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.035),
            bbox_transform=fig.transFigure,
            ncol=min(8, len(legend_labels)),
            frameon=False,
            fontsize=8,
            labelcolor="#2E2E2E",
        )
    fig.suptitle(
        "Aggregate (mean ± SE over seeds): one fixed test index per run",
        fontsize=11,
        y=0.995,
        color="#2E2E2E",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.09, wspace=0.30, hspace=0.42)
    _save_figure_pdf_png(fig, out_path, png_dpi=png_dpi)
    plt.close(fig)


def run_aggregate_metrics_plot(
    csv_dir: str,
    tag_stem: str,
    *,
    figures_dir: str | None = None,
    output_suffix: str | None = None,
    npz_dir: str | None = None,
) -> None:
    """Prefer NPZ archives under ``npz_dir`` (default ``purejaxql``); fallback to legacy CSVs in ``csv_dir``."""
    npz_dir = npz_dir or os.path.join(os.getcwd(), "purejaxql")
    mean_se, counts = aggregate_metrics_mean_stderr_over_seeds_npz(npz_dir, tag_stem)
    if not mean_se:
        mean_se, counts = aggregate_metrics_mean_stderr_over_seeds(csv_dir, tag_stem)
    if not mean_se:
        raise FileNotFoundError(
            f"No NPZ under {npz_dir!r} (whitebox_bandit_run{tag_stem}_seed*.npz) and no CSVs under {csv_dir!r} "
            f"matching tag stem {tag_stem!r}.",
        )
    out_dir = figures_dir or csv_dir
    os.makedirs(out_dir, exist_ok=True)
    suf = output_suffix if output_suffix is not None else f"{tag_stem}_agg"
    out_base = os.path.join(out_dir, f"whitebox_bandit_metrics_combined{suf}")
    plot_metrics_table_figure_combined_aggregate(mean_se, out_base + ".pdf", n_seeds_by_algo=counts)
    print(f"Saved {out_base}.pdf (+ .png)")
    csv_out = os.path.join(out_dir, f"whitebox_bandit_metrics_combined{suf}.csv")
    save_aggregate_combined_csv(mean_se, csv_out)

    paths = [p for _, p in discover_seed_npz_paths(npz_dir, tag_stem)]
    if len(paths) >= 1:
        pan_base = os.path.join(out_dir, f"whitebox_bandit_panels{suf}")
        plot_panels_figure_aggregate(paths, pan_base + ".pdf")
        print(f"Saved {pan_base}.pdf (+ .png)")


def _cli_train_config(args: argparse.Namespace) -> TrainConfig:
    c = TrainConfig()
    if getattr(args, "seed", None) is not None:
        c = replace(c, seed=int(args.seed))
    if getattr(args, "mog_cqn_only", False):
        c = replace(c, train_mog_cqn_only=True, train_cqn=False)
    elif getattr(args, "no_cqn", False):
        c = replace(c, train_cqn=False)
    if args.figures_dir:
        c = replace(c, figures_dir=args.figures_dir)
    if args.figure_tag is not None:
        c = replace(c, figure_tag=args.figure_tag)
    if getattr(args, "append_seed_to_tag", False):
        c = replace(c, figure_tag=f"{c.figure_tag}_seed{c.seed}")
    if args.total_steps is not None:
        c = replace(c, total_steps=int(args.total_steps))
    if args.dataset_size is not None:
        c = replace(c, dataset_size=int(args.dataset_size))
    if args.n_test is not None:
        c = replace(c, n_test=int(args.n_test))
    if args.num_omega is not None:
        c = replace(c, num_omega=int(args.num_omega))
    if args.omega_max is not None:
        c = replace(c, omega_max=float(args.omega_max))
    if args.omega_clip is not None:
        c = replace(c, omega_clip=float(args.omega_clip))
    if args.omega_laplace_scale is not None:
        c = replace(c, omega_laplace_scale=float(args.omega_laplace_scale))
    if args.eval_t_max is not None:
        c = replace(c, eval_t_max=float(args.eval_t_max))
    if args.gp_t_delta is not None:
        c = replace(c, gp_t_delta=float(args.gp_t_delta))
    if args.gp_t_max is not None:
        c = replace(c, gp_t_max=float(args.gp_t_max))
    if args.gp_num_t is not None:
        c = replace(c, gp_num_t=int(args.gp_num_t))
    if args.num_mog_components is not None:
        c = replace(c, num_mog_components=int(args.num_mog_components))
    if getattr(args, "full_export", False):
        c = replace(c, minimal_artifact_export=False)
    if getattr(args, "num_atoms", None) is not None:
        c = replace(c, num_atoms=int(args.num_atoms))
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box contextual-bandit IQN-vs-CQN benchmark.")
    parser.add_argument("--replot", default=None, metavar="NPZ",
                        help="Load arrays from an np.savez archive and regenerate figures + CSVs (no training).")
    parser.add_argument("--seed", type=int, default=None,
                        help="PRNG seed (TrainConfig.seed). Combine with --append-seed-to-tag for unique outputs.")
    parser.add_argument("--no-cqn", action="store_true",
                        help="Disable CQN: do not train or evaluate it; omit from tables, CSVs, NPZ panel curves, msgpack.")
    parser.add_argument(
        "--mog-cqn-only",
        "--train-only-mog",
        action="store_true",
        dest="mog_cqn_only",
        help="Train/eval/log only MoG-CQN (IQN, QR-DQN, C51, CQN skipped; msgpack contains mog_cqn only).",
    )
    parser.add_argument("--append-seed-to-tag", action="store_true",
                        help="Append _seed{seed} to figure_tag so multiple seeds write separate files.")
    parser.add_argument("--aggregate-metrics-dir", default=None, metavar="DIR",
                        help="Only aggregate: load per-seed metric CSVs from DIR and plot mean±SE (no training).")
    parser.add_argument("--aggregate-tag-stem", default=FIG_FILE_TAG,
                        help="Filename stem before _seed<N>.csv, e.g. _omega_15 (default: same as FIG_FILE_TAG).")
    parser.add_argument("--aggregate-output-suffix", default=None,
                        help="Suffix for combined aggregate figure (default: {tag_stem}_agg).")
    parser.add_argument(
        "--aggregate-npz-dir",
        default=None,
        metavar="DIR",
        help="Directory with per-seed whitebox_bandit_run*.npz (default: ./purejaxql).",
    )
    parser.add_argument(
        "--full-export",
        action="store_true",
        help="During training, also write per-algo CSVs and non-aggregate figures (default: NPZ-only).",
    )
    parser.add_argument(
        "--num-atoms",
        type=int,
        default=None,
        help="Override C51 atom count (default: same as num_tau for fair capacity).",
    )
    parser.add_argument("--wandb", action="store_true",
                        help="Log CSV-equivalent metrics to Weights & Biases (WANDB_PROJECT / WANDB_ENTITY / WANDB_MODE).")
    parser.add_argument("--wandb-experiment-tag", default=None,
                        help="W&B group + tag for filtering runs (default: env WANDB_EXPERIMENT_TAG or WhiteboxBandit).")
    parser.add_argument("--figures-dir", default=None,
                        help="Output directory for figures (default: TrainConfig.figures_dir).")
    parser.add_argument("--figure-tag", default=None,
                        help="Suffix in filenames, e.g. _d5_a4_v2. With --replot, omit to infer from NPZ name.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override TrainConfig.total_steps.")
    parser.add_argument("--dataset-size", type=int, default=None, help="Override TrainConfig.dataset_size.")
    parser.add_argument("--n-test", type=int, default=None, help="Override TrainConfig.n_test.")
    parser.add_argument("--num-omega", type=int, default=None, help="Override TrainConfig.num_omega.")
    parser.add_argument("--omega-max", type=float, default=None, help="Override TrainConfig.omega_max.")
    parser.add_argument("--omega-clip", type=float, default=None, help="Override TrainConfig.omega_clip.")
    parser.add_argument("--omega-laplace-scale", type=float, default=None,
                        help="Override TrainConfig.omega_laplace_scale.")
    parser.add_argument("--eval-t-max", type=float, default=None, help="Override TrainConfig.eval_t_max.")
    parser.add_argument("--gp-t-delta", type=float, default=None, help="Override TrainConfig.gp_t_delta.")
    parser.add_argument("--gp-t-max", type=float, default=None, help="Override TrainConfig.gp_t_max.")
    parser.add_argument("--gp-num-t", type=int, default=None, help="Override TrainConfig.gp_num_t.")
    parser.add_argument("--num-mog-components", type=int, default=None,
                        help="Override TrainConfig.num_mog_components.")
    cli = parser.parse_args()

    if cli.aggregate_metrics_dir:
        run_aggregate_metrics_plot(
            cli.aggregate_metrics_dir,
            cli.aggregate_tag_stem,
            figures_dir=cli.figures_dir,
            output_suffix=cli.aggregate_output_suffix,
            npz_dir=cli.aggregate_npz_dir,
        )
    elif cli.replot:
        cfg = _cli_train_config(cli)
        replot_from_npz(cli.replot, cfg, infer_figure_tag=cli.figure_tag is None)
    else:
        main(
            _cli_train_config(cli),
            wandb_enabled=cli.wandb,
            wandb_experiment_tag=cli.wandb_experiment_tag,
        )
