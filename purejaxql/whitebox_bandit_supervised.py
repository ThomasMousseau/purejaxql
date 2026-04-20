"""
White-box contextual-bandit benchmark: IQN (spatial) vs CQN-family (frequency).

Continuous state ``x ~ N(0, I_d)`` (default ``d=5``); four discrete actions, each with a
state-conditioned reward distribution chosen to stress different geometries
(Gaussian baseline, heavy-tailed Cauchy mixture, sharp/bimodal MoG, skewed Log-Normal).
All networks share an *identical* (separate-weights) trunk over ``(x, a)`` and differ
in their head: IQN predicts spatial quantiles via cosine-tau embedding + Hadamard
merge; CQN predicts complex unit-disk-bounded characteristic-function values via a
learned omega projection + Hadamard merge; MoG-CQN predicts a state-conditioned
mixture-of-Gaussians whose CF is computed analytically.

All are trained on the **same** static offline dataset using a white-box 1-step Bellman
target ``R + gamma * Z_next`` with analytical next-state distributions.

Run::

    python -m purejaxql.whitebox_bandit_supervised

Replot from a saved archive (no retraining)::

    python -m purejaxql.whitebox_bandit_supervised --replot purejaxql/whitebox_bandit_run_TAG.npz

Outputs (in ``figures/``):
  - ``whitebox_bandit_panels{tag}.{pdf,png}`` -- 4x3 faceted comparison for one fixed test state
  - ``whitebox_bandit_metrics_iqn{tag}.csv`` / ``..._cqn{tag}.csv`` / ``..._mog_cqn{tag}.csv`` -- per-algorithm tables
    (rows = metrics, columns = action distributions)
  - ``whitebox_bandit_metrics_combined{tag}.{pdf,png}`` -- one stacked 3-table comparison figure
  - ``purejaxql/whitebox_bandit_run{tag}.npz`` -- replot archive (panel curves + per-state metrics)
  - ``purejaxql/whitebox_bandit_run{tag}.msgpack`` -- final IQN/CQN params (for re-evaluation)

To tweak action-distribution shapes, edit the ``ACTION DISTRIBUTIONS`` section's module-level
constants (each distribution lives in one place: sample / phi / cdf / pdf as pure JAX).
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, replace
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

COLORS = {"truth": "#0A0A0A", "cqn": "#7C3AED", "iqn": "#56B4E9", "mog_cqn": "#E41A1C"}
LW_GT, LW_CQN, LW_IQN, LW_MOG = 2.15, 1.41, 1.05, 1.20


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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIG_FILE_TAG = "_omega_15"

PANEL_STATE_SEED_OFFSET = 9_001
PANEL_HIST_SAMPLES = 50_000

# CQN trains on ω ~ Uniform[-omega_max, omega_max].  The characteristic-function plots and
# phi-MSE metric use ``t ∈ [-eval_t_max, eval_t_max]``.  If ``eval_t_max > omega_max``, φ̂(ω)
# for large |ω| has **never been supervised** → systematic drift vs ground truth when the
# true φ(ω) still varies in that band (heavy tails ⇒ slow φ decay).  Keep ``eval_t_max``
# ≤ ``omega_max`` for in-distribution CF comparisons; widen both together if you need a
# wider frequency window (slightly slower training).
#
# ``gp_t_max`` below is unrelated: it is only the truncation for *Gil–Pelaez integration*
# when mapping φ → F(x); it must be large enough for stable CDF inversion, not tied to ω_max.


DEFAULT_OMEGA_MAX = 15.0 #! 15.0
@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    state_dim: int = STATE_DIM

    # Dataset
    dataset_size: int = 500_000
    dataset_seed_offset: int = 1_001

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
    omega_laplace_scale: float | None = None  # None → omega_clip / 3 (same default as mog_cf.py)
    sigma_min: float = 1e-6
    aux_mean_loss_weight: float = 0.1
    num_mog_components: int = 5

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

    def npz_filename(self) -> str:
        return f"purejaxql/whitebox_bandit_run{self.figure_tag}.npz"

    def msgpack_filename(self) -> str:
        return f"purejaxql/whitebox_bandit_run{self.figure_tag}.msgpack"


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


def create_train_states(rng: jax.Array, cfg: TrainConfig):
    rng_i, rng_c, rng_m = jax.random.split(rng, 3)
    iqn = IQNNet(NUM_ACTIONS, cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth)
    cqn = CQNNet(NUM_ACTIONS, cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth, sigma_min=cfg.sigma_min)
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
    # CQN init uses (num_omega + 1) frequencies: a leading ω=0 column for the aux mean head.
    om0 = jnp.zeros((2, cfg.num_omega + 1), jnp.float32)

    v_iqn = iqn.init(rng_i, x0, a0, tau0)
    v_cqn = cqn.init(rng_c, x0, a0, om0)
    v_mog = mog.init(rng_m, x0, a0)

    state_i = train_state.TrainState.create(
        apply_fn=iqn.apply, params=v_iqn["params"], tx=_make_optimizer(cfg.lr_iqn, cfg.max_grad_norm),
    )
    state_c = train_state.TrainState.create(
        apply_fn=cqn.apply, params=v_cqn["params"], tx=_make_optimizer(cfg.lr_cf, cfg.max_grad_norm),
    )
    state_m = train_state.TrainState.create(
        apply_fn=mog.apply, params=v_mog["params"], tx=_make_optimizer(cfg.lr_cf, cfg.max_grad_norm),
    )
    return state_i, state_c, state_m, iqn, cqn, mog


def make_train_chunk(cfg: TrainConfig, iqn: IQNNet, cqn: CQNNet, mog: MoGCQNNet,
                     dataset: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
    """JIT-compiled scan over ``cfg.log_every`` steps. Returns (states, rng) -> (states, rng, mean_losses)."""
    x_all, a_all, r_all, z_next_all = dataset
    n = x_all.shape[0]
    aux_w = jnp.float32(cfg.aux_mean_loss_weight)
    gamma = jnp.float32(cfg.gamma)

    def step(carry, _):
        state_i, state_c, state_m, rng = carry
        rng, kb, kt, ko = jax.random.split(rng, 4)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        x_b = x_all[idx]
        a_b = a_all[idx]
        r_b = r_all[idx]
        z_next_b = z_next_all[idx]
        a_oh = jax.nn.one_hot(a_b, NUM_ACTIONS, dtype=jnp.float32)

        taus = jax.random.uniform(kt, (cfg.batch_size, cfg.num_tau), dtype=jnp.float32)
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
        lc, gc = jax.value_and_grad(loss_cqn)(state_c.params)
        lm, gm = jax.value_and_grad(loss_mog)(state_m.params)
        state_i = state_i.apply_gradients(grads=gi)
        state_c = state_c.apply_gradients(grads=gc)
        state_m = state_m.apply_gradients(grads=gm)
        return (state_i, state_c, state_m, rng), (li, lc, lm)

    @jax.jit
    def run_chunk(state_i, state_c, state_m, rng):
        (state_i, state_c, state_m, rng), losses = jax.lax.scan(
            step,
            (state_i, state_c, state_m, rng),
            None,
            length=cfg.log_every,
        )
        return state_i, state_c, state_m, rng, losses

    return run_chunk


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def make_eval_one_state(cfg: TrainConfig, iqn: IQNNet, cqn: CQNNet, mog: MoGCQNNet):
    """
    Returns ``eval_one_state(params_iqn, params_cqn, params_mog, x_state)`` -> dict of arrays / scalars,
    stacked over the four actions (leading dim = NUM_ACTIONS).
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

    def per_action(params_iqn, params_cqn, params_mog, x_state, k):
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
        # MoG-CQN: state-conditioned mixture with closed-form CF.
        pi_eval, mu_eval, sigma_eval = mog.apply({"params": params_mog}, x1, a1)
        phi_mog_cqn = compose_mog_cf(t_eval, pi_eval[0], mu_eval[0], sigma_eval[0])
        phi_mog_cqn_gp = compose_mog_cf(t_pos, pi_eval[0], mu_eval[0], sigma_eval[0])
        F_mog_cqn = gil_pelaez_cdf(phi_mog_cqn_gp, t_pos, x_eval)
        F_mog_cqn_risk = gil_pelaez_cdf(phi_mog_cqn_gp, t_pos, x_risk)

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
        # Numeric derivative on the evaluation grid for panel overlays.
        pdf_true = jnp.clip(jnp.gradient(F_true, x_eval), 0.0, None)
        pdf_true_risk = jnp.clip(jnp.gradient(F_true_risk, x_risk), 0.0, None)

        # Distributional metrics (per-state scalars).
        phi_mse_iqn = cf_mse_vs_truth(phi_iqn, phi_true)
        phi_mse_cqn = cf_mse_vs_truth(phi_cqn, phi_true)
        phi_mse_mog_cqn = cf_mse_vs_truth(phi_mog_cqn, phi_true)
        ks_iqn = ks_on_grid(F_iqn, F_true)
        ks_cqn = ks_on_grid(F_cqn, F_true)
        ks_mog_cqn = ks_on_grid(F_mog_cqn, F_true)
        w1_iqn_v = w1_cdf(F_iqn, F_true, x_eval)
        w1_cqn_v = w1_cdf(F_cqn, F_true, x_eval)
        w1_mog_cqn_v = w1_cdf(F_mog_cqn, F_true, x_eval)

        # Risk metrics live on the wide risk grid to avoid truncation masking.
        cvar_true = cvar_from_cdf(F_true_risk, x_risk, cfg.cvar_alpha)
        cvar_iqn = cvar_from_cdf(F_iqn_risk, x_risk, cfg.cvar_alpha)
        cvar_cqn = cvar_from_cdf(F_cqn_risk, x_risk, cfg.cvar_alpha)
        cvar_mog_cqn = cvar_from_cdf(F_mog_cqn_risk, x_risk, cfg.cvar_alpha)
        er_true = entropic_risk_from_cdf(F_true_risk, x_risk, cfg.er_beta)
        er_iqn = entropic_risk_from_cdf(F_iqn_risk, x_risk, cfg.er_beta)
        er_cqn = entropic_risk_from_cdf(F_cqn_risk, x_risk, cfg.er_beta)
        # !MoG-CQN admits an exact closed form for ER_beta from (pi, mu, sigma), no GP inversion.
        # !er_mog_cqn = entropic_risk_mog_cqn(pi_eval[0], mu_eval[0], sigma_eval[0], cfg.er_beta)
        er_mog_cqn = entropic_risk_from_cdf(F_mog_cqn_risk, x_risk, cfg.er_beta)

        # Additional moments / mean-variance risk.
        mean_true = jnp.trapezoid(x_risk * pdf_true_risk, x_risk)
        var_true = jnp.trapezoid(jnp.square(x_risk - mean_true) * pdf_true_risk, x_risk)
        mv_true = mean_true - cfg.mv_lambda * var_true

        mean_iqn = jnp.mean(q_samples)
        var_iqn = jnp.mean(jnp.square(q_samples - mean_iqn))
        mv_iqn = mean_iqn - cfg.mv_lambda * var_iqn

        mean_mog_cqn = jnp.sum(pi_eval[0] * mu_eval[0]).astype(jnp.float64)
        second_mog_cqn = jnp.sum(
            pi_eval[0] * (jnp.square(sigma_eval[0]) + jnp.square(mu_eval[0]))
        ).astype(jnp.float64)
        var_mog_cqn = second_mog_cqn - jnp.square(mean_mog_cqn)
        mv_mog_cqn = mean_mog_cqn - cfg.mv_lambda * var_mog_cqn
        mv_cqn = mean_cqn - cfg.mv_lambda * var_cqn

        return dict(
            phi_iqn=phi_iqn, phi_cqn=phi_cqn, phi_mog_cqn=phi_mog_cqn, phi_true=phi_true,
            F_iqn=F_iqn, F_cqn=F_cqn, F_mog_cqn=F_mog_cqn, F_true=F_true, pdf_true=pdf_true,
            phi_mse_iqn=phi_mse_iqn, phi_mse_cqn=phi_mse_cqn,
            phi_mse_mog_cqn=phi_mse_mog_cqn,
            ks_iqn=ks_iqn, ks_cqn=ks_cqn, ks_mog_cqn=ks_mog_cqn,
            w1_iqn=w1_iqn_v, w1_cqn=w1_cqn_v, w1_mog_cqn=w1_mog_cqn_v,
            cvar_err_iqn=jnp.abs(cvar_iqn - cvar_true),
            cvar_err_cqn=jnp.abs(cvar_cqn - cvar_true),
            cvar_err_mog_cqn=jnp.abs(cvar_mog_cqn - cvar_true),
            er_err_iqn=jnp.abs(er_iqn - er_true),
            er_err_cqn=jnp.abs(er_cqn - er_true),
            er_err_mog_cqn=jnp.abs(er_mog_cqn - er_true),
            mean_err_iqn=jnp.abs(mean_iqn - mean_true),
            mean_err_cqn=jnp.abs(mean_cqn - mean_true),
            mean_err_mog_cqn=jnp.abs(mean_mog_cqn - mean_true),
            var_err_iqn=jnp.abs(var_iqn - var_true),
            var_err_cqn=jnp.abs(var_cqn - var_true),
            var_err_mog_cqn=jnp.abs(var_mog_cqn - var_true),
            mv_err_iqn=jnp.abs(mv_iqn - mv_true),
            mv_err_cqn=jnp.abs(mv_cqn - mv_true),
            mv_err_mog_cqn=jnp.abs(mv_mog_cqn - mv_true),
        )

    @jax.jit
    def eval_one_state(params_iqn, params_cqn, params_mog, x_state):
        # Stack across actions (Python loop because GT callables differ).
        per = [per_action(params_iqn, params_cqn, params_mog, x_state, k) for k in range(NUM_ACTIONS)]
        out = {key: jnp.stack([p[key] for p in per], axis=0) for key in per[0]}
        return out

    return eval_one_state, t_eval, x_eval


def evaluate_test_set(cfg: TrainConfig, iqn: IQNNet, cqn: CQNNet,
                      mog: MoGCQNNet, params_iqn, params_cqn, params_mog, x_test: jax.Array):
    """Vmapped over a chunk of test states; iterated over chunks via lax.map."""
    eval_one, t_eval, x_eval = make_eval_one_state(cfg, iqn, cqn, mog)
    eval_chunk = jax.jit(jax.vmap(eval_one, in_axes=(None, None, None, 0)))

    n = x_test.shape[0]
    cs = cfg.eval_chunk_size
    assert n % cs == 0, f"n_test={n} must be divisible by eval_chunk_size={cs}"
    n_chunks = n // cs
    x_chunks = x_test.reshape(n_chunks, cs, -1)

    chunks_out = []
    for c in range(n_chunks):
        chunks_out.append(eval_chunk(params_iqn, params_cqn, params_mog, x_chunks[c]))
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
def _draw_pdf_panel(ax, pdf_true, hist_samples, x_grid, plot_x_min, plot_x_max,
                    *, hist_label="Samples", pdf_label="Truth PDF"):
    mask = (x_grid >= plot_x_min) & (x_grid <= plot_x_max)
    ax.hist(
        np.asarray(hist_samples),
        bins=72, range=(plot_x_min, plot_x_max), density=True,
        color="#E8E8E8", edgecolor="#CCCCCC", linewidth=0.2,
        label=hist_label, zorder=1,
    )
    ax.plot(
        np.asarray(x_grid)[mask], np.asarray(pdf_true)[mask],
        color=COLORS["truth"], lw=LW_GT, label=pdf_label, zorder=3,
    )
    minimal_style(ax)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_xlabel(r"$x$")


def _draw_phi_panel(ax, t_grid, phi_true, phi_cqn, phi_iqn, phi_mog_cqn, plot_t_max, *,
                    legend_labels=("GT", "CQN", "IQN (empirical)", "MoG-CQN")):
    lg_gt, lg_cqn, lg_iqn, lg_mog = legend_labels
    m = np.abs(t_grid) <= plot_t_max
    ax.plot(t_grid[m], np.real(phi_true)[m], color=COLORS["truth"], lw=LW_GT, label=lg_gt, zorder=1)
    ax.plot(t_grid[m], np.real(phi_cqn)[m], color=COLORS["cqn"], lw=LW_CQN, label=lg_cqn, zorder=2)
    if phi_mog_cqn is not None:
        ax.plot(t_grid[m], np.real(phi_mog_cqn)[m], color=COLORS["mog_cqn"], lw=LW_MOG, alpha=0.95, label=lg_mog, zorder=3)
    ax.plot(t_grid[m], np.real(phi_iqn)[m], color=COLORS["iqn"], lw=LW_IQN, alpha=0.95, label=lg_iqn, zorder=4)
    minimal_style(ax)
    ax.set_xlim(-plot_t_max, plot_t_max)
    ax.set_xlabel(r"$\omega$")


def _draw_cdf_panel(ax, x_grid, F_true, F_cqn, F_iqn, F_mog_cqn, plot_x_min, plot_x_max, *,
                    legend_labels=("GT", "CQN", "IQN (empirical)", "MoG-CQN")):
    lg_gt, lg_cqn, lg_iqn, lg_mog = legend_labels
    m = (x_grid >= plot_x_min) & (x_grid <= plot_x_max)
    ax.plot(x_grid[m], F_true[m], color=COLORS["truth"], lw=LW_GT, label=lg_gt, zorder=1)
    ax.plot(x_grid[m], F_cqn[m], color=COLORS["cqn"], lw=LW_CQN, label=lg_cqn, zorder=2)
    if F_mog_cqn is not None:
        ax.plot(x_grid[m], F_mog_cqn[m], color=COLORS["mog_cqn"], lw=LW_MOG, alpha=0.95, label=lg_mog, zorder=3)
    ax.plot(x_grid[m], F_iqn[m], color=COLORS["iqn"], lw=LW_IQN, alpha=0.95, label=lg_iqn, zorder=4)
    minimal_style(ax)
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_xlabel(r"$x$")


def plot_panels_figure(panel_data: dict, t_eval: np.ndarray, x_eval: np.ndarray,
                       out_path: str, *, png_dpi: int = 300) -> None:
    """4x3 grid: rows = action, columns = (PDF + histogram | Re phi(omega) | F(x))."""
    n_rows = NUM_ACTIONS
    fig, axes = plt.subplots(n_rows, 3, figsize=(10.0, 2.55 * n_rows), sharey=False)
    legend_handles, legend_labels = None, None

    for row, dist in enumerate(ACTION_DISTS):
        ax_pdf, ax_phi, ax_cdf = axes[row]
        pdf_true = panel_data["pdf_true"][row]
        hist = panel_data["hist_samples"][row]
        _draw_pdf_panel(
            ax_pdf, pdf_true, hist, x_eval,
            dist.plot_x_min, dist.plot_x_max,
        )
        if row == 0:
            ax_pdf.set_title("Density", fontsize=10, color="#2E2E2E")
        ax_pdf.set_ylabel("density")

        _draw_phi_panel(
            ax_phi, t_eval,
            panel_data["phi_true"][row], panel_data["phi_cqn"][row], panel_data["phi_iqn"][row],
            panel_data.get("phi_mog_cqn", None)[row] if "phi_mog_cqn" in panel_data else None,
            dist.plot_t_max,
        )
        if row == 0:
            ax_phi.set_title("Characteristic function", fontsize=10, color="#2E2E2E")
        ax_phi.set_ylabel(r"$\mathrm{Re}\,\varphi(\omega)$")

        _draw_cdf_panel(
            ax_cdf, x_eval,
            panel_data["F_true"][row], panel_data["F_cqn"][row], panel_data["F_iqn"][row],
            panel_data.get("F_mog_cqn", None)[row] if "F_mog_cqn" in panel_data else None,
            dist.plot_x_min, dist.plot_x_max,
        )
        if row == 0:
            ax_cdf.set_title("CDF", fontsize=10, color="#2E2E2E")
        ax_cdf.set_ylabel(r"$F(x)$")

        # Row label on the left margin
        ax_pdf.text(
            -0.30, 0.5, dist.name,
            transform=ax_pdf.transAxes,
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
            bbox_to_anchor=(0.5, 0.045),
            bbox_transform=fig.transFigure,
            ncol=len(legend_labels),
            frameon=False, fontsize=9, labelcolor="#2E2E2E",
        )
    fig.suptitle("White-box bandit: per-action distributional fits (one test state)",
                 fontsize=11, y=0.995, color="#2E2E2E")
    fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.07, wspace=0.28, hspace=0.45)
    _save_figure_pdf_png(fig, out_path, png_dpi=png_dpi)
    plt.close(fig)


def _fmt_metric(v: float) -> str:
    return f"{float(v):.3g}"


ALGO_TITLES = {
    "iqn": "IQN (spatial)",
    "cqn": "CQN (frequency)",
    "mog_cqn": "MoG-CQN (frequency)",
}

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
    return [algo for algo in ALGO_TITLES if all(f"{mk}_{algo}" in per_state_metrics for mk in _METRIC_KEYS)]


def _algo_metric_table(per_state_metrics: dict, algo: str) -> tuple[list[str], list[str], list[list[str]]]:
    means = _per_action_means(per_state_metrics, algo)
    cells = [[_fmt_metric(v) for v in row] for row in means]
    return list(_METRIC_ROW_LABELS), [d.name for d in ACTION_DISTS], cells


def plot_metrics_table_figure(per_state_metrics: dict, algo: str, out_path: str) -> None:
    """
    Per-algorithm table with cell-by-cell green highlighting for the cells where this
    algorithm beats the other (lower distributional / risk error is better).
    """
    available_algos = _available_algos(per_state_metrics)
    other_algos = [a for a in available_algos if a != algo]
    row_labels, col_labels, cells = _algo_metric_table(per_state_metrics, algo)
    means_self = _per_action_means(per_state_metrics, algo)
    if other_algos:
        means_others = np.stack([_per_action_means(per_state_metrics, a) for a in other_algos], axis=0)
        win_mask = means_self < np.min(means_others, axis=0)  # (n_metrics, n_actions)
    else:
        win_mask = np.zeros_like(means_self, dtype=bool)

    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.axis("off")
    fig.subplots_adjust(top=0.85, bottom=0.06)
    tbl = ax.table(
        cellText=cells, rowLabels=row_labels, colLabels=col_labels,
        loc="center", cellLoc="center", colLoc="center", rowLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        # Header (row 0) and row labels (col -1) bolded.
        if row == 0 or col < 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
        # Body cells (row >= 1, col >= 0): highlight green if this algo beats the other.
        if row >= 1 and col >= 0 and win_mask[row - 1, col]:
            cell.set_facecolor(_WIN_FACECOLOR)
            cell.set_text_props(fontweight="bold", color="#0A0A0A")
    title = ALGO_TITLES.get(algo, algo)
    fig.suptitle(
        f"{title} — per-action distributional & risk metrics (mean over test states)",
        fontsize=10, y=0.965, color="#2E2E2E",
    )
    fig.text(
        0.5, 0.04,
        f"Green cells: {title.split(' (')[0]} is best across compared algorithms on this (metric, action) pair "
        f"(lower is better).",
        ha="center", fontsize=7, color="#5A5A5A",
    )
    _save_figure_pdf_png(fig, out_path)
    plt.close(fig)


def plot_metrics_table_figure_combined(per_state_metrics: dict, out_path: str) -> None:
    """
    Combined comparison figure with one table per algorithm stacked vertically.
    Best value per (metric, action) across all algorithms is highlighted in green.
    """
    available_algos = _available_algos(per_state_metrics)
    if not available_algos:
        return

    all_means = np.stack([_per_action_means(per_state_metrics, algo) for algo in available_algos], axis=0)
    # (n_algos, n_metrics, n_actions) -> global argmin over algo axis
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
        means = all_means[i]
        cells = [[_fmt_metric(v) for v in row] for row in means]
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
            if row >= 1 and col >= 0 and best_algo_idx[row - 1, col] == i:
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
    fig.text(
        0.5, 0.02,
        "Green cells: best across IQN / CQN / MoG-CQN for that (metric, action) pair (lower is better). "
        "Plots use x in [-15, 15]; risk metrics use the dual-grid x in [-200, 200].",
        ha="center", fontsize=7, color="#5A5A5A",
    )
    _save_figure_pdf_png(fig, out_path)
    plt.close(fig)


def save_metrics_csv(per_state_metrics: dict, algo: str, out_path: str) -> None:
    row_labels, col_labels, cells = _algo_metric_table(per_state_metrics, algo)
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
        for lbl, row in zip(label_for_csv, cells):
            w.writerow([lbl, *row])


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
        "phi_cqn": np.asarray(panel_eval["phi_cqn"]),
        "F_true": np.asarray(panel_eval["F_true"]),
        "F_iqn": np.asarray(panel_eval["F_iqn"]),
        "F_cqn": np.asarray(panel_eval["F_cqn"]),
        "hist_samples": np.stack(hist, axis=0),
    }
    if "phi_mog_cqn" in panel_eval:
        out["phi_mog_cqn"] = np.asarray(panel_eval["phi_mog_cqn"])
    if "F_mog_cqn" in panel_eval:
        out["F_mog_cqn"] = np.asarray(panel_eval["F_mog_cqn"])
    return out


def export_artifacts(cfg: TrainConfig, eval_results: dict, t_eval: np.ndarray, x_eval: np.ndarray,
                     panel_state_x: np.ndarray, panel_state_index: int,
                     panel_rng_seed: int) -> None:
    os.makedirs(cfg.figures_dir, exist_ok=True)

    # Build per-state metric arrays in a flat dict.
    per_state_metrics = {
        "phi_mse_iqn": np.asarray(eval_results["phi_mse_iqn"]),
        "phi_mse_cqn": np.asarray(eval_results["phi_mse_cqn"]),
        "phi_mse_mog_cqn": np.asarray(eval_results["phi_mse_mog_cqn"]),
        "ks_iqn": np.asarray(eval_results["ks_iqn"]),
        "ks_cqn": np.asarray(eval_results["ks_cqn"]),
        "ks_mog_cqn": np.asarray(eval_results["ks_mog_cqn"]),
        "w1_iqn": np.asarray(eval_results["w1_iqn"]),
        "w1_cqn": np.asarray(eval_results["w1_cqn"]),
        "w1_mog_cqn": np.asarray(eval_results["w1_mog_cqn"]),
        "cvar_err_iqn": np.asarray(eval_results["cvar_err_iqn"]),
        "cvar_err_cqn": np.asarray(eval_results["cvar_err_cqn"]),
        "cvar_err_mog_cqn": np.asarray(eval_results["cvar_err_mog_cqn"]),
        "er_err_iqn": np.asarray(eval_results["er_err_iqn"]),
        "er_err_cqn": np.asarray(eval_results["er_err_cqn"]),
        "er_err_mog_cqn": np.asarray(eval_results["er_err_mog_cqn"]),
        "mean_err_iqn": np.asarray(eval_results["mean_err_iqn"]),
        "mean_err_cqn": np.asarray(eval_results["mean_err_cqn"]),
        "mean_err_mog_cqn": np.asarray(eval_results["mean_err_mog_cqn"]),
        "var_err_iqn": np.asarray(eval_results["var_err_iqn"]),
        "var_err_cqn": np.asarray(eval_results["var_err_cqn"]),
        "var_err_mog_cqn": np.asarray(eval_results["var_err_mog_cqn"]),
        "mv_err_iqn": np.asarray(eval_results["mv_err_iqn"]),
        "mv_err_cqn": np.asarray(eval_results["mv_err_cqn"]),
        "mv_err_mog_cqn": np.asarray(eval_results["mv_err_mog_cqn"]),
    }

    # Panel-state slice (one row over actions) from full eval results.
    panel_eval = {
        key: np.asarray(eval_results[key])[panel_state_index]
        for key in (
            "pdf_true",
            "phi_true",
            "phi_iqn",
            "phi_cqn",
            "phi_mog_cqn",
            "F_true",
            "F_iqn",
            "F_cqn",
            "F_mog_cqn",
        )
    }
    panel_data = _build_panel_data(
        jax.random.PRNGKey(panel_rng_seed),
        panel_eval,
        x_eval,
        panel_state_x,
        cfg,
    )

    tag = cfg.figure_tag

    # Figures
    panels_path = os.path.join(cfg.figures_dir, f"whitebox_bandit_panels{tag}.pdf")
    plot_panels_figure(panel_data, t_eval, x_eval, panels_path)
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


def save_params_msgpack(cfg: TrainConfig, params_iqn, params_cqn, params_mog_cqn) -> None:
    blob = flax.serialization.to_bytes({"iqn": params_iqn, "cqn": params_cqn, "mog_cqn": params_mog_cqn})
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

    panels_path = os.path.join(c.figures_dir, f"whitebox_bandit_panels{c.figure_tag}.pdf")
    plot_panels_figure(panel_data, t_eval, x_eval, panels_path)
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
def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.figures_dir, exist_ok=True)

    rng = jax.random.PRNGKey(cfg.seed)
    rng_init, rng_data, rng_test, rng_train = jax.random.split(rng, 4)

    print(f"Building offline dataset: N={cfg.dataset_size:,}, d={cfg.state_dim}, |A|={NUM_ACTIONS}", flush=True)
    t_data = time.perf_counter()
    dataset = build_dataset(jax.random.fold_in(rng_data, cfg.dataset_seed_offset), cfg)
    jax.tree.map(lambda x: x.block_until_ready(), dataset)
    print(f"  dataset built in {time.perf_counter() - t_data:.1f}s", flush=True)

    state_i, state_c, state_m, iqn, cqn, mog = create_train_states(rng_init, cfg)
    run_chunk = make_train_chunk(cfg, iqn, cqn, mog, dataset)

    n_chunks = cfg.total_steps // cfg.log_every
    print(f"Training: {cfg.total_steps:,} steps in {n_chunks} chunks of {cfg.log_every:,}", flush=True)

    t0 = time.perf_counter()
    rng_local = rng_train
    for c in range(n_chunks):
        t_c = time.perf_counter()
        state_i, state_c, state_m, rng_local, (li, lc, lm) = run_chunk(state_i, state_c, state_m, rng_local)
        li.block_until_ready()
        dt_c = time.perf_counter() - t_c
        cur_steps = (c + 1) * cfg.log_every
        sps = cur_steps / (time.perf_counter() - t0)
        print(
            f"  step={cur_steps:>6,}  loss_iqn={float(li[-1]):.5f}  loss_cqn={float(lc[-1]):.5f}  "
            f"loss_mog_cqn={float(lm[-1]):.5f}  "
            f"chunk_dt={dt_c:.1f}s  sps={sps:.1f}",
            flush=True,
        )
    train_time = time.perf_counter() - t0
    print(f"Training done in {train_time:.1f}s ({cfg.total_steps/train_time:.1f} steps/s)", flush=True)

    # Test set
    rng_test = jax.random.fold_in(rng_test, cfg.test_seed_offset)
    x_test = jax.random.normal(rng_test, (cfg.n_test, cfg.state_dim), dtype=jnp.float32)

    # Pick a panel state with non-trivial variability (relatively large |x_1|, |x_2| so that
    # the Cauchy / MoG shapes are visibly non-degenerate). Use a fixed seed for reproducibility.
    panel_seed = panel_state_seed(cfg)
    panel_rng = jax.random.PRNGKey(panel_seed)
    # Index sampling: pick an index in [0, n_test) deterministically.
    panel_idx = int(jax.random.randint(panel_rng, (), 0, cfg.n_test))
    panel_state_x = np.asarray(x_test[panel_idx])

    print(f"Evaluating on {cfg.n_test:,} test states (chunk size {cfg.eval_chunk_size}) ...", flush=True)
    t_eval0 = time.perf_counter()
    eval_results, t_eval, x_eval = evaluate_test_set(
        cfg, iqn, cqn, mog, state_i.params, state_c.params, state_m.params, x_test
    )
    jax.tree.map(lambda x: x.block_until_ready(), eval_results)
    print(f"  eval done in {time.perf_counter() - t_eval0:.1f}s", flush=True)

    export_artifacts(
        cfg, eval_results, t_eval, x_eval,
        panel_state_x=panel_state_x, panel_state_index=panel_idx,
        panel_rng_seed=panel_seed,
    )
    save_params_msgpack(cfg, state_i.params, state_c.params, state_m.params)


def _cli_train_config(args: argparse.Namespace) -> TrainConfig:
    c = TrainConfig()
    if args.figures_dir:
        c = replace(c, figures_dir=args.figures_dir)
    if args.figure_tag is not None:
        c = replace(c, figure_tag=args.figure_tag)
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
    return c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box contextual-bandit IQN-vs-CQN benchmark.")
    parser.add_argument("--replot", default=None, metavar="NPZ",
                        help="Load arrays from an np.savez archive and regenerate figures + CSVs (no training).")
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

    if cli.replot:
        cfg = _cli_train_config(cli)
        replot_from_npz(cli.replot, cfg, infer_figure_tag=cli.figure_tag is None)
    else:
        main(_cli_train_config(cli))
