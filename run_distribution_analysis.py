#!/usr/bin/env python3
"""Offline bandit distribution benchmark: CTD, QTD, and φTD (Dirac / categorical / quantile).

CTD uses categorical cross-entropy (C51-style); QTD uses quantile Huber (QR-DQN-style).
The three φTD heads use the same sampled Bellman target as CTD/QTD (sample-only supervision):
``target = r + gamma * z_sample`` and ``phi_target(omega)=exp(i*omega*target)``. ω sampling and CF
loss scaling are set by ``DAConfig.close_to_theory``: **False** → half-Laplacian ω and CF MSE divided
by ``ω²``; **True** → truncated Pareto (α=1) on ``[omega_min_pareto, omega_clip]`` and **unweighted**
CF MSE (see ``purejaxql.utils.mog_cf.sample_frequencies``). Representation differs only in ``build_*_cf``.
Optional ``aux_mean_loss_weight`` adds a Huber penalty on predicted vs sampled mean return for φTD only;
default **0** keeps the comparison CE (CTD) / quantile Huber (QTD) vs **pure CF** (φTD).
``phi_qt_same_arch_as_qtd`` (default **True**) builds QTD and φTD Quant from the same ``QTDNet`` constructor
(two parameter sets), mirroring CTD vs φ Cat.

Config: ``purejaxql/config/analysis/distribution.yaml`` (overridable via ``--config``).
``num_seeds`` (default 1 in ``DAConfig``; YAML may override) runs independent RNG seeds **sequentially in one process** by default
(one GPU context; avoids multi-process CUDA OOM). Use ``--parallel-seed-workers`` only if you can
afford one JAX process per seed (e.g. multiple GPUs / tuned memory limits). Combined outputs use
95% CIs for the mean across seeds (Student t). Metrics: ``w1``, ``cramer_l2``, ``cf_l2_w``.
Each run writes under ``figures/distribution_analysis/<YYYY-mm-dd_HH:MM:SS>/`` (parseval
plot/CSV, panel/metric figures, payload JSON, checkpoints, resolved experiment YAML).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from plot_colors import algo_color
from paper_plots import (
    configure_matplotlib,
    pdf_path_for_png_stem,
    save_figure_png_and_pdf,
    style_axes_panel,
    style_axes_wandb_curve,
)

from purejaxql.utils.mog_cf import (
    build_categorical_cf,
    build_quantile_cf,
    sample_frequencies,
)

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parent

# Two-sided 95% Student t critical values t_{0.975, df} for df = 1..30 (mean CI: ± t * s / √n).
_T975_DF1_TO_30: tuple[float, ...] = (
    12.70620473617471,
    4.30265272974946,
    3.18244630528371,
    2.7764451051978,
    2.57058183563637,
    2.446911846023004,
    2.364624251592874,
    2.305962926813099,
    2.262157158654392,
    2.228138851986496,
    2.200985160082949,
    2.178812829672518,
    2.160368652931024,
    2.144786687053816,
    2.131449546022662,
    2.119905299285527,
    2.109815577583358,
    2.100922040309607,
    2.093024054408617,
    2.085963447428391,
    2.079613844727773,
    2.073873067878009,
    2.068657610523972,
    2.063898561628096,
    2.059538552753294,
    2.055529438843488,
    2.051830516480699,
    2.048407141795935,
    2.045229642134926,
    2.042272456301236,
)


def _t975_critical(df: int) -> float:
    if df < 1:
        return 0.0
    if df <= len(_T975_DF1_TO_30):
        return float(_T975_DF1_TO_30[df - 1])
    return 1.96


def _mean_ci95_halfwidth(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half-width of 95% CI for the mean (Student t), NaN-aware."""
    n = int(stack.shape[axis])
    if n < 2:
        return np.zeros_like(np.nanmean(stack, axis=axis))
    s = np.nanstd(stack, axis=axis, ddof=1)
    return (_t975_critical(n - 1) * s / np.sqrt(n)).astype(np.float64)


def _artifact_purejaxql_dir() -> Path:
    """Artifact directory next to this script (same layout as package ``purejaxql``)."""
    return _REPO_ROOT / "purejaxql"


def _huber(u: jax.Array, kappa: float) -> jax.Array:
    au = jnp.abs(u)
    return jnp.where(au <= kappa, 0.5 * u**2, kappa * (au - 0.5 * kappa))


def quantile_huber_loss_pairwise(q_preds: jax.Array, rewards: jax.Array, taus: jax.Array, kappa: float) -> jax.Array:
    u = rewards[:, None] - q_preds
    hub = _huber(u, kappa)
    ind = (u < 0).astype(q_preds.dtype)
    return jnp.mean(jnp.abs(taus - ind) * hub / kappa)


def gil_pelaez_cdf(phi_positive: jax.Array, t_pos: jax.Array, x_grid: jax.Array) -> jax.Array:
    phi_positive = phi_positive - 1j * jnp.imag(phi_positive[0])
    x = x_grid[:, None]
    t = t_pos[None, :]
    integrand = jnp.imag(jnp.exp(-1j * t * x) * phi_positive) / t
    return 0.5 - jnp.trapezoid(integrand, t_pos, axis=1) / jnp.pi


def cramer_l2_sq_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid(jnp.square(f_hat - f_true), x_grid)


def cf_l2_sq_over_omega2(phi_hat: jax.Array, phi_t: jax.Array, omega_grid: jax.Array, omega_eps: float) -> jax.Array:
    d = phi_hat - phi_t
    num = jnp.real(d) ** 2 + jnp.imag(d) ** 2
    den = jnp.maximum(jnp.square(jnp.abs(omega_grid)), jnp.float64(omega_eps**2))
    # Parseval-Plancherel normalization for CDF L2: (1 / 2π) ∫ |Δφ(ω)|² / ω² dω.
    return jnp.trapezoid(num / den, omega_grid) / (2.0 * jnp.pi)


def w1_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid(jnp.abs(f_hat - f_true), x_grid)


def empirical_cdf(sorted_samples: jax.Array, x_grid: jax.Array) -> jax.Array:
    n = sorted_samples.shape[0]
    idx = jnp.searchsorted(sorted_samples, x_grid, side="right")
    return idx.astype(jnp.float64) / jnp.maximum(n, 1)


def empirical_cf_from_samples(samples: jax.Array, t_grid: jax.Array) -> jax.Array:
    return jnp.mean(jnp.exp(1j * t_grid[:, None] * samples[None, :]), axis=1)


def entropic_risk_mog(pi: jax.Array, mu: jax.Array, sigma: jax.Array, beta: float) -> jax.Array:
    """Closed-form entropic risk for a Gaussian mixture (used for MoG ground-truth Bellman ER)."""
    dt = jnp.result_type(pi, mu, sigma)
    b = jnp.asarray(beta, dtype=dt)
    mgf = jnp.exp(b * mu + 0.5 * (b**2) * jnp.square(sigma))
    ee = jnp.maximum(jnp.sum(pi * mgf, axis=-1), jnp.finfo(dt).tiny)
    return jnp.where(jnp.abs(b) < 1e-8, jnp.sum(pi * mu, axis=-1), jnp.log(ee) / b)


def project_delta_onto_c51_atoms(y: jax.Array, support: jax.Array) -> jax.Array:
    vmin, vmax = support[0], support[-1]
    n = support.shape[0]
    dz = (vmax - vmin) / jnp.float32(n - 1)
    y_c = jnp.clip(y, vmin, vmax)
    b = (y_c - vmin) / dz
    lo = jnp.floor(b).astype(jnp.int32)
    hi = jnp.ceil(b).astype(jnp.int32)
    lo = jnp.clip(lo, 0, n - 1)
    hi = jnp.clip(hi, 0, n - 1)
    lm = hi.astype(jnp.float32) - b
    um = b - lo.astype(jnp.float32)
    same = lo == hi
    lm, um = jnp.where(same, jnp.ones_like(lm), lm), jnp.where(same, jnp.zeros_like(um), um)
    batch = jnp.arange(y.shape[0], dtype=jnp.int32)
    mat = jnp.zeros((y.shape[0], n), jnp.float32)
    mat = mat.at[batch, lo].add(lm)
    return mat.at[batch, hi].add(um)


def discrete_return_cdf(probs: jax.Array, atoms: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.sum(probs[:, None] * (atoms[:, None] <= x_grid[None, :]).astype(jnp.float64), axis=0)


def discrete_return_cf(probs: jax.Array, atoms: jax.Array, t: jax.Array) -> jax.Array:
    return jnp.sum(probs[:, None] * jnp.exp(1j * atoms[:, None] * t[None, :].astype(jnp.complex128)), axis=0)


def dirac_mixture_cf_batched(pi: jax.Array, mu: jax.Array, om: jax.Array) -> jax.Array:
    """Mixture-of-Diracs CF; pi, mu (B, M), om (B, N) → (B, N) complex."""
    return jnp.sum(pi[..., None, :] * jnp.exp(1j * om[..., None] * mu[..., None, :]), axis=-1)


def phi_dirac_cf_on_grid(pi: jax.Array, mu: jax.Array, t_grid: jax.Array) -> jax.Array:
    """π, μ over components (M,) and frequencies t_grid (T,) → φ(t) complex (T,)."""
    return jnp.sum(pi[:, None] * jnp.exp(1j * t_grid[None, :] * mu[:, None]), axis=0)


def mixture_dirac_cdf(x_grid: jax.Array, pi: jax.Array, mu: jax.Array) -> jax.Array:
    return jnp.sum(pi[:, None] * (mu[:, None] <= x_grid[None, :]).astype(jnp.float64), axis=0)


NUM_ACTIONS = 4
STATE_DIM = 5
GAUSS_SIGMA = 1.0
CAUCHY_PEAK_BASE, CAUCHY_PEAK_SCALE, CAUCHY_GAMMA = 2.0, 2.0, 0.5
MOG_MU, MOG_SIGMA_FLOOR, MOG_SIGMA_SCALE = 3.0, 0.05, 0.4
# Skewed positive returns: Gamma(shape, scale). Scale depends on state[..., 3] (same slot as old log-normal).
GAMMA_SHAPE = 2.0
GAMMA_LOG_THETA_BASE, GAMMA_LOG_THETA_SCALE = -0.15, 0.28


def _cauchy_d(x_state):
    return CAUCHY_PEAK_BASE + CAUCHY_PEAK_SCALE * jnp.abs(x_state[..., 1])


def _mog_sigma(x_state):
    return MOG_SIGMA_FLOOR + MOG_SIGMA_SCALE * jnp.abs(x_state[..., 2])


def _gamma_scale(x_state):
    return jnp.exp(GAMMA_LOG_THETA_BASE + GAMMA_LOG_THETA_SCALE * x_state[..., 3])


def _gauss_sample(rng, x_state):
    mu = x_state[..., 0]
    return mu + GAUSS_SIGMA * jax.random.normal(rng, mu.shape, dtype=x_state.dtype)


def _gauss_phi(t, x_state):
    mu = x_state[0]
    return jnp.exp(1j * t * mu - 0.5 * (GAUSS_SIGMA**2) * t**2)


def _cauchy_sample(rng, x_state):
    rk1, rk2 = jax.random.split(rng)
    pick = jax.random.bernoulli(rk1, 0.5, x_state.shape[:-1])
    u = jax.random.uniform(rk2, x_state.shape[:-1], dtype=x_state.dtype)
    z, d = jnp.tan(jnp.pi * (u - 0.5)), _cauchy_d(x_state)
    return jnp.where(pick, -d + CAUCHY_GAMMA * z, d + CAUCHY_GAMMA * z)


def _cauchy_phi(t, x_state):
    d, at = _cauchy_d(x_state), jnp.abs(t)
    return 0.5 * (jnp.exp(-1j * t * d - CAUCHY_GAMMA * at) + jnp.exp(1j * t * d - CAUCHY_GAMMA * at))


def _mog_sample(rng, x_state):
    rk1, rk2 = jax.random.split(rng)
    pick = jax.random.bernoulli(rk1, 0.5, x_state.shape[:-1])
    eps = jax.random.normal(rk2, x_state.shape[:-1], dtype=x_state.dtype)
    sig = _mog_sigma(x_state)
    return jnp.where(pick, -MOG_MU, MOG_MU) + sig * eps


def _mog_phi(t, x_state):
    sig = _mog_sigma(x_state)
    return (jnp.exp(-0.5 * (sig**2) * t**2) * jnp.cos(MOG_MU * t)).astype(jnp.complex128)


def _gamma_sample(rng, x_state):
    th = _gamma_scale(x_state)
    g = jax.random.gamma(rng, GAMMA_SHAPE, x_state.shape[:-1], dtype=x_state.dtype)
    return th * g


def _gamma_phi(t, x_state):
    th = _gamma_scale(x_state)
    z = (1.0 - 1j * th * t).astype(jnp.complex128)
    return jnp.power(z, -jnp.float64(GAMMA_SHAPE))


def _gauss_cdf(x_grid, x_state):
    mu = x_state[0]
    return 0.5 * (1.0 + jax.scipy.special.erf((x_grid - mu) / (GAUSS_SIGMA * jnp.sqrt(2.0))))


def _cauchy_cdf(x_grid, x_state):
    d = _cauchy_d(x_state)
    return 0.5 * (
        jnp.arctan((x_grid + d) / CAUCHY_GAMMA) / jnp.pi + 0.5 + jnp.arctan((x_grid - d) / CAUCHY_GAMMA) / jnp.pi + 0.5
    )


def _mog_cdf(x_grid, x_state):
    sig = _mog_sigma(x_state)
    s2 = sig * jnp.sqrt(2.0)
    f1 = 0.5 * (1.0 + jax.scipy.special.erf((x_grid + MOG_MU) / s2))
    f2 = 0.5 * (1.0 + jax.scipy.special.erf((x_grid - MOG_MU) / s2))
    return 0.5 * (f1 + f2)


def _gamma_cdf(x_grid, x_state):
    th = _gamma_scale(x_state)
    a = jnp.float64(GAMMA_SHAPE)
    rat = jnp.maximum(x_grid, 0.0) / jnp.maximum(th, jnp.finfo(jnp.float64).tiny)
    return jnp.where(x_grid > 0, jax.scipy.special.gammainc(a, rat), 0.0)


def _gauss_pdf(x_grid, x_state):
    mu = x_state[0]
    return jnp.exp(-0.5 * ((x_grid - mu) / GAUSS_SIGMA) ** 2) / (GAUSS_SIGMA * jnp.sqrt(2.0 * jnp.pi))


def _cauchy_pdf(x_grid, x_state):
    d = _cauchy_d(x_state)
    p1 = 1.0 / (jnp.pi * CAUCHY_GAMMA * (1.0 + ((x_grid + d) / CAUCHY_GAMMA) ** 2))
    p2 = 1.0 / (jnp.pi * CAUCHY_GAMMA * (1.0 + ((x_grid - d) / CAUCHY_GAMMA) ** 2))
    return 0.5 * (p1 + p2)


def _mog_pdf(x_grid, x_state):
    sig = _mog_sigma(x_state)
    p1 = jnp.exp(-0.5 * ((x_grid + MOG_MU) / sig) ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))
    p2 = jnp.exp(-0.5 * ((x_grid - MOG_MU) / sig) ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))
    return 0.5 * (p1 + p2)


def _gamma_pdf(x_grid, x_state):
    th = _gamma_scale(x_state)
    a = jnp.float64(GAMMA_SHAPE)
    x = jnp.maximum(x_grid, jnp.finfo(jnp.float64).tiny)
    logp = (a - 1.0) * jnp.log(x) - x / th - jax.scipy.special.gammaln(a) - a * jnp.log(th)
    return jnp.where(x_grid > 0, jnp.exp(logp), 0.0)


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


ACTION_DISTS = (
    ActionDist("Gaussian (baseline)", "Gauss", _gauss_sample, _gauss_phi, _gauss_cdf, _gauss_pdf, -6.0, 6.0),
    ActionDist("Cauchy mixture (heavy-tailed)", "Cauchy", _cauchy_sample, _cauchy_phi, _cauchy_cdf, _cauchy_pdf, -12.0, 12.0),
    ActionDist("MoG (sharp / bimodal)", "MoG", _mog_sample, _mog_phi, _mog_cdf, _mog_pdf, -6.0, 6.0),
    ActionDist("Gamma (skewed positive)", "Gam", _gamma_sample, _gamma_phi, _gamma_cdf, _gamma_pdf, -0.5, 8.0),
)


def analytic_bellman_moments(
    k: int, x_state: jax.Array, g: jax.Array, sr: jax.Array, lam: jax.Array, beta: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns Mean, Variance, Mean-Variance, and Entropic Risk for the Bellman target."""
    xs = x_state.astype(jnp.float64)
    
    if k == 0: # Gaussian Baseline
        mn, vn = xs[0], jnp.float64(GAUSS_SIGMA**2)
        mz, vz = g * mn, sr**2 + g**2 * vn
        # Exact ER for Gaussian: mu + 0.5 * beta * sigma^2
        er_z = mz + 0.5 * beta * vz
        return mz, vz, mz - lam * vz, er_z
        
    elif k == 1: # Cauchy (All moments undefined)
        n = jnp.float64(jnp.nan)
        return n, n, n, n
        
    elif k == 2: # MoG (Bimodal)
        sm = _mog_sigma(xs)
        vn = sm**2 + jnp.float64(MOG_MU**2)
        mz, vz = g * jnp.float64(0.0), sr**2 + g**2 * vn
        
        # Exact ER for the Bellman MoG target
        pi_z = jnp.array([0.5, 0.5], dtype=jnp.float64)
        mu_z = jnp.array([-g * MOG_MU, g * MOG_MU], dtype=jnp.float64)
        sig_z = jnp.array([jnp.sqrt(sr**2 + g**2 * sm**2)] * 2, dtype=jnp.float64)
        er_z = entropic_risk_mog(pi_z, mu_z, sig_z, beta)
        return mz, vz, mz - lam * vz, er_z
        
    else:  # k == 3 (Gamma next-return)
        th = _gamma_scale(xs)
        a = jnp.float64(GAMMA_SHAPE)
        mn, vn = a * th, a * th**2
        mz, vz = g * mn, sr**2 + g**2 * vn
        b = beta
        bgth = b * g * th
        mgf_ok = bgth < 1.0 - 1e-9
        mgf = jnp.exp(0.5 * sr**2 * b**2) * jnp.power(jnp.maximum(1.0 - bgth, jnp.float64(1e-300)), -a)
        er_z = jnp.where(jnp.abs(b) < 1e-10, mz, jnp.log(mgf) / b)
        er_z = jnp.where(mgf_ok, er_z, jnp.float64(jnp.nan))
        return mz, vz, mz - lam * vz, er_z


DATASET_SEED_OFFSET = 1001
_PANEL_ANCHOR_PRNG_SEED = 42_424_242


def _panel_anchor_prng_key() -> jax.Array:
    """Lazy key — avoids running JAX on GPU at import time (spawn workers import this module)."""
    return jax.random.PRNGKey(_PANEL_ANCHOR_PRNG_SEED)

DA_ORDER = ("ctd", "qtd", "dirac", "phi_cat", "phi_qt")
LEGACY_ALGO_ALIASES = {
    "qtd": "qr_dqn",
    "ctd": "c51",
    "phi_cat": "phitd_fcm",
    "phi_qt": "phitd_fqm",
    "dirac": "phitd_fm",
}
MK = ("w1", "cramer_l2", "cf_l2_w")


@dataclass(frozen=True)
class DAConfig:
    seed: int = 0
    num_seeds: int = 1
    state_dim: int = STATE_DIM
    dataset_size: int = 500_000
    dataset_seed_offset: int = DATASET_SEED_OFFSET
    total_steps: int = 20_000 
    log_every: int = 20_000
    # Gradient steps per JIT chunk; parseval / anchor metrics are logged after each chunk (use < total_steps for curves).
    parseval_every: int = 1_000
    batch_size: int = 1024
    num_tau: int = 51
    num_atoms: int = 51
    num_dirac_components: int = 51
    num_omega: int = 32
    gamma: float = 0.9
    immediate_reward_std: float = 0.2
    omega_max: float = 15.0
    huber_kappa: float = 1.0
    lr: float = 1e-3
    max_grad_norm: float = 10.0
    omega_clip: float = 15.0
    omega_laplace_scale: float = 0.8
    cf_loss_omega_eps: float = 1e-3
    # Optional φTD-only auxiliary Huber on mean return (0 ⇒ pure CF loss vs CE/Huber for CTD/QTD).
    aux_mean_loss_weight: float = 0.0
    c51_v_min: float = -25.0
    c51_v_max: float = 25.0
    hidden_dim: int = 256
    trunk_layers: int = 2
    n_test: int = 10_000
    test_seed_offset: int = 7_777
    eval_chunk_size: int = 50
    eval_num_t: int = 201
    eval_x_min: float = -15.0
    eval_x_max: float = 15.0
    eval_num_x: int = 301
    risk_x_min: float = -200.0
    risk_x_max: float = 200.0
    risk_num_x: int = 2001
    gp_t_delta: float = 1e-4
    gp_num_t: int = 2001
    cvar_alpha: float = 0.05
    er_beta: float = 0.1
    mv_lambda: float = 0.1
    # φTD CF losses only (Dirac / cat / quant): ``False`` → half-Laplacian ω on ``[0, omega_clip]`` and CF MSE
    # scaled by ``1/ω²`` (weighted). ``True`` → truncated Pareto (α=1) on ``[omega_min_pareto, omega_clip]`` and
    # **unweighted** CF MSE (Cramér-type; matches ``phi_td_pqn_minatar`` with Pareto ω and no ``1/ω²``).
    close_to_theory: bool = False
    omega_min_pareto: float = 0.001
    # ``True``: φTD Quant uses the same ``QTDNet(hidden_dim, trunk_layers, num_tau)`` factory as QTD (two separate
    # instances / weights, same architecture as CTD vs φ Cat). ``False``: φ Quant uses ``phi_qt_num_quantiles``.
    phi_qt_same_arch_as_qtd: bool = True
    # Used only when ``phi_qt_same_arch_as_qtd`` is ``False``; if ``None``, defaults to ``num_tau``.
    phi_qt_num_quantiles: int | None = None
    # When set (e.g. figures/distribution_analysis/<timestamp>), checkpoints and local plots go here.
    artifacts_dir: Path | None = None

    def msgpack_filename(self) -> str:
        base = self.artifacts_dir if self.artifacts_dir is not None else _artifact_purejaxql_dir()
        return str(base / f"distribution_analysis_run_seed{self.seed}.msgpack")


def _default_distribution_yaml_path() -> Path:
    return _artifact_purejaxql_dir() / "config" / "analysis" / "distribution.yaml"


def load_da_config_yaml(path: Path | None, overrides: dict) -> DAConfig:
    """Load DAConfig from YAML (OmegaConf), then apply non-None overrides (e.g. CLI)."""
    from omegaconf import OmegaConf

    base = DAConfig()
    known = {f.name for f in fields(DAConfig)} - {"artifacts_dir"}
    merged: dict = {}
    p = path or _default_distribution_yaml_path()
    if p.exists():
        raw = OmegaConf.load(str(p))
        container = OmegaConf.to_container(raw, resolve=True)
        if isinstance(container, dict):
            merged.update(container)
    if "num_mog_components" in merged and "num_dirac_components" not in merged:
        merged["num_dirac_components"] = merged["num_mog_components"]
    for k, v in overrides.items():
        if v is not None and k in known:
            merged[k] = v
    kwargs = {k: merged[k] for k in known if k in merged}
    return replace(base, **kwargs)


def _config_for_logging(cfg: DAConfig) -> dict:
    d = asdict(cfg)
    if d.get("artifacts_dir") is not None:
        d["artifacts_dir"] = str(d["artifacts_dir"])
    return d


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, Path):
        return json.dumps(str(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if np.isnan(val):
            return "nan"
        if np.isinf(val):
            return "inf" if val > 0 else "-inf"
        return repr(val)
    return json.dumps(str(value))


def _yaml_lines(value, indent: int = 0) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = str(k)
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_lines(v, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(v)}")
        return lines
    if isinstance(value, (list, tuple)):
        if not value:
            return [f"{pad}[]"]
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{pad}-")
                lines.extend(_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return lines
    return [f"{pad}{_yaml_scalar(value)}"]


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _yaml_lines(data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_qr_tau_grid(n: int, dtype=jnp.float32):
    idx = jnp.arange(1, n + 1, dtype=dtype)
    return (idx - jnp.float32(0.5)) / jnp.float32(n)


def c51_atoms(cfg: DAConfig):
    return jnp.linspace(jnp.float32(cfg.c51_v_min), jnp.float32(cfg.c51_v_max), cfg.num_atoms, dtype=jnp.float32)


def ctd_atoms(cfg: DAConfig):
    return c51_atoms(cfg)


def _dense_ln_relu(x: jax.Array, hidden_dim: int, name: str) -> jax.Array:
    x = nn.Dense(hidden_dim, name=f"{name}_dense")(x)
    x = nn.LayerNorm(name=f"{name}_ln")(x)
    return nn.relu(x)


class QTDNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_quantiles: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_quantiles, name="qt")(h)


class CTDNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_atoms: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_atoms, name="at")(h)


# Backward-compatibility aliases.
C51Net = CTDNet


class PhiDiracNet(nn.Module):
    """Mixture of Diracs (π, μ) per action — same head layout as MoG without σ."""

    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        pi = jax.nn.softmax(nn.Dense(self.num_components, name="pi")(h), -1)
        mu = nn.Dense(self.num_components, name="mu")(h)
        return pi, mu


def _opt(lr: float, clip: float):
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def _build_qtd_net(cfg: DAConfig, *, num_quantiles: int | None = None) -> QTDNet:
    """Same ``QTDNet`` spec as QTD; ``num_quantiles`` overrides ``cfg.num_tau`` when set."""
    nq = int(cfg.num_tau if num_quantiles is None else num_quantiles)
    return QTDNet(cfg.hidden_dim, cfg.trunk_layers, nq)


def create_train_states(rng: jax.Array, cfg: DAConfig):
    rctd, rqtd, rphid, rphic, rphiq = jax.random.split(rng, 5)
    ctd = CTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    qtd = _build_qtd_net(cfg)
    phi_d = PhiDiracNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_dirac_components)
    phi_cat = CTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    if cfg.phi_qt_same_arch_as_qtd:
        phi_qt = _build_qtd_net(cfg)
    else:
        n_phi = cfg.phi_qt_num_quantiles if cfg.phi_qt_num_quantiles is not None else cfg.num_tau
        phi_qt = _build_qtd_net(cfg, num_quantiles=n_phi)
    x0, a0 = jnp.zeros((2, cfg.state_dim), jnp.float32), jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    tx = _opt(cfg.lr, cfg.max_grad_norm)
    v5, vq, vd, vc, vqt = (
        ctd.init(rctd, x0, a0),
        qtd.init(rqtd, x0, a0),
        phi_d.init(rphid, x0, a0),
        phi_cat.init(rphic, x0, a0),
        phi_qt.init(rphiq, x0, a0),
    )
    return (
        train_state.TrainState.create(apply_fn=ctd.apply, params=v5["params"], tx=tx),
        train_state.TrainState.create(apply_fn=qtd.apply, params=vq["params"], tx=tx),
        train_state.TrainState.create(apply_fn=phi_d.apply, params=vd["params"], tx=tx),
        train_state.TrainState.create(apply_fn=phi_cat.apply, params=vc["params"], tx=tx),
        train_state.TrainState.create(apply_fn=phi_qt.apply, params=vqt["params"], tx=tx),
        ctd,
        qtd,
        phi_d,
        phi_cat,
        phi_qt,
    )


def build_dataset(rng: jax.Array, cfg: DAConfig):
    rx, ra, rr, rz = jax.random.split(rng, 4)
    n, d = cfg.dataset_size, cfg.state_dim
    x = jax.random.normal(rx, (n, d), jnp.float32)
    a = jax.random.randint(ra, (n,), 0, NUM_ACTIONS)
    r = jax.random.normal(rr, (n,), jnp.float32) * jnp.float32(cfg.immediate_reward_std)
    sk = jax.random.split(rz, NUM_ACTIONS)
    zn = jnp.stack([ACTION_DISTS[k].sample(sk[k], x).astype(jnp.float32) for k in range(NUM_ACTIONS)])
    return x, a, r, zn[a, jnp.arange(n)]


def make_train_chunk(
    cfg: DAConfig,
    ctd: CTDNet,
    qtd: QTDNet,
    phi_d: PhiDiracNet,
    phi_cat: CTDNet,
    phi_qt: QTDNet,
    ds,
):
    xa, aa, ra, zn = ds
    n, aux_w = xa.shape[0], jnp.float32(cfg.aux_mean_loss_weight)
    g, atoms = jnp.float32(cfg.gamma), ctd_atoms(cfg)

    def step(c, _):
        sctd, sqtd, sphid, sphic, sphiq, rng = c
        rng, kb, kt, ko = jax.random.split(rng, 4)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        xb, ab, rb, znb = xa[idx], aa[idx], ra[idx], zn[idx]
        oh = jax.nn.one_hot(ab, NUM_ACTIONS, dtype=jnp.float32)
        tau = jax.random.uniform(kt, (cfg.batch_size, cfg.num_tau), jnp.float32)
        tauf = jnp.broadcast_to(fixed_qr_tau_grid(cfg.num_tau)[None, :], (cfg.batch_size, cfg.num_tau))
        if cfg.close_to_theory:
            om = sample_frequencies(
                ko,
                cfg.batch_size * cfg.num_omega,
                float(cfg.omega_clip),
                scale=None,
                distribution="pareto_1",
                omega_min=float(cfg.omega_min_pareto),
            ).reshape(cfg.batch_size, cfg.num_omega)
        else:
            om = sample_frequencies(
                ko,
                cfg.batch_size * cfg.num_omega,
                float(cfg.omega_clip),
                scale=float(cfg.omega_laplace_scale),
                distribution="half_laplacian",
            ).reshape(cfg.batch_size, cfg.num_omega)
        tgt = rb + g * znb

        # Sample-only fair supervision: same information budget as CTD/QTD.
        # Bellman sample target is tgt = r + gamma * z_sample, so CF target is exp(i * omega * tgt).
        target_cf = jnp.exp(1j * om.astype(jnp.complex64) * tgt[:, None].astype(jnp.complex64))
        if cfg.close_to_theory:

            def cf_phi_mean(residual: jnp.ndarray) -> jnp.ndarray:
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2)

        else:
            w2 = jnp.maximum(jnp.square(om), jnp.float32(cfg.cf_loss_omega_eps**2))

            def cf_phi_mean(residual: jnp.ndarray) -> jnp.ndarray:
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2 / w2)

        def lctd(pp):
            lg = jax.nn.log_softmax(ctd.apply({"params": pp}, xb, oh))
            tp = project_delta_onto_c51_atoms(tgt, atoms)
            return -jnp.mean(jnp.sum(tp * lg, -1))

        def lqtd(pp):
            return quantile_huber_loss_pairwise(qtd.apply({"params": pp}, xb, oh), tgt, tauf, cfg.huber_kappa)

        def lphid(pp):
            pi, mu = phi_d.apply({"params": pp}, xb, oh)
            pred = dirac_mixture_cf_batched(pi, mu, om)
            d = pred - target_cf
            mp = jnp.sum(pi * mu, axis=-1)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        def lphic(pp):
            logits = phi_cat.apply({"params": pp}, xb, oh)
            probs = jax.nn.softmax(logits, axis=-1)

            def row_cat(p, o):
                return build_categorical_cf(p, atoms, o)

            pred = jax.vmap(row_cat)(probs, om)
            d = pred - target_cf
            mexp = jnp.sum(probs * atoms, axis=-1)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mexp - tgt, cfg.huber_kappa))

        def lphiq(pp):
            qv = phi_qt.apply({"params": pp}, xb, oh)

            def row_qt(q, o):
                return build_quantile_cf(q, o)

            pred = jax.vmap(row_qt)(qv, om)
            d = pred - target_cf
            mq = jnp.mean(qv, axis=-1)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mq - tgt, cfg.huber_kappa))

        v5, g5 = jax.value_and_grad(lctd)(sctd.params)
        vq, gq = jax.value_and_grad(lqtd)(sqtd.params)
        vd, gd = jax.value_and_grad(lphid)(sphid.params)
        vc, gc = jax.value_and_grad(lphic)(sphic.params)
        vqt, gqt = jax.value_and_grad(lphiq)(sphiq.params)
        return (
            sctd.apply_gradients(grads=g5),
            sqtd.apply_gradients(grads=gq),
            sphid.apply_gradients(grads=gd),
            sphic.apply_gradients(grads=gc),
            sphiq.apply_gradients(grads=gqt),
            rng,
        ), (v5, vq, vd, vc, vqt)

    scan_len = int(cfg.parseval_every)

    @jax.jit
    def run(sctd, sqtd, sphid, sphic, sphiq, rng):
        (sctd, sqtd, sphid, sphic, sphiq, rng), loss = jax.lax.scan(
            step, (sctd, sqtd, sphid, sphic, sphiq, rng), None, scan_len
        )
        return sctd, sqtd, sphid, sphic, sphiq, rng, loss

    return run


def _make_eval(cfg: DAConfig, ctd: CTDNet, qtd: QTDNet, phi_d: PhiDiracNet, phi_cat: CTDNet, phi_qt: QTDNet):
    wm = cfg.omega_max
    te = jnp.linspace(-wm, wm, cfg.eval_num_t, jnp.float64)
    xe = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, jnp.float64)
    tp = jnp.linspace(cfg.gp_t_delta, wm, cfg.gp_num_t, jnp.float64)
    g64, rs64 = jnp.float64(cfg.gamma), jnp.float64(cfg.immediate_reward_std)
    atoms = ctd_atoms(cfg)
    phis = [d.phi for d in ACTION_DISTS]

    def one(p5, pq, pd, pc, pqt, xs, k):
        oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1, a1 = xs[None, :].astype(jnp.float32), oh[None, :]

        lc = ctd.apply({"params": p5}, x1, a1)[0]
        ctd_probs = jax.nn.softmax(lc)
        F5 = discrete_return_cdf(ctd_probs, atoms, xe)
        phi5 = discrete_return_cf(ctd_probs, atoms, te)

        qq = qtd.apply({"params": pq}, x1, a1)[0].astype(jnp.float64)
        Fq = empirical_cdf(jnp.sort(qq), xe)
        phiq = empirical_cf_from_samples(qq.astype(jnp.complex128), te.astype(jnp.complex128))

        piv, mv = phi_d.apply({"params": pd}, x1, a1)
        pmv, mmv = piv[0], mv[0]
        phid = phi_dirac_cf_on_grid(pmv, mmv, te.astype(jnp.complex128))
        Fd = mixture_dirac_cdf(xe, pmv, mmv)

        lc2 = phi_cat.apply({"params": pc}, x1, a1)[0]
        pc2 = jax.nn.softmax(lc2)
        Fpc = discrete_return_cdf(pc2, atoms, xe)
        phipc = discrete_return_cf(pc2, atoms, te)

        qq2 = phi_qt.apply({"params": pqt}, x1, a1)[0].astype(jnp.float64)
        Fq2 = empirical_cdf(jnp.sort(qq2), xe)
        phiq2 = empirical_cf_from_samples(qq2.astype(jnp.complex128), te.astype(jnp.complex128))

        pr = jnp.exp(-0.5 * (rs64**2) * te**2)
        pz = jax.lax.switch(k, phis, g64 * te, xs.astype(jnp.float64))
        pt = pr * pz

        prp = jnp.exp(-0.5 * (rs64**2) * tp**2)
        pzp = jax.lax.switch(k, phis, g64 * tp, xs.astype(jnp.float64))
        ptp = prp * pzp
        Ft = gil_pelaez_cdf(ptp, tp, xe)
        pdf_t = jnp.clip(jnp.gradient(Ft, xe), 0, None)

        pdf5 = jnp.clip(jnp.gradient(F5, xe), 0, None)
        pdfq = jnp.clip(jnp.gradient(Fq, xe), 0, None)
        pdfd = jnp.clip(jnp.gradient(Fd, xe), 0, None)
        pdfpc = jnp.clip(jnp.gradient(Fpc, xe), 0, None)
        pdfq2 = jnp.clip(jnp.gradient(Fq2, xe), 0, None)

        cf_l2_ctd = cf_l2_sq_over_omega2(phi5, pt, te, cfg.gp_t_delta)
        cf_l2_qtd = cf_l2_sq_over_omega2(phiq, pt, te, cfg.gp_t_delta)
        cf_l2_phi_dirac = cf_l2_sq_over_omega2(phid, pt, te, cfg.gp_t_delta)
        cf_l2_phi_cat = cf_l2_sq_over_omega2(phipc, pt, te, cfg.gp_t_delta)
        cf_l2_phi_qt = cf_l2_sq_over_omega2(phiq2, pt, te, cfg.gp_t_delta)

        out = {
            "w1_ctd": w1_cdf(F5, Ft, xe),
            "w1_qtd": w1_cdf(Fq, Ft, xe),
            "w1_dirac": w1_cdf(Fd, Ft, xe),
            "w1_phi_cat": w1_cdf(Fpc, Ft, xe),
            "w1_phi_qt": w1_cdf(Fq2, Ft, xe),
            "cramer_l2_ctd": cramer_l2_sq_cdf(F5, Ft, xe),
            "cramer_l2_qtd": cramer_l2_sq_cdf(Fq, Ft, xe),
            "cramer_l2_dirac": cramer_l2_sq_cdf(Fd, Ft, xe),
            "cramer_l2_phi_cat": cramer_l2_sq_cdf(Fpc, Ft, xe),
            "cramer_l2_phi_qt": cramer_l2_sq_cdf(Fq2, Ft, xe),
            "cf_l2_w_ctd": cf_l2_ctd,
            "cf_l2_w_qtd": cf_l2_qtd,
            "cf_l2_w_dirac": cf_l2_phi_dirac,
            "cf_l2_w_phi_cat": cf_l2_phi_cat,
            "cf_l2_w_phi_qt": cf_l2_phi_qt,
            "phi_ctd": phi5,
            "phi_qtd": phiq,
            "phi_dirac": phid,
            "phi_phi_cat": phipc,
            "phi_phi_qt": phiq2,
            "phi_true": pt,
            "F_ctd": F5,
            "F_qtd": Fq,
            "F_dirac": Fd,
            "F_phi_cat": Fpc,
            "F_phi_qt": Fq2,
            "F_true": Ft,
            "pdf_true": pdf_t,
            "pdf_ctd": pdf5,
            "pdf_qtd": pdfq,
            "pdf_dirac": pdfd,
            "pdf_phi_cat": pdfpc,
            "pdf_phi_qt": pdfq2,
        }

        out.update(
            {
                "phi_qr_dqn": out["phi_qtd"],
                "phi_c51": out["phi_ctd"],
                "F_qr_dqn": out["F_qtd"],
                "F_c51": out["F_ctd"],
                "pdf_qr_dqn": out["pdf_qtd"],
                "pdf_c51": out["pdf_ctd"],
            }
        )
        for m in MK:
            out[f"{m}_qr_dqn"] = out[f"{m}_qtd"]
            out[f"{m}_c51"] = out[f"{m}_ctd"]
            out[f"{m}_phitd_fcm"] = out[f"{m}_phi_cat"]
            out[f"{m}_phitd_fqm"] = out[f"{m}_phi_qt"]
            out[f"{m}_phitd_fm"] = out[f"{m}_dirac"]

        return out

    @jax.jit
    def ev(p5, pq, pd, pc, pqt, xs):
        P = [one(p5, pq, pd, pc, pqt, xs, kk) for kk in range(NUM_ACTIONS)]
        return {k: jnp.stack([p[k] for p in P]) for k in P[0]}

    return ev, te, xe


def evaluate_test_set(cfg: DAConfig, nets, params, xt):
    ev, te, xe = _make_eval(cfg, *nets)
    ech = jax.jit(jax.vmap(ev, (None, None, None, None, None, 0)))
    n, cs = xt.shape[0], cfg.eval_chunk_size
    assert n % cs == 0
    ch = xt.reshape(n // cs, cs, -1)
    parts = [ech(*params, ch[i]) for i in range(ch.shape[0])]
    return {k: jnp.concatenate([p[k] for p in parts]) for k in parts[0]}, np.asarray(te), np.asarray(xe)


def _means(pm: dict, algo: str):
    return np.stack([np.mean(np.asarray(pm[f"{m}_{algo}"]), 0) for m in MK])


def _ok(pm: dict):
    return [a for a in DA_ORDER if all(f"{m}_{a}" in pm for m in MK)]


def _wandb_log_eval_metrics(pm: dict, tag: str, *, step: int) -> None:
    """Log eval metrics to W&B. Per-cell logs skip NaNs; row nanmeans add always-finite scalars when possible."""
    import wandb

    ok = _ok(pm)
    wandb.summary["tag"] = tag
    wandb.summary["experiment_tag"] = tag
    wandb.summary["algos_in_metrics"] = ",".join(ok) if ok else ""
    out: dict = {}
    for a in ok:
        marr = _means(pm, a)
        for mj, m in enumerate(MK):
            row = marr[mj]
            nm = float(np.nanmean(row))
            if np.isfinite(nm):
                out[f"eval/row_nanmean/{a}/{m}"] = nm
        for mj, m in enumerate(MK):
            for i in range(4):
                v = float(marr[mj, i])
                if np.isfinite(v):
                    out[f"eval/cell/{a}/{m}/{ACTION_DISTS[i].short}"] = v
    wandb.summary["eval_scalar_keys"] = len(out)
    if not ok:
        wandb.summary["eval_warning"] = "No algorithm had full metric keys in pm"
    elif not out:
        wandb.summary["eval_warning"] = "All eval values were non-finite (check eval / JAX numerics)"
    wandb.log(out, step=step)


PANEL_EXPORT_KEYS = (
    "pdf_true",
    "phi_true",
    "phi_ctd",
    "phi_qtd",
    "phi_dirac",
    "phi_phi_cat",
    "phi_phi_qt",
    "phi_qr_dqn",
    "phi_c51",
    "F_true",
    "F_ctd",
    "F_qtd",
    "F_dirac",
    "F_phi_cat",
    "F_phi_qt",
    "F_qr_dqn",
    "F_c51",
    "pdf_ctd",
    "pdf_qtd",
    "pdf_dirac",
    "pdf_phi_cat",
    "pdf_phi_qt",
    "pdf_qr_dqn",
    "pdf_c51",
)


def _collect_eval_payload(cfg: DAConfig, tag: str, er: dict, te: np.ndarray, xe: np.ndarray, *, panel_index: int = 0) -> tuple[dict, dict]:
    pm = {f"{m}_{a}": np.asarray(er[f"{m}_{a}"]) for a in DA_ORDER for m in MK if f"{m}_{a}" in er}
    panel_data: dict[str, list] = {}
    for k in PANEL_EXPORT_KEYS:
        if k not in er:
            continue
        arr = np.asarray(er[k])[panel_index]
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        panel_data[f"panel_{k}"] = arr.tolist()
    metric_means = {a: _means(pm, a).tolist() for a in _ok(pm)}
    for new_name, old_name in LEGACY_ALGO_ALIASES.items():
        if new_name in metric_means:
            metric_means[old_name] = metric_means[new_name]
    payload = {
        "seed": int(cfg.seed),
        "experiment_tag": tag,
        "omega_max": float(cfg.omega_max),
        "t_eval": np.asarray(te).tolist(),
        "x_eval": np.asarray(xe).tolist(),
        "metric_means": metric_means,
        "metric_keys": list(MK),
        "action_names": [d.name for d in ACTION_DISTS],
        "action_shorts": [d.short for d in ACTION_DISTS],
        "panel": panel_data,
    }
    return payload, pm


def _save_eval_payload_to_wandb(payload: dict) -> None:
    import wandb

    run = wandb.run
    if run is None:
        return
    payload_name = "distribution_analysis_payload.json"
    payload_path = Path(run.dir) / payload_name
    payload_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    wandb.save(str(payload_path), base_path=str(run.dir), policy="now")
    wandb.summary["distribution_analysis_payload_file"] = payload_name
    wandb.summary["distribution_analysis_payload_seed"] = payload["seed"]


def _save_parseval_training_plot(
    cfg: DAConfig, tag: str, history: dict[str, list[float]], output_dir: Path
) -> dict[str, str] | None:
    """Plot / CSV: Cramér vs CF on the parseval anchor (φTD Dirac / mixture-of-Diracs head)."""
    if not history.get("step"):
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"warning: parseval plot skipped (matplotlib import failed: {exc})", flush=True)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"distribution_analysis_parseval_training_dirac_seed{cfg.seed}"
    csv_path = stem.with_suffix(".csv")

    steps = np.asarray(history["step"], dtype=np.float64)

    def _as_seed_matrix(values: list[float] | list[list[float]]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Expected 1D or 2D parseval history, got shape {arr.shape}.")

    cr_matrix = _as_seed_matrix(history["cramer_l2_dirac"])
    cf_matrix = _as_seed_matrix(history["cf_l2_w_dirac"])
    y_cr = np.clip(np.nan_to_num(np.nanmean(cr_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_cf = np.clip(np.nan_to_num(np.nanmean(cf_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_cr_ci = np.nan_to_num(_mean_ci95_halfwidth(cr_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    y_cf_ci = np.nan_to_num(_mean_ci95_halfwidth(cf_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    cr_color = "#5E2B97"
    cf_color = "#A06CD5"

    configure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(6.1, 3.5))
    ax.fill_between(steps, np.clip(y_cr - y_cr_ci, 0.0, None), y_cr + y_cr_ci, color=cr_color, alpha=0.2, lw=0)
    ax.fill_between(steps, np.clip(y_cf - y_cf_ci, 0.0, None), y_cf + y_cf_ci, color=cf_color, alpha=0.2, lw=0)
    ax.plot(steps, y_cr, color=cr_color, lw=2.0, label=r"Cramer $L_2^2$ (CDF)")
    ax.plot(steps, y_cf, color=cf_color, lw=2.0, ls="--", label=r"CF $L_2^2/\omega^2$")
    pos = np.concatenate([y_cr[np.isfinite(y_cr)], y_cf[np.isfinite(y_cf)]])
    pos = pos[pos > 0]
    if pos.size and float(np.nanmin(pos)) > 0:
        ax.set_yscale("log")
    style_axes_wandb_curve(ax)
    ax.set_title("Parseval anchor")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 1])
    png_out, pdf_out = save_figure_png_and_pdf(fig, stem, dpi_png=300, dpi_pdf=300)
    plt.close(fig)

    fieldnames = (
        "step",
        "cramer_l2_dirac",
        "cf_l2_w_dirac",
        "cramer_l2_dirac_ci95",
        "cf_l2_w_dirac_ci95",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, st in enumerate(steps.tolist()):
            writer.writerow(
                {
                    "step": st,
                    "cramer_l2_dirac": y_cr[i],
                    "cf_l2_w_dirac": y_cf[i],
                    "cramer_l2_dirac_ci95": y_cr_ci[i],
                    "cf_l2_w_dirac_ci95": y_cf_ci[i],
                }
            )

    return {"png": png_out, "pdf": pdf_out, "csv": str(csv_path)}


def save_msgpack(cfg: DAConfig, *params):
    os.makedirs(os.path.dirname(cfg.msgpack_filename()), exist_ok=True)
    ctd_p, qtd_p, dirac_p, phi_cat_p, phi_qt_p = params
    with open(cfg.msgpack_filename(), "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                {
                    "ctd": ctd_p,
                    "qtd": qtd_p,
                    "dirac": dirac_p,
                    "phi_cat": phi_cat_p,
                    "phi_qt": phi_qt_p,
                    "qr_dqn": qtd_p,
                    "c51": ctd_p,
                    "phitd_fm": dirac_p,
                    "phitd_fcm": phi_cat_p,
                    "phitd_fqm": phi_qt_p,
                }
            )
        )


# --- Figures (merged from plot_distribution_analysis.py) ---------------------------------
PLOT_DA_TITLES = {
    "ctd": "CTD",
    "qtd": "QTD",
    "dirac": r"$\varphi$TD Dirac",
    "phi_cat": r"$\varphi$TD Cat",
    "phi_qt": r"$\varphi$TD Quant",
}
PLOT_VISIBLE_ALGOS = ("ctd", "qtd", "dirac", "phi_cat", "phi_qt")
PLOT_ROW_LBL = {
    "w1": r"$W_1$ $\downarrow$",
    "cramer_l2": r"Cramér $\ell_2^2$ $\downarrow$",
    "cf_l2_w": r"CF-$\ell_2^2/\omega^2$ $\downarrow$",
}
PLOT_WIN_BG = "#D2F4D9"
PLOT_COLORS = {
    "truth": "#0A0A0A",
    "qtd": algo_color("qtd"),
    "ctd": algo_color("ctd"),
    "dirac": algo_color("phitd_fm"),
    "phi_cat": algo_color("phitd_fcm"),
    "phi_qt": algo_color("phitd_fqm"),
    "qr_dqn": algo_color("qr_dqn"),
    "c51": algo_color("c51"),
}
PLOT_LW = {
    "truth": 2.15,
    "qtd": 1.08,
    "ctd": 1.12,
    "dirac": 1.14,
    "phi_cat": 1.14,
    "phi_qt": 1.14,
    "qr_dqn": 1.08,
    "c51": 1.12,
}


def _plot_minimal_style(ax) -> None:
    """Match ``paper_plots.style_axes_panel`` / Cauchy / MinAtar-derived panel aesthetic."""
    style_axes_panel(ax)


def _plot_save_fig(fig, path: str, *, png_dpi: int = 300) -> None:
    stem = os.path.splitext(path)[0]
    save_figure_png_and_pdf(fig, stem, dpi_png=png_dpi, dpi_pdf=png_dpi, pad_inches=0.08)


def _plot_fmt_disp(x: float) -> str:
    return "-" if np.isnan(float(x)) else f"{float(x):.3g}"


def _plot_smooth_curve(y: np.ndarray, passes: int = 2) -> np.ndarray:
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


def _plot_row_labels(row_keys: tuple[str, ...]) -> list[str]:
    return [PLOT_ROW_LBL.get(k, k) for k in row_keys]


def _plot_resolve_row_keys(payloads: list[dict]) -> tuple[str, ...]:
    known_order = list(MK)
    seen = set()
    for payload in payloads:
        for key in payload.get("metric_keys", []):
            if isinstance(key, str):
                seen.add(key)
    if not seen:
        return tuple(MK)
    ordered = [k for k in known_order if k in seen]
    extras = [k for k in sorted(seen) if k not in known_order]
    merged = tuple(ordered + extras)
    return merged if merged else tuple(MK)


def _plot_resolve_algo_key(metric_means: dict, algo: str) -> str | None:
    if algo in metric_means:
        return algo
    legacy = LEGACY_ALGO_ALIASES.get(algo)
    if legacy and legacy in metric_means:
        return legacy
    return None


def _plot_resolve_panel_key(data: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in data:
            return key
    return None


def _plot_metric_tables(metric_pack: dict[str, tuple[np.ndarray, np.ndarray]], row_keys: tuple[str, ...], path: str) -> None:
    import matplotlib.pyplot as plt

    algos = [a for a in PLOT_VISIBLE_ALGOS if a in metric_pack]
    if not algos:
        raise ValueError("No metric data found for plotting.")
    means = np.stack([metric_pack[a][0] for a in algos])
    cis = np.stack([metric_pack[a][1] for a in algos])
    inf = np.where(np.isfinite(means), means, np.inf)
    best = np.argmin(inf, 0)
    best = np.where(~np.isfinite(means).any(0), -1, best)

    nfig = len(algos)
    per_algo_height = 2.2 + max(0.0, 0.1 * (len(row_keys) - 8))
    fig, axes = plt.subplots(nfig, 1, figsize=(8.3, per_algo_height * nfig + 0.7))
    if nfig == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.34)
    cols = [d.name for d in ACTION_DISTS]
    row_labels = _plot_row_labels(row_keys)

    for i, (ax, algo) in enumerate(zip(axes, algos)):
        ax.axis("off")
        cells = []
        for r in range(len(row_keys)):
            row = []
            for c in range(len(ACTION_DISTS)):
                mv = means[i, r, c]
                hw = cis[i, r, c]
                if np.isfinite(mv) and np.isfinite(hw) and hw > 0:
                    row.append(f"{mv:.4g}\n±{hw:.3g}")
                else:
                    row.append(_plot_fmt_disp(mv))
            cells.append(row)
        tbl = ax.table(cellText=cells, rowLabels=row_labels, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.08, 1.42)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col < 0:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
            if row >= 1 and col >= 0 and best[row - 1, col] == i and best[row - 1, col] >= 0:
                cell.set_facecolor(PLOT_WIN_BG)
        ax.text(0.5, 1.02, PLOT_DA_TITLES[algo], transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")

    _plot_save_fig(fig, path)
    plt.close(fig)


def _plot_panels(mean_p: dict[str, np.ndarray], ci_p: dict[str, np.ndarray], te: np.ndarray, xe: np.ndarray, cf_hw: float, path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cf_series = (
        (("panel_phi_true",), "truth", "Truth"),
        (("panel_phi_qtd", "panel_phi_qr_dqn"), "qtd", "QTD"),
        (("panel_phi_ctd", "panel_phi_c51"), "ctd", "CTD"),
        (("panel_phi_dirac",), "dirac", r"$\varphi$TD Dir"),
        (("panel_phi_phi_cat",), "phi_cat", r"$\varphi$TD Cat"),
        (("panel_phi_phi_qt",), "phi_qt", r"$\varphi$TD Quant"),
    )
    pdf_series = (
        (("panel_pdf_true",), "truth", "Truth"),
        (("panel_pdf_qtd", "panel_pdf_qr_dqn"), "qtd", "QTD"),
        (("panel_pdf_ctd", "panel_pdf_c51"), "ctd", "CTD"),
        (("panel_pdf_dirac",), "dirac", r"$\varphi$TD Dir"),
        (("panel_pdf_phi_cat",), "phi_cat", r"$\varphi$TD Cat"),
        (("panel_pdf_phi_qt",), "phi_qt", r"$\varphi$TD Quant"),
    )
    cdf_series = (
        (("panel_F_true",), "truth", "Truth"),
        (("panel_F_qtd", "panel_F_qr_dqn"), "qtd", "QTD"),
        (("panel_F_ctd", "panel_F_c51"), "ctd", "CTD"),
        (("panel_F_dirac",), "dirac", r"$\varphi$TD Dir"),
        (("panel_F_phi_cat",), "phi_cat", r"$\varphi$TD Cat"),
        (("panel_F_phi_qt",), "phi_qt", r"$\varphi$TD Quant"),
    )

    fig, axes = plt.subplots(len(ACTION_DISTS), 4, figsize=(13.5, 2.45 * len(ACTION_DISTS)))
    legend = None
    for row, dist in enumerate(ACTION_DISTS):
        axd, axcf, axp, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        if "panel_pdf_true" in mean_p:
            y_truth = np.asarray(mean_p["panel_pdf_true"][row], dtype=np.float64)
            axd.plot(xe[xmask], y_truth[xmask], color=PLOT_COLORS["truth"], lw=PLOT_LW["truth"])
        _plot_minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axd.set_title("Density")

        tmask = np.abs(te) <= cf_hw
        for candidates, ckey, label in cf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = np.real(mean_p[pkey][row])
            ys = np.real(ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row])
            axcf.fill_between(te[tmask], (y - ys)[tmask], (y + ys)[tmask], color=PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axcf.plot(te[tmask], y[tmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axcf)
        axcf.set_xlim(-cf_hw, cf_hw)
        if row == 0:
            axcf.set_title(r"Re $\varphi$")

        for candidates, ckey, label in pdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            if ckey in ("ctd", "phi_cat"):
                # Categorical heads: Dirac-like spikes from CDF jumps.
                cdf_candidates = (
                    ("panel_F_ctd", "panel_F_c51") if ckey == "ctd" else ("panel_F_phi_cat",)
                )
                cdf_key = _plot_resolve_panel_key(mean_p, cdf_candidates)
                if cdf_key is None:
                    continue
                cdf_vals = np.asarray(mean_p[cdf_key][row], dtype=np.float64)
                jumps = np.diff(np.concatenate(([0.0], cdf_vals)))
                jumps = np.clip(jumps, 0.0, None)
                jump_se = np.zeros_like(jumps)
                if cdf_key in ci_p:
                    cdf_se = np.asarray(ci_p[cdf_key][row], dtype=np.float64)
                    jump_se = np.clip(np.diff(np.concatenate(([0.0], cdf_se))), 0.0, None)
                atom_x = xe
                keep = (atom_x >= dist.plot_x_min) & (atom_x <= dist.plot_x_max) & (jumps > 1e-6)
                if np.any(keep):
                    segs = [((x, 0.0), (x, h)) for x, h in zip(atom_x[keep], jumps[keep])]
                    axp.add_collection(LineCollection(segs, colors=PLOT_COLORS[ckey], linewidths=PLOT_LW[ckey], alpha=0.95))
                    if np.any(jump_se[keep] > 0):
                        axp.errorbar(
                            atom_x[keep],
                            jumps[keep],
                            yerr=jump_se[keep],
                            fmt="none",
                            ecolor=PLOT_COLORS[ckey],
                            elinewidth=0.8,
                            alpha=0.35,
                            capsize=0,
                        )
                    axp.plot([], [], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
                continue
            if ckey == "truth":
                y = np.asarray(mean_p[pkey][row], dtype=np.float64)
                ys = np.asarray(ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], dtype=np.float64)
            else:
                y = _plot_smooth_curve(mean_p[pkey][row])
                ys = _plot_smooth_curve(ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], passes=1)
            axp.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axp.plot(xe[xmask], y[xmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axp)
        axp.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axp.set_title("PDF")

        for candidates, ckey, label in cdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][row]
            ys = ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row]
            axc.fill_between(
                xe[xmask],
                (y - ys)[xmask],
                (y + ys)[xmask],
                color=PLOT_COLORS[ckey],
                alpha=0.2,
                lw=0,
                step="post" if ckey in ("ctd", "phi_cat") else None,
            )
            if ckey in ("ctd", "phi_cat"):
                axc.step(xe[xmask], y[xmask], where="post", color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
            else:
                axc.plot(xe[xmask], y[xmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axc.set_title("CDF")

        axd.text(-0.34, 0.5, dist.name, transform=axd.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
        if legend is None:
            legend = axc.get_legend_handles_labels()

    if legend:
        fig.legend(*legend, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.09, wspace=0.3, hspace=0.42)
    _plot_save_fig(fig, path)
    plt.close(fig)


def _plot_density_cdf_only_panels(
    mean_p: dict[str, np.ndarray],
    ci_p: dict[str, np.ndarray],
    xe: np.ndarray,
    path_2x4: str,
    path_4x2: str,
) -> None:
    def _title_case_name(name: str) -> str:
        titled = name.title()
        return titled.replace("Mog", "MoG")

    import matplotlib.pyplot as plt

    cdf_series = (
        (("panel_F_true",), "truth", "Truth"),
        (("panel_F_qtd", "panel_F_qr_dqn"), "qtd", "QTD"),
        (("panel_F_ctd", "panel_F_c51"), "ctd", "CTD"),
        (("panel_F_dirac",), "dirac", r"$\varphi$TD Dir"),
        (("panel_F_phi_cat",), "phi_cat", r"$\varphi$TD Cat"),
        (("panel_F_phi_qt",), "phi_qt", r"$\varphi$TD Quant"),
    )

    # Layout A: 2x4 (top row Density, bottom row CDF)
    fig, axes = plt.subplots(2, len(ACTION_DISTS), figsize=(13.2, 5.2))
    density_truth_key = _plot_resolve_panel_key(mean_p, ("panel_pdf_true",))
    if density_truth_key is None:
        raise ValueError("Missing panel_pdf_true in payload; cannot draw Ground Truth density column.")
    for col, dist in enumerate(ACTION_DISTS):
        axd = axes[0, col]
        axc = axes[1, col]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)

        y_truth = np.asarray(mean_p[density_truth_key][col], dtype=np.float64)
        axd.plot(xe[xmask], y_truth[xmask], color=PLOT_COLORS["truth"], lw=PLOT_LW["truth"], label="Truth")
        _plot_minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        axd.set_title(_title_case_name(dist.name), fontsize=9, fontweight="bold")

        for candidates, ckey, label in cdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][col]
            ys = ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[col]
            axc.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axc.plot(xe[xmask], y[xmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)

    axes[0, 0].set_ylabel("Density", fontweight="bold")
    axes[1, 0].set_ylabel("CDF", fontweight="bold")
    axes[1, 0].legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.10, wspace=0.24, hspace=0.22)
    _plot_save_fig(fig, path_2x4)
    plt.close(fig)

    # Layout B: 4x2 (rows by action, columns Density/CDF)
    fig, axes = plt.subplots(len(ACTION_DISTS), 2, figsize=(7.4, 9.4))
    axes[0, 0].set_title("Density", fontweight="bold")
    axes[0, 1].set_title("CDF", fontweight="bold")
    for row, dist in enumerate(ACTION_DISTS):
        axd, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)

        y_truth = np.asarray(mean_p[density_truth_key][row], dtype=np.float64)
        axd.plot(xe[xmask], y_truth[xmask], color=PLOT_COLORS["truth"], lw=PLOT_LW["truth"], label="Truth")
        _plot_minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)

        for candidates, ckey, label in cdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][row]
            ys = ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row]
            axc.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axc.plot(xe[xmask], y[xmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)

        axd.text(
            -0.34,
            0.5,
            _title_case_name(dist.name),
            transform=axd.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )
    axes[0, 1].legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.07, wspace=0.18, hspace=0.32)
    _plot_save_fig(fig, path_4x2)
    plt.close(fig)


def _plot_aggregate_payloads(
    payloads: list[dict], row_keys: tuple[str, ...]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """Returns per-algorithm (mean, ci_halfwidth), panel means/CIs; CI is 95% for the mean across seeds."""
    if not payloads:
        raise ValueError("No payloads to aggregate.")

    metric_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for algo in DA_ORDER:
        mats = []
        for payload in payloads:
            metric_means = payload.get("metric_means", {})
            algo_key = _plot_resolve_algo_key(metric_means, algo)
            if algo_key is None:
                continue

            arr = np.asarray(metric_means[algo_key], dtype=np.float64)
            payload_metric_keys = payload.get("metric_keys", [])
            if not payload_metric_keys:
                payload_metric_keys = list(MK[: arr.shape[0]])
            payload_metric_idx = {k: i for i, k in enumerate(payload_metric_keys)}

            aligned = np.full((len(row_keys), len(ACTION_DISTS)), np.nan, dtype=np.float64)
            for row_idx, row_key in enumerate(row_keys):
                payload_idx = payload_metric_idx.get(row_key)
                if payload_idx is None or payload_idx >= arr.shape[0]:
                    continue
                row_vals = np.asarray(arr[payload_idx], dtype=np.float64)
                width = min(row_vals.shape[0], len(ACTION_DISTS))
                aligned[row_idx, :width] = row_vals[:width]
            mats.append(aligned)
        if not mats:
            continue
        stack = np.stack(mats, axis=0)
        mean = np.nanmean(stack, axis=0)
        ci = _mean_ci95_halfwidth(stack, axis=0)
        metric_stats[algo] = (mean, ci)

    t_eval = np.asarray(payloads[0]["t_eval"], dtype=np.float64)
    x_eval = np.asarray(payloads[0]["x_eval"], dtype=np.float64)
    omega_max = float(payloads[0]["omega_max"])

    panel_keys = sorted(payloads[0].get("panel", {}).keys())
    panel_mean: dict[str, np.ndarray] = {}
    panel_ci: dict[str, np.ndarray] = {}
    for key in panel_keys:
        vals = []
        for payload in payloads:
            if key in payload.get("panel", {}):
                vals.append(np.asarray(payload["panel"][key], dtype=np.float64))
        if not vals:
            continue
        stack = np.stack(vals, axis=0)
        panel_mean[key] = np.nanmean(stack, axis=0)
        panel_ci[key] = _mean_ci95_halfwidth(stack, axis=0)

    return metric_stats, panel_mean, panel_ci, t_eval, x_eval, omega_max


def _plot_write_metrics_csv(path: str, metric_stats: dict[str, tuple[np.ndarray, np.ndarray]], row_keys: tuple[str, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["algorithm", "metric"] + [f"{x.short}_mean" for x in ACTION_DISTS] + [f"{x.short}_ci95" for x in ACTION_DISTS]
        writer.writerow(header)
        for algo in DA_ORDER:
            if algo not in metric_stats:
                continue
            mm, sm = metric_stats[algo]
            for row_idx, row_key in enumerate(row_keys):
                writer.writerow([algo, row_key, *[float(x) for x in mm[row_idx]], *[float(x) for x in sm[row_idx]]])


def _plot_download_run_payload(run, root_dir: str) -> dict | None:
    try:
        remote_file = run.file("distribution_analysis_payload.json")
        downloaded = remote_file.download(root=root_dir, replace=True)
    except Exception:
        return None

    local_path = Path(getattr(downloaded, "name", Path(root_dir) / "distribution_analysis_payload.json"))
    if not local_path.exists():
        fallback = Path(root_dir) / "distribution_analysis_payload.json"
        if fallback.exists():
            local_path = fallback
        else:
            return None
    with local_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_distribution_analysis_local(payloads: list[dict], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    if not payloads:
        raise ValueError("No payloads to plot.")
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_keys = _plot_resolve_row_keys(payloads)
    metric_stats, panel_mean, panel_ci, t_eval, x_eval, omega_max = _plot_aggregate_payloads(payloads, row_keys)

    metrics_png_base = str(output_dir / "distribution_analysis_metrics_combined.png")
    panels_png_base = str(output_dir / "distribution_analysis_panels.png")
    panels_density_cdf_2x4_base = str(output_dir / "distribution_analysis_panels_density_cdf_2x4.png")
    panels_density_cdf_4x2_base = str(output_dir / "distribution_analysis_panels_density_cdf_4x2.png")
    metrics_csv = str(output_dir / "distribution_analysis_metrics_combined.csv")

    _plot_metric_tables(metric_stats, row_keys, metrics_png_base)
    _plot_panels(panel_mean, panel_ci, t_eval, x_eval, omega_max, panels_png_base)
    _plot_density_cdf_only_panels(
        panel_mean,
        panel_ci,
        x_eval,
        panels_density_cdf_2x4_base,
        panels_density_cdf_4x2_base,
    )
    _plot_write_metrics_csv(metrics_csv, metric_stats, row_keys)

    return {
        "metrics_png": os.path.splitext(metrics_png_base)[0] + ".png",
        "metrics_pdf": str(pdf_path_for_png_stem(Path(metrics_png_base).with_suffix(""))),
        "panels_png": os.path.splitext(panels_png_base)[0] + ".png",
        "panels_pdf": str(pdf_path_for_png_stem(Path(panels_png_base).with_suffix(""))),
        "panels_density_cdf_2x4_png": os.path.splitext(panels_density_cdf_2x4_base)[0] + ".png",
        "panels_density_cdf_2x4_pdf": str(pdf_path_for_png_stem(Path(panels_density_cdf_2x4_base).with_suffix(""))),
        "panels_density_cdf_4x2_png": os.path.splitext(panels_density_cdf_4x2_base)[0] + ".png",
        "panels_density_cdf_4x2_pdf": str(pdf_path_for_png_stem(Path(panels_density_cdf_4x2_base).with_suffix(""))),
        "metrics_csv": metrics_csv,
    }


def plot_distribution_analysis_from_wandb(
    experiment_tag: str, *, project: str, entity: str | None, figures_dir: str
) -> dict[str, str]:
    """Aggregate finished W&B runs by tag and write figures under figures_dir/<timestamp>."""
    import matplotlib.pyplot as plt
    import wandb

    configure_matplotlib()

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
            payload = _plot_download_run_payload(run, tmp_dir)
            if payload is not None and payload.get("experiment_tag") == experiment_tag:
                payloads.append(payload)

    if not payloads:
        raise FileNotFoundError(
            f"No runs found with payload file 'distribution_analysis_payload.json' for experiment tag '{experiment_tag}'."
        )

    now = dt.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    output_dir = Path(figures_dir) / timestamp_str
    out_paths = plot_distribution_analysis_local(payloads, output_dir)
    meta = {
        "generated_at_local": now.isoformat(),
        "experiment_tag": experiment_tag,
        "project": project,
        "entity": final_entity,
        "num_runs": len(payloads),
        "figures_root": figures_dir,
        "output_dir": str(output_dir),
        "files": out_paths,
    }
    _write_yaml(output_dir / "distribution_analysis_wandb_aggregate.yaml", meta)
    return {"num_runs": str(len(payloads)), "output_dir": str(output_dir), **out_paths}


def _merge_parseval_histories(histories: list[dict[str, list[float]]]) -> dict[str, list]:
    """Stack per-seed parseval series for CI shading (list-of-series per metric)."""
    if not histories:
        return {}
    out = {"step": list(histories[0]["step"]), "cramer_l2_dirac": [], "cf_l2_w_dirac": []}
    for h in histories:
        out["cramer_l2_dirac"].append(list(h["cramer_l2_dirac"]))
        out["cf_l2_w_dirac"].append(list(h["cf_l2_w_dirac"]))
    return out


def train_and_export(
    cfg: DAConfig,
    tag: str,
    wb: bool,
    *,
    figures_run_dir: Path,
    skip_plots: bool = False,
    skip_local_aggregate_plots: bool = False,
    wandb_managed_externally: bool = False,
) -> tuple[dict, dict[str, list[float]], dict]:
    """Train one seed; returns (payload dict, parseval_history, pm)."""
    local_wb = wb and not wandb_managed_externally
    if local_wb:
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "Deep-CVI-Experiments"),
            entity=os.environ.get("WANDB_ENTITY"),
            group=tag,
            name=f"{tag}_seed{cfg.seed}",
            tags=[tag],
            config=_config_for_logging(cfg),
        )
    try:
        figures_run_dir.mkdir(parents=True, exist_ok=True)
        cfg_yaml_name = (
            "distribution_analysis_experiment_config.yaml"
            if int(cfg.num_seeds) <= 1
            else f"distribution_analysis_experiment_config_seed{cfg.seed}.yaml"
        )
        _write_yaml(figures_run_dir / cfg_yaml_name, _config_for_logging(cfg))

        rng = jax.random.PRNGKey(cfg.seed)
        ri, rd, rt, ru = jax.random.split(rng, 4)
        print(f"seed={cfg.seed}  Dataset... offset={cfg.dataset_seed_offset}", flush=True)
        if cfg.close_to_theory:
            print(
                f"φTD CF: close_to_theory=True (Pareto ω, ω_min={cfg.omega_min_pareto}, unweighted CF MSE)",
                flush=True,
            )
        else:
            print(
                "φTD CF: close_to_theory=False (half-Laplacian ω, CF MSE / ω²)",
                flush=True,
            )
        ds = build_dataset(jax.random.fold_in(rd, cfg.dataset_seed_offset), cfg)
        jax.tree.map(lambda x: x.block_until_ready(), ds)
        sctd, sqtd, sphid, sphic, sphiq, *nets = create_train_states(ri, cfg)
        parseval_eval, _, _ = _make_eval(cfg, *nets)
        anchor_x = jax.random.normal(_panel_anchor_prng_key(), (cfg.state_dim,), jnp.float32)
        parseval_history: dict[str, list[float]] = {"step": [], "cramer_l2_dirac": [], "cf_l2_w_dirac": []}
        run = make_train_chunk(cfg, *nets, ds)
        pe = max(1, min(int(cfg.parseval_every), int(cfg.total_steps)))
        if int(cfg.total_steps) % pe != 0:
            raise ValueError(
                f"total_steps ({cfg.total_steps}) must be divisible by parseval_every ({pe}) "
                "so training chunks and parseval logs align; adjust parseval_every in YAML or CLI."
            )
        nc = int(cfg.total_steps) // pe
        t0 = time.perf_counter()
        rng = ru
        last_losses: dict[str, float] | None = None
        for c in range(nc):
            tc = time.perf_counter()
            sctd, sqtd, sphid, sphic, sphiq, rng, ls = run(sctd, sqtd, sphid, sphic, sphiq, rng)
            jax.tree.map(lambda x: x.block_until_ready(), ls)
            st = (c + 1) * pe
            last_losses = {
                "train/loss_ctd_final": float(ls[0][-1]),
                "train/loss_qtd_final": float(ls[1][-1]),
                "train/loss_dirac_final": float(ls[2][-1]),
                "train/loss_phi_cat_final": float(ls[3][-1]),
                "train/loss_phi_qt_final": float(ls[4][-1]),
            }
            last_losses["train/loss_qr_dqn_final"] = last_losses["train/loss_qtd_final"]
            last_losses["train/loss_c51_final"] = last_losses["train/loss_ctd_final"]
            if st % int(cfg.log_every) == 0 or st == int(cfg.total_steps) or c == 0:
                print(
                    f"  {st:>7}  ctd={last_losses['train/loss_ctd_final']:.4f} "
                    f"qtd={last_losses['train/loss_qtd_final']:.4f} "
                    f"dirac={last_losses['train/loss_dirac_final']:.4f} "
                    f"phi_cat={last_losses['train/loss_phi_cat_final']:.4f} "
                    f"phi_qt={last_losses['train/loss_phi_qt_final']:.4f}  {time.perf_counter()-tc:.1f}s",
                    flush=True,
                )

            pv = parseval_eval(
                sctd.params,
                sqtd.params,
                sphid.params,
                sphic.params,
                sphiq.params,
                anchor_x,
            )
            jax.tree.map(lambda x: x.block_until_ready(), pv)
            parseval_history["step"].append(float(st))
            cr_m = float(jnp.mean(pv["cramer_l2_dirac"]))
            cf_m = float(jnp.mean(pv["cf_l2_w_dirac"]))
            parseval_history["cramer_l2_dirac"].append(cr_m)
            parseval_history["cf_l2_w_dirac"].append(cf_m)
            pv_log = {
                "train/parseval/dirac/cramer_l2_anchor": cr_m,
                "train/parseval/dirac/cf_l2_w_anchor": cf_m,
            }
            if wb:
                import wandb

                wandb.log(pv_log, step=st)

        print(f"train seed={cfg.seed} {time.perf_counter()-t0:.1f}s", flush=True)
        parseval_plot_paths = None
        if not skip_plots and not skip_local_aggregate_plots:
            parseval_plot_paths = _save_parseval_training_plot(cfg, tag, parseval_history, figures_run_dir)
        xt = jax.random.fold_in(rt, cfg.test_seed_offset)
        xv = jax.random.normal(xt, (cfg.n_test, cfg.state_dim), jnp.float32).at[0].set(
            jax.random.normal(_panel_anchor_prng_key(), (cfg.state_dim,), jnp.float32)
        )
        er, te, xe = evaluate_test_set(
            cfg,
            tuple(nets),
            (
                sctd.params,
                sqtd.params,
                sphid.params,
                sphic.params,
                sphiq.params,
            ),
            xv,
        )
        jax.tree.map(lambda x: x.block_until_ready(), er)
        payload, pm = _collect_eval_payload(cfg, tag, er, te, xe, panel_index=0)
        save_msgpack(cfg, sctd.params, sqtd.params, sphid.params, sphic.params, sphiq.params)
        payload_name = (
            "distribution_analysis_payload.json"
            if int(cfg.num_seeds) <= 1
            else f"distribution_analysis_payload_seed{cfg.seed}.json"
        )
        payload_path = figures_run_dir / payload_name
        payload_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
        if not skip_plots and not skip_local_aggregate_plots:
            try:
                plot_paths = plot_distribution_analysis_local([payload], figures_run_dir)
                print(f"Saved figures: {plot_paths}", flush=True)
            except Exception as exc:
                print(f"warning: post-run plotting failed: {exc}", flush=True)
        if wb:
            import wandb

            if last_losses:
                wandb.summary.update(last_losses)
            if parseval_plot_paths:
                for key, p in parseval_plot_paths.items():
                    if key == "csv":
                        continue
                    wandb.save(p, policy="now")
                wandb.summary["parseval_train_plot_png"] = parseval_plot_paths["png"]
                wandb.summary["parseval_train_plot_pdf"] = parseval_plot_paths.get("pdf", "")
                wandb.summary["parseval_train_metrics_csv"] = parseval_plot_paths["csv"]
            _wandb_log_eval_metrics(pm, tag, step=cfg.total_steps)
            _save_eval_payload_to_wandb(payload)
            wandb.summary["distribution_analysis_artifacts_dir"] = str(figures_run_dir)

        return payload, parseval_history, pm
    finally:
        if local_wb:
            import wandb

            if wandb.run is not None:
                wandb.finish()


def _distribution_analysis_seed_worker(args: tuple[bytes, int, str, str, bool, bool]) -> tuple[dict, dict[str, list[float]], dict]:
    """Picklable entry point for ProcessPoolExecutor (spawn)."""
    cfg_blob, seed, run_dir_s, tag, wb, skip_plots = args
    base: DAConfig = pickle.loads(cfg_blob)
    cfg = replace(base, seed=int(seed), artifacts_dir=Path(run_dir_s))
    payload, parseval_history, pm = train_and_export(
        cfg,
        tag,
        wb,
        figures_run_dir=Path(run_dir_s),
        skip_plots=skip_plots,
        skip_local_aggregate_plots=True,
        wandb_managed_externally=False,
    )
    return payload, parseval_history, pm


def parse_args(a=None):
    p = argparse.ArgumentParser(
        description="Offline distribution analysis: CTD, QTD, φTD (Dirac/cat/quant), save artifacts, plot locally.",
    )
    p.add_argument(
        "--aggregate-from-wandb",
        action="store_true",
        help="Skip training; download W&B runs by --experiment-tag and write aggregate figures.",
    )
    p.add_argument("--config", type=Path, default=None, help="YAML overriding defaults (default: package config/analysis/distribution_quick_5seeds.yaml).")
    p.add_argument("--figures-dir", type=Path, default=None, help="Root for run output (default: <repo>/figures/distribution_analysis).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        dest="num_seeds",
        help="Independent runs with seeds seed, seed+1, ...; combined plots/CSV use 95%% Student CIs for the mean (default: DAConfig / YAML).",
    )
    p.add_argument("--experiment-tag", default="DistAnalysis")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument(
        "--parseval-every",
        type=int,
        default=None,
        help="Gradient steps between parseval snapshots (must divide --total-steps); also JIT chunk size.",
    )
    p.add_argument("--dataset-size", type=int)
    p.add_argument("--omega-laplace-scale", type=float)
    p.add_argument("--num-qtd-quantiles", type=int)
    p.add_argument(
        "--phi-qt-same-arch-as-qtd",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="phi_qt_same_arch_as_qtd",
        help="When true (default from DAConfig), φTD Quant and QTD use the same QTDNet(width, layers, M) "
        "spec (separate weights). When false, φ Quant head width is phi_qt_num_quantiles (or num_tau).",
    )
    p.add_argument(
        "--phi-qt-num-quantiles",
        type=int,
        default=None,
        dest="phi_qt_num_quantiles",
        help="φTD Quant output size when --no-phi-qt-same-arch-as-qtd (default: num_tau).",
    )
    p.add_argument("--num-ctd-atoms", type=int)
    p.add_argument(
        "--num-dirac-components",
        type=int,
        dest="num_dirac_components",
        help="Mixture size M for the φTD Dirac head (alias: --num-mog-components).",
    )
    p.add_argument(
        "--num-mog-components",
        type=int,
        dest="num_dirac_components",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--close-to-theory",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="close_to_theory",
        help="φTD only: Pareto(α=1) ω on [omega_min_pareto, omega_clip] + unweighted CF MSE. "
        "Use --no-close-to-theory or omit for half-Laplacian ω + CF MSE/ω² (default from YAML).",
    )
    p.add_argument(
        "--omega-min-pareto",
        type=float,
        default=None,
        dest="omega_min_pareto",
        help="Lower truncation for Pareto ω when --close-to-theory (default 0.001).",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument(
        "--parallel-seed-workers",
        action="store_true",
        help="When num_seeds>1, run each seed in a separate process (can CUDA OOM on one GPU; default is sequential).",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib outputs (parseval + panel/metric figures).")
    p.add_argument("--wandb-project", default=None, help="W&B project (for --aggregate-from-wandb; else env WANDB_PROJECT).")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (for --aggregate-from-wandb; else env WANDB_ENTITY).")
    return p.parse_args(a)


def main(a=None):
    args = parse_args(a)
    figures_root = args.figures_dir if args.figures_dir is not None else (_REPO_ROOT / "figures" / "distribution_analysis")

    if args.aggregate_from_wandb:
        tag = os.environ.get("WANDB_EXPERIMENT_TAG", args.experiment_tag)
        project = args.wandb_project or os.environ.get("WANDB_PROJECT", "Deep-CVI-Experiments")
        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        out = plot_distribution_analysis_from_wandb(tag, project=project, entity=entity, figures_dir=str(figures_root))
        print(f"Runs aggregated: {out['num_runs']}")
        print(f"Output folder: {out['output_dir']}")
        for k in ("metrics_png", "panels_png", "metrics_csv"):
            if k in out:
                print(f"  {k}: {out[k]}")
        return

    now = dt.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    run_dir = figures_root / timestamp_str
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run artifacts and figures: {run_dir}", flush=True)

    overrides = {
        "seed": args.seed,
        "num_seeds": args.num_seeds,
        "total_steps": args.total_steps,
        "log_every": args.log_every,
        "parseval_every": args.parseval_every,
        "dataset_size": args.dataset_size,
        "omega_laplace_scale": args.omega_laplace_scale,
        "num_tau": args.num_qtd_quantiles,
        "num_atoms": args.num_ctd_atoms,
        "num_dirac_components": args.num_dirac_components,
    }
    if getattr(args, "close_to_theory", None) is not None:
        overrides["close_to_theory"] = args.close_to_theory
    if getattr(args, "omega_min_pareto", None) is not None:
        overrides["omega_min_pareto"] = args.omega_min_pareto
    if getattr(args, "phi_qt_same_arch_as_qtd", None) is not None:
        overrides["phi_qt_same_arch_as_qtd"] = args.phi_qt_same_arch_as_qtd
    if getattr(args, "phi_qt_num_quantiles", None) is not None:
        overrides["phi_qt_num_quantiles"] = args.phi_qt_num_quantiles
    cfg = load_da_config_yaml(args.config, overrides)
    cfg = replace(cfg, artifacts_dir=run_dir)

    use_wandb = not args.no_wandb
    tag = os.environ.get("WANDB_EXPERIMENT_TAG", args.experiment_tag)
    n_seeds = int(cfg.num_seeds)

    if n_seeds <= 1:
        if use_wandb:
            import wandb

            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "Deep-CVI-Experiments"),
                entity=os.environ.get("WANDB_ENTITY"),
                group=tag,
                name=f"{tag}_seed{cfg.seed}",
                tags=[tag],
                config=_config_for_logging(cfg),
            )
        try:
            train_and_export(
                cfg,
                tag,
                use_wandb,
                figures_run_dir=run_dir,
                skip_plots=args.no_plots,
                wandb_managed_externally=use_wandb,
            )
        finally:
            if use_wandb:
                import wandb

                if wandb.run:
                    wandb.finish()
    else:
        seeds = [int(cfg.seed) + i for i in range(n_seeds)]
        parseval_histories: list[dict[str, list[float]]] = []
        payloads: list[dict] = []

        if args.parallel_seed_workers:
            print(
                f"Running {n_seeds} seeds in parallel (spawn); expect ~one JAX GPU context per worker — "
                f"may OOM on a single GPU. base_seed={cfg.seed}",
                flush=True,
            )
            cfg_blob = pickle.dumps(cfg)
            worker_args = [(cfg_blob, s, str(run_dir), tag, use_wandb, args.no_plots) for s in seeds]
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_seeds, mp_context=ctx) as ex:
                futures = [ex.submit(_distribution_analysis_seed_worker, a) for a in worker_args]
                for fut in futures:
                    payload, ph, _pm = fut.result()
                    payloads.append(payload)
                    parseval_histories.append(ph)
        else:
            print(
                f"Running {n_seeds} seeds sequentially (one process / one GPU context); base_seed={cfg.seed}",
                flush=True,
            )
            for s in seeds:
                cfg_s = replace(cfg, seed=s)
                payload, ph, _pm = train_and_export(
                    cfg_s,
                    tag,
                    use_wandb,
                    figures_run_dir=run_dir,
                    skip_plots=args.no_plots,
                    skip_local_aggregate_plots=True,
                    wandb_managed_externally=False,
                )
                payloads.append(payload)
                parseval_histories.append(ph)

        if not args.no_plots:
            merged_ph = _merge_parseval_histories(parseval_histories)
            plot_cfg = replace(cfg, seed=int(cfg.seed))
            _save_parseval_training_plot(plot_cfg, tag, merged_ph, run_dir)
            try:
                agg_paths = plot_distribution_analysis_local(payloads, run_dir)
                print(f"Saved aggregate figures: {agg_paths}", flush=True)
            except Exception as exc:
                print(f"warning: aggregate plotting failed: {exc}", flush=True)


def replot(figures_dir: Path | None = None) -> None:
    """Re-generate plots from the most recent local payload(s) without retraining."""
    figures_root = figures_dir if figures_dir is not None else (_REPO_ROOT / "figures" / "distribution_analysis")
    run_dirs = sorted([p for p in figures_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {figures_root}")

    latest: tuple[Path, list[Path]] | None = None
    for run_dir in reversed(run_dirs):
        main_p = run_dir / "distribution_analysis_payload.json"
        if main_p.exists():
            latest = (run_dir, [main_p])
            break
        seed_ps = sorted(run_dir.glob("distribution_analysis_payload_seed*.json"))
        if seed_ps:
            latest = (run_dir, seed_ps)
            break
    if latest is None:
        raise FileNotFoundError(
            f"No distribution_analysis_payload.json or distribution_analysis_payload_seed*.json under: {figures_root}"
        )

    run_dir, paths = latest
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    out = plot_distribution_analysis_local(payloads, run_dir)
    print(f"Replotted from: {[str(p) for p in paths]}")
    print(f"Output folder: {run_dir}")
    for k in ("metrics_png", "panels_png", "metrics_csv"):
        if k in out:
            print(f"  {k}: {out[k]}")


if __name__ == "__main__":
    main(sys.argv[1:])
    # replot()
