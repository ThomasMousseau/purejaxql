#!/usr/bin/env python3
"""IQN, QTD, CTD, MoG-CQN offline bandit (self-contained).

Config: ``purejaxql/config/analysis/distribution.yaml`` (overridable via ``--config``).
Each run writes under ``figures/distribution_analysis/<YYYY-mm-dd_HH:MM:SS>/`` (MoG-CQN
parseval plot/CSV, panel/metric figures, payload JSON, checkpoints, resolved experiment YAML).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import tempfile
import time
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
jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parent


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


def cf_mse_vs_truth(phi_hat: jax.Array, phi_t: jax.Array) -> jax.Array:
    d = phi_hat - phi_t
    return jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)


def cramer_l2_sq_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid(jnp.square(f_hat - f_true), x_grid)


def cf_l2_sq_over_omega2(phi_hat: jax.Array, phi_t: jax.Array, omega_grid: jax.Array, omega_eps: float) -> jax.Array:
    d = phi_hat - phi_t
    num = jnp.real(d) ** 2 + jnp.imag(d) ** 2
    den = jnp.maximum(jnp.square(jnp.abs(omega_grid)), jnp.float64(omega_eps**2))
    return jnp.trapezoid(num / den, omega_grid)


def ks_on_grid(f_hat: jax.Array, f_true: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(f_hat - f_true))


def w1_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid(jnp.abs(f_hat - f_true), x_grid)


def empirical_cdf(sorted_samples: jax.Array, x_grid: jax.Array) -> jax.Array:
    n = sorted_samples.shape[0]
    idx = jnp.searchsorted(sorted_samples, x_grid, side="right")
    return idx.astype(jnp.float64) / jnp.maximum(n, 1)


def empirical_cf_from_samples(samples: jax.Array, t_grid: jax.Array) -> jax.Array:
    return jnp.mean(jnp.exp(1j * t_grid[:, None] * samples[None, :]), axis=1)


def cvar_from_cdf(F: jax.Array, x_grid: jax.Array, alpha: float = 0.05) -> jax.Array:
    F_mono = jnp.clip(jax.lax.cummax(F.astype(jnp.float64)), 0.0, 1.0)
    u = jnp.linspace(alpha / 200.0, alpha, 200, dtype=jnp.float64)
    Q = jnp.interp(u, F_mono, x_grid.astype(jnp.float64))
    return jnp.mean(Q)


def entropic_risk_from_cdf(F: jax.Array, x_grid: jax.Array, beta: float) -> jax.Array:
    F64, x64 = F.astype(jnp.float64), x_grid.astype(jnp.float64)
    dF = jnp.clip(jnp.diff(F64), 0.0, None)
    xm = 0.5 * (x64[:-1] + x64[1:])
    ex = jnp.maximum(jnp.sum(jnp.exp(beta * xm) * dF), 1e-300)
    return jnp.log(ex) / beta


def entropic_risk_mog(pi: jax.Array, mu: jax.Array, sigma: jax.Array, beta: float) -> jax.Array:
    dt = jnp.result_type(pi, mu, sigma)
    b = jnp.asarray(beta, dtype=dt)
    mgf = jnp.exp(b * mu + 0.5 * (b**2) * jnp.square(sigma))
    ee = jnp.maximum(jnp.sum(pi * mgf, axis=-1), jnp.finfo(dt).tiny)
    return jnp.where(jnp.abs(b) < 1e-8, jnp.sum(pi * mu, axis=-1), jnp.log(ee) / b)


def exact_mog_cdf(x_grid: jax.Array, pi: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    z = (x_grid[:, None] - mu) / (sigma * jnp.sqrt(2.0))
    return jnp.sum(pi * 0.5 * (1.0 + jax.scipy.special.erf(z)), axis=-1)


def compose_mog_cf(omega_signed: jax.Array, pi: jax.Array, mu: jax.Array, sigma: jax.Array, *, dtype=jnp.complex128) -> jax.Array:
    w = omega_signed[..., None]
    if pi.ndim == omega_signed.ndim + 1:
        pass
    elif omega_signed.ndim == 1 and pi.ndim == 1:
        pi, mu, sigma = pi[None, :], mu[None, :], sigma[None, :]
    else:
        pi, mu, sigma = pi[..., None, :], mu[..., None, :], sigma[..., None, :]
    term = jnp.exp((1j * w * mu).astype(dtype) - 0.5 * jnp.square(w).astype(dtype) * jnp.square(sigma).astype(dtype))
    return jnp.sum(pi.astype(dtype) * term, axis=-1)


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


def sample_frequencies_half_laplacian(key: jax.Array, num_samples: int, omega_max: jax.Array, scale: jax.Array) -> jax.Array:
    u = jax.random.uniform(key, (num_samples,))
    mcdf = 1.0 - jnp.exp(-omega_max / scale)
    return -scale * jnp.log(1.0 - u * mcdf)


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
PANEL_ANCHOR_PRNG_KEY = jax.random.PRNGKey(42_424_242)

DA_ORDER = ("iqn", "qtd", "ctd", "mog_cqn")
LEGACY_ALGO_ALIASES = {"qtd": "qr_dqn", "ctd": "c51"}
MK = ("phi_mse", "ks", "w1", "cramer_l2", "cf_l2_w", "cvar_err", "er_err", "mean_err", "var_err", "mv_err")


@dataclass(frozen=True)
class DAConfig:
    seed: int = 0
    state_dim: int = STATE_DIM
    dataset_size: int = 500_000
    dataset_seed_offset: int = DATASET_SEED_OFFSET
    total_steps: int = 20_000 #100_000
    log_every: int = 20_000
    # Gradient steps per JIT chunk; parseval / anchor metrics are logged after each chunk (use < total_steps for curves).
    parseval_every: int = 1_000
    batch_size: int = 1024
    num_tau: int = 51
    num_atoms: int = 51
    num_mog_components: int = 51
    num_omega: int = 32
    gamma: float = 0.95
    immediate_reward_std: float = 0.1
    omega_max: float = 15.0
    huber_kappa: float = 1.0
    lr: float = 1e-3
    max_grad_norm: float = 10.0
    omega_clip: float = 15.0
    omega_laplace_scale: float = 0.8
    cf_loss_omega_eps: float = 1e-3
    sigma_min: float = 1e-6
    sigma_max: float = 20.0
    aux_mean_loss_weight: float = 0.1
    c51_v_min: float = -25.0
    c51_v_max: float = 25.0
    embed_dim: int = 64
    hidden_dim: int = 256
    trunk_layers: int = 2
    n_test: int = 10_000
    test_seed_offset: int = 7_777
    eval_chunk_size: int = 50
    num_taus_iqn_eval: int = 2_000
    eval_iqn_seed: int = 42
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


class IQNNet(nn.Module):
    embed_dim: int
    hidden_dim: int
    trunk_layers: int

    @nn.compact
    def __call__(self, x, a_onehot, tau):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        tau_c = jnp.clip(tau, 1e-6, 1 - 1e-6)
        idx = jnp.arange(1, self.embed_dim + 1, dtype=tau.dtype)
        emb = jnp.cos(jnp.pi * tau_c[..., None] * idx)
        emb = _dense_ln_relu(emb, self.hidden_dim, "tp")
        m = _dense_ln_relu(h[:, None, :] * emb, self.hidden_dim, "pm")
        return nn.Dense(1, name="hd")(m).squeeze(-1)


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


class MoGCQNNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int
    sigma_min: float
    sigma_max: float

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = nn.relu(nn.Dense(self.hidden_dim, name=f"t{i}")(h))
        pi = jax.nn.softmax(nn.Dense(self.num_components, name="pi")(h), -1)
        mu = nn.Dense(self.num_components, name="mu")(h)
        rs = nn.Dense(self.num_components, name="sg")(h)
        sg = jnp.minimum(nn.softplus(rs) + self.sigma_min, self.sigma_max)
        return pi, mu, sg


def _opt(lr: float, clip: float):
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def create_train_states(rng: jax.Array, cfg: DAConfig):
    ri, rq, r5, rm = jax.random.split(rng, 4)
    iqn = IQNNet(cfg.embed_dim, cfg.hidden_dim, cfg.trunk_layers)
    qtd = QTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_tau)
    ctd = CTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    mog = MoGCQNNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_mog_components, cfg.sigma_min, cfg.sigma_max)
    x0, a0 = jnp.zeros((2, cfg.state_dim), jnp.float32), jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    tau0 = jnp.full((2, cfg.num_tau), 0.5, jnp.float32)
    tx = _opt(cfg.lr, cfg.max_grad_norm)
    vi, vq, v5, vm = iqn.init(ri, x0, a0, tau0), qtd.init(rq, x0, a0), ctd.init(r5, x0, a0), mog.init(rm, x0, a0)
    return (
        train_state.TrainState.create(apply_fn=iqn.apply, params=vi["params"], tx=tx),
        train_state.TrainState.create(apply_fn=qtd.apply, params=vq["params"], tx=tx),
        train_state.TrainState.create(apply_fn=ctd.apply, params=v5["params"], tx=tx),
        train_state.TrainState.create(apply_fn=mog.apply, params=vm["params"], tx=tx),
        iqn, qtd, ctd, mog,
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


def make_train_chunk(cfg: DAConfig, iqn: IQNNet, qtd: QTDNet, ctd: CTDNet, mog: MoGCQNNet, ds):
    xa, aa, ra, zn = ds
    n, aux_w = xa.shape[0], jnp.float32(cfg.aux_mean_loss_weight)
    g, atoms = jnp.float32(cfg.gamma), ctd_atoms(cfg)
    phis = [d.phi for d in ACTION_DISTS]

    def step(c, _):
        si, sq, s5, sm, rng = c
        rng, kb, kt, ko = jax.random.split(rng, 4)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        xb, ab, rb, znb = xa[idx], aa[idx], ra[idx], zn[idx]
        oh = jax.nn.one_hot(ab, NUM_ACTIONS, dtype=jnp.float32)
        tau = jax.random.uniform(kt, (cfg.batch_size, cfg.num_tau), jnp.float32)
        tauf = jnp.broadcast_to(fixed_qr_tau_grid(cfg.num_tau)[None, :], (cfg.batch_size, cfg.num_tau))
        om = sample_frequencies_half_laplacian(
            ko, cfg.batch_size * cfg.num_omega, jnp.asarray(cfg.omega_clip, jnp.float32), jnp.asarray(cfg.omega_laplace_scale, jnp.float32),
        ).reshape(cfg.batch_size, cfg.num_omega)
        tgt = rb + g * znb

        def li(pp):
            return quantile_huber_loss_pairwise(iqn.apply({"params": pp}, xb, oh, tau), tgt, tau, cfg.huber_kappa)

        def lq(pp):
            return quantile_huber_loss_pairwise(qtd.apply({"params": pp}, xb, oh), tgt, tauf, cfg.huber_kappa)

        def l5(pp):
            lg = jax.nn.log_softmax(ctd.apply({"params": pp}, xb, oh))
            tp = project_delta_onto_c51_atoms(tgt, atoms)
            return -jnp.mean(jnp.sum(tp * lg, -1))

        def lm(pp):
            pi, mu, sg = mog.apply({"params": pp}, xb, oh)
            pred = compose_mog_cf(om, pi, mu, sg, dtype=jnp.complex64)
            gw = g * om
            fr = jnp.exp(1j * om.astype(jnp.complex64) * rb[:, None].astype(jnp.complex64))

            def ph1(ai, wi, xi):
                return jax.lax.switch(ai, phis, wi.astype(jnp.float64), xi.astype(jnp.float64))

            fz = jax.vmap(ph1)(ab, gw, xb).astype(jnp.complex64)
            d = pred - fr * fz
            w2 = jnp.maximum(jnp.square(om), jnp.float32(cfg.cf_loss_omega_eps**2))
            cf = 0.5 * jnp.mean((jnp.real(d) ** 2 + jnp.imag(d) ** 2) / w2)
            mp = jnp.sum(pi * mu, -1)
            return cf + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        vi, gi = jax.value_and_grad(li)(si.params)
        vq, gq = jax.value_and_grad(lq)(sq.params)
        v5, g5 = jax.value_and_grad(l5)(s5.params)
        vm, gm = jax.value_and_grad(lm)(sm.params)
        return (
            si.apply_gradients(grads=gi),
            sq.apply_gradients(grads=gq),
            s5.apply_gradients(grads=g5),
            sm.apply_gradients(grads=gm),
            rng,
        ), (vi, vq, v5, vm)

    scan_len = int(cfg.parseval_every)

    @jax.jit
    def run(si, sq, s5, sm, rng):
        (si, sq, s5, sm, rng), loss = jax.lax.scan(step, (si, sq, s5, sm, rng), None, scan_len)
        return si, sq, s5, sm, rng, loss

    return run


def _make_eval(cfg: DAConfig, iqn: IQNNet, qtd: QTDNet, ctd: CTDNet, mog: MoGCQNNet):
    wm = cfg.omega_max
    te = jnp.linspace(-wm, wm, cfg.eval_num_t, jnp.float64)
    xe = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, jnp.float64)
    xr = jnp.linspace(cfg.risk_x_min, cfg.risk_x_max, cfg.risk_num_x, jnp.float64)
    tp = jnp.linspace(cfg.gp_t_delta, wm, cfg.gp_num_t, jnp.float64)
    g64, rs64 = jnp.float64(cfg.gamma), jnp.float64(cfg.immediate_reward_std)
    taue = jax.random.uniform(jax.random.PRNGKey(cfg.eval_iqn_seed), (cfg.num_taus_iqn_eval,), jnp.float32)
    atoms = ctd_atoms(cfg)
    phis = [d.phi for d in ACTION_DISTS]

    def one(pi, pq, p5, pm, xs, k):
        oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1, a1 = xs[None, :].astype(jnp.float32), oh[None, :]
        qs = iqn.apply({"params": pi}, x1, a1, taue[None, :])[0].astype(jnp.float64)
        Fi, Fir = empirical_cdf(jnp.sort(qs), xe), empirical_cdf(jnp.sort(qs), xr)
        phii = empirical_cf_from_samples(qs.astype(jnp.complex128), te.astype(jnp.complex128))
        qq = qtd.apply({"params": pq}, x1, a1)[0].astype(jnp.float64)
        Fq, Fqr = empirical_cdf(jnp.sort(qq), xe), empirical_cdf(jnp.sort(qq), xr)
        phiq = empirical_cf_from_samples(qq.astype(jnp.complex128), te.astype(jnp.complex128))
        lc = ctd.apply({"params": p5}, x1, a1)[0]
        pc = jax.nn.softmax(lc)
        F5, F5r = discrete_return_cdf(pc, atoms, xe), discrete_return_cdf(pc, atoms, xr)
        phi5 = discrete_return_cf(pc, atoms, te)
        piv, mv, sv = mog.apply({"params": pm}, x1, a1)
        pmv, mmv, smv = piv[0], mv[0], sv[0]
        phim = compose_mog_cf(te, pmv, mmv, smv)
        Fm, Fmr = exact_mog_cdf(xe, pmv, mmv, smv), exact_mog_cdf(xr, pmv, mmv, smv)
        pr = jnp.exp(-0.5 * (rs64**2) * te**2)
        pz = jax.lax.switch(k, phis, g64 * te, xs.astype(jnp.float64))
        pt = pr * pz

        prp = jnp.exp(-0.5 * (rs64**2) * tp**2)
        pzp = jax.lax.switch(k, phis, g64 * tp, xs.astype(jnp.float64))
        ptp = prp * pzp
        Ft = gil_pelaez_cdf(ptp, tp, xe)
        Ftr = gil_pelaez_cdf(ptp, tp, xr)
        pdf_t = jnp.clip(jnp.gradient(Ft, xe), 0, None)

        pdfi = jnp.clip(jnp.gradient(Fi, xe), 0, None)
        pdfq = jnp.clip(jnp.gradient(Fq, xe), 0, None)
        pdf5 = jnp.clip(jnp.gradient(F5, xe), 0, None)
        pdfm = jnp.clip(jnp.gradient(Fm, xe), 0, None)
       # Get exact analytical truth for all 4 metrics!
        mtn, vtn, mvtn, ertn = analytic_bellman_moments(
            k, xs, g64, rs64, jnp.float64(cfg.mv_lambda), jnp.float64(cfg.er_beta)
        )
        
        cv_t = cvar_from_cdf(Ftr, xr, cfg.cvar_alpha)
        
        # Calculate errors against the exact analytical Entropic Risk (ertn)
        er_m = entropic_risk_mog(pmv, mmv, smv, cfg.er_beta)
        er_err_iqn = jnp.abs(entropic_risk_from_cdf(Fir, xr, cfg.er_beta) - ertn)
        er_err_qr_dqn = jnp.abs(entropic_risk_from_cdf(Fqr, xr, cfg.er_beta) - ertn)
        er_err_c51 = jnp.abs(entropic_risk_from_cdf(F5r, xr, cfg.er_beta) - ertn)
        er_err_mog_cqn = jnp.abs(er_m - ertn)
        er_undefined = k == 1
        # if er_undefined:
        #     er_err_iqn = jnp.float64(jnp.nan)
        #     er_err_qr_dqn = jnp.float64(jnp.nan)
        #     er_err_c51 = jnp.float64(jnp.nan)
        #     er_err_mog_cqn = jnp.float64(jnp.nan)
        # else:
        #     er_t = entropic_risk_from_cdf(Ftr, xr, cfg.er_beta)
        #     er_m = entropic_risk_mog(pmv, mmv, smv, cfg.er_beta)
        #     er_err_iqn = jnp.abs(entropic_risk_from_cdf(Fir, xr, cfg.er_beta) - er_t)
        #     er_err_qr_dqn = jnp.abs(entropic_risk_from_cdf(Fqr, xr, cfg.er_beta) - er_t)
        #     er_err_c51 = jnp.abs(entropic_risk_from_cdf(F5r, xr, cfg.er_beta) - er_t)
        #     er_err_mog_cqn = jnp.abs(er_m - er_t)
        mi, mqi = jnp.mean(qs), jnp.mean(jnp.square(qs - jnp.mean(qs)))
        mq, mqq = jnp.mean(qq), jnp.mean(jnp.square(qq - jnp.mean(qq)))
        m5 = jnp.sum(pc * atoms)
        v5 = jnp.sum(pc * atoms**2) - m5**2
        mm = jnp.sum(pmv * mmv)
        vm = jnp.sum(pmv * (smv**2 + mmv**2)) - mm**2

        def mer(mp, vp, mf, vf):
            return jnp.abs(mp - mf), jnp.abs(vp - vf), jnp.abs((mp - cfg.mv_lambda * vp) - mvtn)

        mei = mer(mi, mqi, mtn, vtn)
        meq = mer(mq, mqq, mtn, vtn)
        me5 = mer(m5.astype(jnp.float64), v5.astype(jnp.float64), mtn, vtn)
        mem = mer(mm.astype(jnp.float64), vm.astype(jnp.float64), mtn, vtn)

        cf_l2_iqn = cf_l2_sq_over_omega2(phii, pt, te, cfg.gp_t_delta)
        cf_l2_qtd = cf_l2_sq_over_omega2(phiq, pt, te, cfg.gp_t_delta)
        cf_l2_ctd = cf_l2_sq_over_omega2(phi5, pt, te, cfg.gp_t_delta)
        cf_l2_mog = cf_l2_sq_over_omega2(phim, pt, te, cfg.gp_t_delta)

        out = {
            "phi_mse_iqn": cf_mse_vs_truth(phii, pt),
            "phi_mse_qtd": cf_mse_vs_truth(phiq, pt),
            "phi_mse_ctd": cf_mse_vs_truth(phi5, pt),
            "phi_mse_mog_cqn": cf_mse_vs_truth(phim, pt),
            "ks_iqn": ks_on_grid(Fi, Ft),
            "ks_qtd": ks_on_grid(Fq, Ft),
            "ks_ctd": ks_on_grid(F5, Ft),
            "ks_mog_cqn": ks_on_grid(Fm, Ft),
            "w1_iqn": w1_cdf(Fi, Ft, xe),
            "w1_qtd": w1_cdf(Fq, Ft, xe),
            "w1_ctd": w1_cdf(F5, Ft, xe),
            "w1_mog_cqn": w1_cdf(Fm, Ft, xe),
            "cramer_l2_iqn": cramer_l2_sq_cdf(Fi, Ft, xe),
            "cramer_l2_qtd": cramer_l2_sq_cdf(Fq, Ft, xe),
            "cramer_l2_ctd": cramer_l2_sq_cdf(F5, Ft, xe),
            "cramer_l2_mog_cqn": cramer_l2_sq_cdf(Fm, Ft, xe),
            "cf_l2_w_iqn": cf_l2_iqn,
            "cf_l2_w_qtd": cf_l2_qtd,
            "cf_l2_w_ctd": cf_l2_ctd,
            "cf_l2_w_mog_cqn": cf_l2_mog,
            "cvar_err_iqn": jnp.abs(cvar_from_cdf(Fir, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_qtd": jnp.abs(cvar_from_cdf(Fqr, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_ctd": jnp.abs(cvar_from_cdf(F5r, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_mog_cqn": jnp.abs(cvar_from_cdf(Fmr, xr, cfg.cvar_alpha) - cv_t),
            "er_err_iqn": er_err_iqn,
            "er_err_qtd": er_err_qr_dqn,
            "er_err_ctd": er_err_c51,
            "er_err_mog_cqn": er_err_mog_cqn,
            "phi_iqn": phii,
            "phi_qtd": phiq,
            "phi_ctd": phi5,
            "phi_mog_cqn": phim,
            "phi_true": pt,
            "F_iqn": Fi,
            "F_qtd": Fq,
            "F_ctd": F5,
            "F_mog_cqn": Fm,
            "F_true": Ft,
            "pdf_true": pdf_t,
            "pdf_iqn": pdfi,
            "pdf_qtd": pdfq,
            "pdf_ctd": pdf5,
            "pdf_mog_cqn": pdfm,
            "mean_err_iqn": mei[0],
            "mean_err_qtd": meq[0],
            "mean_err_ctd": me5[0],
            "mean_err_mog_cqn": mem[0],
            "var_err_iqn": mei[1],
            "var_err_qtd": meq[1],
            "var_err_ctd": me5[1],
            "var_err_mog_cqn": mem[1],
            "mv_err_iqn": mei[2],
            "mv_err_qtd": meq[2],
            "mv_err_ctd": me5[2],
            "mv_err_mog_cqn": mem[2],
        }

        # Legacy aliases for existing scripts/plots expecting QR-DQN/C51 keys.
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

        return out

    @jax.jit
    def ev(pi, pq, p5, pm, xs):
        P = [one(pi, pq, p5, pm, xs, kk) for kk in range(NUM_ACTIONS)]
        return {k: jnp.stack([p[k] for p in P]) for k in P[0]}

    return ev, te, xe


def evaluate_test_set(cfg: DAConfig, nets, params, xt):
    ev, te, xe = _make_eval(cfg, *nets)
    ech = jax.jit(jax.vmap(ev, (None, None, None, None, 0)))
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
    "phi_iqn",
    "phi_qtd",
    "phi_ctd",
    "phi_qr_dqn",
    "phi_c51",
    "phi_mog_cqn",
    "F_true",
    "F_iqn",
    "F_qtd",
    "F_ctd",
    "F_qr_dqn",
    "F_c51",
    "F_mog_cqn",
    "pdf_iqn",
    "pdf_qtd",
    "pdf_ctd",
    "pdf_qr_dqn",
    "pdf_c51",
    "pdf_mog_cqn",
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
    """Plot / CSV for MoG-CQN only (Cramér vs CF on the parseval anchor)."""
    if not history.get("step"):
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"warning: parseval plot skipped (matplotlib import failed: {exc})", flush=True)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"distribution_analysis_parseval_training_mog_cqn_seed{cfg.seed}"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    csv_path = stem.with_suffix(".csv")

    steps = np.asarray(history["step"], dtype=np.float64)

    def _as_seed_matrix(values: list[float] | list[list[float]]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Expected 1D or 2D parseval history, got shape {arr.shape}.")

    cr_matrix = _as_seed_matrix(history["cramer_l2_mog_cqn"])
    cf_matrix = _as_seed_matrix(history["cf_l2_w_mog_cqn"])
    y_cr = np.clip(np.nan_to_num(np.nanmean(cr_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_cf = np.clip(np.nan_to_num(np.nanmean(cf_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_cr_std = np.nan_to_num(np.nanstd(cr_matrix, axis=0, ddof=0), nan=0.0, posinf=0.0, neginf=0.0)
    y_cf_std = np.nan_to_num(np.nanstd(cf_matrix, axis=0, ddof=0), nan=0.0, posinf=0.0, neginf=0.0)
    line_color = algo_color("mog_cqn")

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
    ax.fill_between(steps, np.clip(y_cr - y_cr_std, 0.0, None), y_cr + y_cr_std, color=line_color, alpha=0.18, lw=0)
    ax.fill_between(steps, np.clip(y_cf - y_cf_std, 0.0, None), y_cf + y_cf_std, color=line_color, alpha=0.1, lw=0)
    ax.plot(steps, y_cr, color=line_color, lw=2.0, label=r"Cramér $\ell_2^2$ (CDF)")
    ax.plot(steps, y_cf, color=line_color, lw=2.0, ls="--", label=r"CF $\ell_2^2/\omega^2$")
    pos = np.concatenate([y_cr[np.isfinite(y_cr)], y_cf[np.isfinite(y_cf)]])
    pos = pos[pos > 0]
    if pos.size and float(np.nanmin(pos)) > 0:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", color="#D8D8D8", alpha=0.7)
    ax.set_title("Empirical Parseval-Plancherel Theorem Validation")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=9)
    # fig.suptitle(f"Training ({tag})", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    fieldnames = (
        "step",
        "cramer_l2_mog_cqn",
        "cf_l2_w_mog_cqn",
        "cramer_l2_mog_cqn_std",
        "cf_l2_w_mog_cqn_std",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, st in enumerate(steps.tolist()):
            writer.writerow(
                {
                    "step": st,
                    "cramer_l2_mog_cqn": y_cr[i],
                    "cf_l2_w_mog_cqn": y_cf[i],
                    "cramer_l2_mog_cqn_std": y_cr_std[i],
                    "cf_l2_w_mog_cqn_std": y_cf_std[i],
                }
            )

    return {"png": str(png_path), "pdf": str(pdf_path), "csv": str(csv_path)}


def save_msgpack(cfg: DAConfig, *params):
    os.makedirs(os.path.dirname(cfg.msgpack_filename()), exist_ok=True)
    iqn_p, qtd_p, ctd_p, mog_p = params
    with open(cfg.msgpack_filename(), "wb") as f:
        f.write(
            flax.serialization.to_bytes(
                {
                    "iqn": iqn_p,
                    "qtd": qtd_p,
                    "ctd": ctd_p,
                    "mog_cqn": mog_p,
                    "qr_dqn": qtd_p,
                    "c51": ctd_p,
                }
            )
        )


# --- Figures (merged from plot_distribution_analysis.py) ---------------------------------
PLOT_DA_TITLES = {"iqn": "IQN", "qtd": "QTD", "ctd": "CTD", "mog_cqn": "MoG-CQN"}
PLOT_ROW_LBL = {
    "phi_mse": r"$\varphi$-MSE $\downarrow$",
    "ks": r"KS $\downarrow$",
    "w1": r"$W_1$ $\downarrow$",
    "cramer_l2": r"Cramér $\ell_2^2$ $\downarrow$",
    "cf_l2_w": r"CF-$\ell_2^2/\omega^2$ $\downarrow$",
    "cvar_err": r"CVaR err $\downarrow$",
    "er_err": r"Entropic err $\downarrow$",
    "mean_err": r"Mean err $\downarrow$",
    "var_err": r"Var err $\downarrow$",
    "mv_err": r"M-V err $\downarrow$",
}
PLOT_WIN_BG = "#D2F4D9"
PLOT_COLORS = {
    "truth": "#0A0A0A",
    "iqn": algo_color("iqn"),
    "qtd": algo_color("qtd"),
    "ctd": algo_color("ctd"),
    "mog_cqn": algo_color("mog_cqn"),
    "qr_dqn": algo_color("qr_dqn"),
    "c51": algo_color("c51"),
}
PLOT_LW = {"truth": 2.15, "iqn": 1.05, "qtd": 1.08, "ctd": 1.12, "mog_cqn": 1.2, "qr_dqn": 1.08, "c51": 1.12}


def _plot_minimal_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", color="#E0E0E0", alpha=0.7)
    ax.set_axisbelow(True)


def _plot_save_fig(fig, pdf_path: str, *, png_dpi: int = 300) -> None:
    stem = os.path.splitext(pdf_path)[0]
    fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{stem}.png", format="png", bbox_inches="tight", pad_inches=0.08, dpi=png_dpi)


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


def _plot_metric_tables(stderr: dict[str, tuple[np.ndarray, np.ndarray]], row_keys: tuple[str, ...], path: str) -> None:
    import matplotlib.pyplot as plt

    algos = [a for a in DA_ORDER if a in stderr]
    if not algos:
        raise ValueError("No metric data found for plotting.")
    means = np.stack([stderr[a][0] for a in algos])
    ses = np.stack([stderr[a][1] for a in algos])
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
                sv = ses[i, r, c]
                if np.isfinite(mv) and sv > 0:
                    row.append(f"{mv:.4g}\n({sv:.3g})")
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

    fig.suptitle("Mean ± SE over seeds", fontsize=11, y=0.985)
    _plot_save_fig(fig, path)
    plt.close(fig)


def _plot_panels(mean_p: dict[str, np.ndarray], se_p: dict[str, np.ndarray], te: np.ndarray, xe: np.ndarray, cf_hw: float, path: str) -> None:
    import matplotlib.pyplot as plt

    cf_series = (
        (("panel_phi_true",), "truth", "Truth"),
        (("panel_phi_iqn",), "iqn", "IQN"),
        (("panel_phi_qtd", "panel_phi_qr_dqn"), "qtd", "QTD"),
        (("panel_phi_ctd", "panel_phi_c51"), "ctd", "CTD"),
        (("panel_phi_mog_cqn",), "mog_cqn", "MoG-CQN"),
    )
    pdf_series = (
        (("panel_pdf_true",), "truth", "Truth"),
        (("panel_pdf_iqn",), "iqn", "IQN"),
        (("panel_pdf_qtd", "panel_pdf_qr_dqn"), "qtd", "QTD"),
        (("panel_pdf_ctd", "panel_pdf_c51"), "ctd", "CTD"),
        (("panel_pdf_mog_cqn",), "mog_cqn", "MoG-CQN"),
    )
    cdf_series = (
        (("panel_F_true",), "truth", "Truth"),
        (("panel_F_iqn",), "iqn", "IQN"),
        (("panel_F_qtd", "panel_F_qr_dqn"), "qtd", "QTD"),
        (("panel_F_ctd", "panel_F_c51"), "ctd", "CTD"),
        (("panel_F_mog_cqn",), "mog_cqn", "MoG-CQN"),
    )

    fig, axes = plt.subplots(len(ACTION_DISTS), 4, figsize=(13.5, 2.45 * len(ACTION_DISTS)))
    legend = None
    for row, dist in enumerate(ACTION_DISTS):
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
            ys = np.real(se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row])
            axcf.fill_between(te[tmask], (y - ys)[tmask], (y + ys)[tmask], color=PLOT_COLORS[ckey], alpha=0.18, lw=0)
            axcf.plot(te[tmask], y[tmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axcf)
        axcf.set_xlim(-cf_hw, cf_hw)
        if row == 0:
            axcf.set_title(r"Re $\varphi$")

        for candidates, ckey, label in pdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            if ckey == "truth":
                y = np.asarray(mean_p[pkey][row], dtype=np.float64)
                ys = np.asarray(se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], dtype=np.float64)
            else:
                y = _plot_smooth_curve(mean_p[pkey][row])
                ys = _plot_smooth_curve(se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], passes=1)
            axp.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=PLOT_COLORS[ckey], alpha=0.18, lw=0)
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
            ys = se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row]
            axc.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=PLOT_COLORS[ckey], alpha=0.18, lw=0)
            axc.plot(xe[xmask], y[xmask], color=PLOT_COLORS[ckey], lw=PLOT_LW[ckey], label=label)
        _plot_minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axc.set_title("CDF")

        axd.text(-0.34, 0.5, dist.name, transform=axd.transAxes, rotation=90, ha="center", va="center", fontweight="bold")
        if legend is None:
            legend = axc.get_legend_handles_labels()

    if legend:
        fig.legend(*legend, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Distribution panels", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.09, wspace=0.3, hspace=0.42)
    _plot_save_fig(fig, path)
    plt.close(fig)


def _plot_aggregate_payloads(
    payloads: list[dict], row_keys: tuple[str, ...]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
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


def _plot_write_metrics_csv(path: str, metric_stats: dict[str, tuple[np.ndarray, np.ndarray]], row_keys: tuple[str, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["algorithm", "metric"] + [f"{x.short}_mean" for x in ACTION_DISTS] + [f"{x.short}_se" for x in ACTION_DISTS]
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
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    output_dir.mkdir(parents=True, exist_ok=True)
    row_keys = _plot_resolve_row_keys(payloads)
    metric_stats, panel_mean, panel_se, t_eval, x_eval, omega_max = _plot_aggregate_payloads(payloads, row_keys)

    metrics_pdf = str(output_dir / "distribution_analysis_metrics_combined.pdf")
    panels_pdf = str(output_dir / "distribution_analysis_panels.pdf")
    metrics_csv = str(output_dir / "distribution_analysis_metrics_combined.csv")

    _plot_metric_tables(metric_stats, row_keys, metrics_pdf)
    _plot_panels(panel_mean, panel_se, t_eval, x_eval, omega_max, panels_pdf)
    _plot_write_metrics_csv(metrics_csv, metric_stats, row_keys)

    return {
        "metrics_pdf": metrics_pdf,
        "metrics_png": os.path.splitext(metrics_pdf)[0] + ".png",
        "panels_pdf": panels_pdf,
        "panels_png": os.path.splitext(panels_pdf)[0] + ".png",
        "metrics_csv": metrics_csv,
    }


def plot_distribution_analysis_from_wandb(
    experiment_tag: str, *, project: str, entity: str | None, figures_dir: str
) -> dict[str, str]:
    """Aggregate finished W&B runs by tag and write figures under figures_dir/<timestamp>."""
    import matplotlib.pyplot as plt
    import wandb

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})

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


def train_and_export(
    cfg: DAConfig,
    tag: str,
    wb: bool,
    *,
    figures_run_dir: Path,
    skip_plots: bool = False,
) -> None:
    figures_run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(figures_run_dir / "distribution_analysis_experiment_config.yaml", _config_for_logging(cfg))

    rng = jax.random.PRNGKey(cfg.seed)
    ri, rd, rt, ru = jax.random.split(rng, 4)
    print(f"Dataset... offset={cfg.dataset_seed_offset}", flush=True)
    ds = build_dataset(jax.random.fold_in(rd, cfg.dataset_seed_offset), cfg)
    jax.tree.map(lambda x: x.block_until_ready(), ds)
    si, sq, s5, sm, *nets = create_train_states(ri, cfg)
    parseval_eval, _, _ = _make_eval(cfg, nets[0], nets[1], nets[2], nets[3])
    anchor_x = jax.random.normal(PANEL_ANCHOR_PRNG_KEY, (cfg.state_dim,), jnp.float32)
    parseval_history: dict[str, list[float]] = {"step": [], "cramer_l2_mog_cqn": [], "cf_l2_w_mog_cqn": []}
    run = make_train_chunk(cfg, nets[0], nets[1], nets[2], nets[3], ds)
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
    wandb_mod = None
    if wb:
        import wandb as _wandb

        wandb_mod = _wandb
    for c in range(nc):
        tc = time.perf_counter()
        si, sq, s5, sm, rng, ls = run(si, sq, s5, sm, rng)
        jax.tree.map(lambda x: x.block_until_ready(), ls)
        st = (c + 1) * pe
        last_losses = {
            "train/loss_iqn_final": float(ls[0][-1]),
            "train/loss_qtd_final": float(ls[1][-1]),
            "train/loss_ctd_final": float(ls[2][-1]),
            "train/loss_mog_cqn_final": float(ls[3][-1]),
        }
        last_losses["train/loss_qr_dqn_final"] = last_losses["train/loss_qtd_final"]
        last_losses["train/loss_c51_final"] = last_losses["train/loss_ctd_final"]
        if st % int(cfg.log_every) == 0 or st == int(cfg.total_steps) or c == 0:
            print(
                f"  {st:>7}  iqn={last_losses['train/loss_iqn_final']:.4f} "
                f"qtd={last_losses['train/loss_qtd_final']:.4f} "
                f"ctd={last_losses['train/loss_ctd_final']:.4f} "
                f"mog={last_losses['train/loss_mog_cqn_final']:.4f}  {time.perf_counter()-tc:.1f}s",
                flush=True,
            )

        pv = parseval_eval(si.params, sq.params, s5.params, sm.params, anchor_x)
        jax.tree.map(lambda x: x.block_until_ready(), pv)
        parseval_history["step"].append(float(st))
        cr_m = float(jnp.mean(pv["cramer_l2_mog_cqn"]))
        cf_m = float(jnp.mean(pv["cf_l2_w_mog_cqn"]))
        parseval_history["cramer_l2_mog_cqn"].append(cr_m)
        parseval_history["cf_l2_w_mog_cqn"].append(cf_m)
        pv_log = {
            "train/parseval/mog_cqn/cramer_l2_anchor": cr_m,
            "train/parseval/mog_cqn/cf_l2_w_anchor": cf_m,
        }
        if wb and wandb_mod is not None:
            wandb_mod.log(pv_log, step=st)

    print(f"train {time.perf_counter()-t0:.1f}s", flush=True)
    parseval_plot_paths = None
    if not skip_plots:
        parseval_plot_paths = _save_parseval_training_plot(cfg, tag, parseval_history, figures_run_dir)
    xt = jax.random.fold_in(rt, cfg.test_seed_offset)
    xv = jax.random.normal(xt, (cfg.n_test, cfg.state_dim), jnp.float32).at[0].set(
        jax.random.normal(PANEL_ANCHOR_PRNG_KEY, (cfg.state_dim,), jnp.float32)
    )
    er, te, xe = evaluate_test_set(cfg, tuple(nets), (si.params, sq.params, s5.params, sm.params), xv)
    jax.tree.map(lambda x: x.block_until_ready(), er)
    payload, pm = _collect_eval_payload(cfg, tag, er, te, xe, panel_index=0)
    save_msgpack(cfg, si.params, sq.params, s5.params, sm.params)
    payload_path = figures_run_dir / "distribution_analysis_payload.json"
    payload_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    if not skip_plots:
        try:
            plot_paths = plot_distribution_analysis_local([payload], figures_run_dir)
            print(f"Saved figures: {plot_paths}", flush=True)
        except Exception as exc:
            print(f"warning: post-run plotting failed: {exc}", flush=True)
    if wb:
        if last_losses:
            wandb_mod.summary.update(last_losses)
        if parseval_plot_paths:
            for p in parseval_plot_paths.values():
                wandb_mod.save(p, policy="now")
            wandb_mod.summary["parseval_train_plot_png"] = parseval_plot_paths["png"]
            wandb_mod.summary["parseval_train_plot_pdf"] = parseval_plot_paths["pdf"]
            wandb_mod.summary["parseval_train_metrics_csv"] = parseval_plot_paths["csv"]
        _wandb_log_eval_metrics(pm, tag, step=cfg.total_steps)
        _save_eval_payload_to_wandb(payload)
        wandb_mod.summary["distribution_analysis_artifacts_dir"] = str(figures_run_dir)


def parse_args(a=None):
    p = argparse.ArgumentParser(
        description="Offline distribution analysis: train IQN/QTD/CTD/MoG-CQN, save artifacts, plot locally.",
    )
    p.add_argument(
        "--aggregate-from-wandb",
        action="store_true",
        help="Skip training; download W&B runs by --experiment-tag and write aggregate figures.",
    )
    p.add_argument("--config", type=Path, default=None, help="YAML overriding defaults (default: package config/analysis/distribution_quick_5seeds.yaml).")
    p.add_argument("--figures-dir", type=Path, default=None, help="Root for run output (default: <repo>/figures/distribution_analysis).")
    p.add_argument("--seed", type=int, default=None)
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
    p.add_argument("--num-ctd-atoms", type=int)
    p.add_argument("--num-mog-components", type=int)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
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
        for k in ("metrics_pdf", "metrics_png", "panels_pdf", "panels_png", "metrics_csv"):
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
        "total_steps": args.total_steps,
        "log_every": args.log_every,
        "parseval_every": args.parseval_every,
        "dataset_size": args.dataset_size,
        "omega_laplace_scale": args.omega_laplace_scale,
        "num_tau": args.num_qtd_quantiles,
        "num_atoms": args.num_ctd_atoms,
        "num_mog_components": args.num_mog_components,
    }
    cfg = load_da_config_yaml(args.config, overrides)
    cfg = replace(cfg, artifacts_dir=run_dir)

    use_wandb = not args.no_wandb
    tag = os.environ.get("WANDB_EXPERIMENT_TAG", args.experiment_tag)
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
        train_and_export(cfg, tag, use_wandb, figures_run_dir=run_dir, skip_plots=args.no_plots)
    finally:
        if use_wandb:
            import wandb

            if wandb.run:
                wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
