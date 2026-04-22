#!/usr/bin/env python3
"""IQN, QR-DQN, C51, MoG-CQN offline bandit (self-contained). Dataset fold_in(rng,1001); trunk 2×Dense+ReLU; ω_max=15."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from numpy.polynomial.hermite import hermgauss

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
LN_MU_BASE, LN_MU_SCALE, LN_SIGMA = 0.0, 0.3, 0.6
_GH_X, _GH_W = hermgauss(64)
LN_GH_NODES = jnp.asarray(np.sqrt(2.0) * _GH_X, jnp.float64)
LN_GH_WEIGHTS = jnp.asarray(_GH_W / np.sqrt(np.pi), jnp.float64)


def _cauchy_d(x_state):
    return CAUCHY_PEAK_BASE + CAUCHY_PEAK_SCALE * jnp.abs(x_state[..., 1])


def _mog_sigma(x_state):
    return MOG_SIGMA_FLOOR + MOG_SIGMA_SCALE * jnp.abs(x_state[..., 2])


def _ln_mu(x_state):
    return LN_MU_BASE + LN_MU_SCALE * x_state[..., 3]


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


def _ln_sample(rng, x_state):
    eps = jax.random.normal(rng, x_state.shape[:-1], dtype=x_state.dtype)
    return jnp.exp(_ln_mu(x_state) + LN_SIGMA * eps)


def _ln_phi(t, x_state):
    mu = _ln_mu(x_state)
    y = jnp.exp(mu + LN_SIGMA * LN_GH_NODES)
    return jnp.sum(LN_GH_WEIGHTS[None, :] * jnp.exp(1j * t[:, None] * y[None, :]), axis=1)


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


def _ln_cdf(x_grid, x_state):
    mu = _ln_mu(x_state)
    sx = jnp.where(x_grid > 0, x_grid, 1.0)
    z = (jnp.log(sx) - mu) / (LN_SIGMA * jnp.sqrt(2.0))
    return jnp.where(x_grid > 0, 0.5 * (1.0 + jax.scipy.special.erf(z)), 0.0)


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


def _ln_pdf(x_grid, x_state):
    mu = _ln_mu(x_state)
    sx = jnp.where(x_grid > 0, x_grid, 1.0)
    return jnp.where(
        x_grid > 0,
        jnp.exp(-0.5 * ((jnp.log(sx) - mu) / LN_SIGMA) ** 2) / (sx * LN_SIGMA * jnp.sqrt(2.0 * jnp.pi)),
        0.0,
    )


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
    ActionDist("Log-Normal (skewed)", "LogN", _ln_sample, _ln_phi, _ln_cdf, _ln_pdf, -1.0, 8.0),
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
        
    else: # k == 3 (Log-Normal)
        mu_ln = _ln_mu(xs)
        sg = jnp.float64(LN_SIGMA)
        mn = jnp.exp(mu_ln + 0.5 * sg**2)
        vn = (jnp.exp(sg**2) - 1.0) * jnp.exp(2.0 * mu_ln + sg**2)
        mz, vz = g * mn, sr**2 + g**2 * vn
        # ER is infinity for Log-Normal, so return NaN
        return mz, vz, mz - lam * vz, jnp.float64(jnp.nan)


DATASET_SEED_OFFSET = 1001
PANEL_ANCHOR_PRNG_KEY = jax.random.PRNGKey(42_424_242)

DA_ORDER = ("iqn", "qr_dqn", "c51", "mog_cqn")
MK = ("phi_mse", "ks", "w1", "cvar_err", "er_err", "mean_err", "var_err", "mv_err")


@dataclass(frozen=True)
class DAConfig:
    seed: int = 0
    state_dim: int = STATE_DIM
    dataset_size: int = 500_000
    dataset_seed_offset: int = DATASET_SEED_OFFSET
    total_steps: int = 20_000 #100_000
    log_every: int = 20_000
    batch_size: int = 1024
    num_tau: int = 32
    num_atoms: int = 32 #
    num_omega: int = 32
    gamma: float = 0.95
    immediate_reward_std: float = 0.1
    omega_max: float = 15.0
    huber_kappa: float = 1.0
    lr: float = 1e-3
    max_grad_norm: float = 10.0
    omega_clip: float = 15.0
    omega_laplace_scale: float = 1.0
    sigma_min: float = 1e-6
    sigma_max: float = 20.0
    aux_mean_loss_weight: float = 0.1
    num_mog_components: int = 3
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

    def msgpack_filename(self) -> str:
        return str(_artifact_purejaxql_dir() / f"distribution_analysis_run_seed{self.seed}.msgpack")


def fixed_qr_tau_grid(n: int, dtype=jnp.float32):
    idx = jnp.arange(1, n + 1, dtype=dtype)
    return (idx - jnp.float32(0.5)) / jnp.float32(n)


def c51_atoms(cfg: DAConfig):
    return jnp.linspace(jnp.float32(cfg.c51_v_min), jnp.float32(cfg.c51_v_max), cfg.num_atoms, dtype=jnp.float32)


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


class C51Net(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_atoms: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_atoms, name="at")(h)


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
    qrn = IQNNet(cfg.embed_dim, cfg.hidden_dim, cfg.trunk_layers)
    c51 = C51Net(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    mog = MoGCQNNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_mog_components, cfg.sigma_min, cfg.sigma_max)
    x0, a0 = jnp.zeros((2, cfg.state_dim), jnp.float32), jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    tau0 = jnp.full((2, cfg.num_tau), 0.5, jnp.float32)
    tx = _opt(cfg.lr, cfg.max_grad_norm)
    vi, vq, v5, vm = iqn.init(ri, x0, a0, tau0), qrn.init(rq, x0, a0, tau0), c51.init(r5, x0, a0), mog.init(rm, x0, a0)
    return (
        train_state.TrainState.create(apply_fn=iqn.apply, params=vi["params"], tx=tx),
        train_state.TrainState.create(apply_fn=qrn.apply, params=vq["params"], tx=tx),
        train_state.TrainState.create(apply_fn=c51.apply, params=v5["params"], tx=tx),
        train_state.TrainState.create(apply_fn=mog.apply, params=vm["params"], tx=tx),
        iqn, qrn, c51, mog,
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


def make_train_chunk(cfg: DAConfig, iqn: IQNNet, qrn: IQNNet, c51: C51Net, mog: MoGCQNNet, ds):
    xa, aa, ra, zn = ds
    n, aux_w = xa.shape[0], jnp.float32(cfg.aux_mean_loss_weight)
    g, atoms = jnp.float32(cfg.gamma), c51_atoms(cfg)
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
            return quantile_huber_loss_pairwise(qrn.apply({"params": pp}, xb, oh, tauf), tgt, tauf, cfg.huber_kappa)

        def l5(pp):
            lg = jax.nn.log_softmax(c51.apply({"params": pp}, xb, oh))
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
            cf = 0.5 * jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)
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

    @jax.jit
    def run(si, sq, s5, sm, rng):
        (si, sq, s5, sm, rng), loss = jax.lax.scan(step, (si, sq, s5, sm, rng), None, cfg.log_every)
        return si, sq, s5, sm, rng, loss

    return run


def _make_eval(cfg: DAConfig, iqn: IQNNet, qrn: IQNNet, c51: C51Net, mog: MoGCQNNet):
    wm = cfg.omega_max
    te = jnp.linspace(-wm, wm, cfg.eval_num_t, jnp.float64)
    xe = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, jnp.float64)
    xr = jnp.linspace(cfg.risk_x_min, cfg.risk_x_max, cfg.risk_num_x, jnp.float64)
    tp = jnp.linspace(cfg.gp_t_delta, wm, cfg.gp_num_t, jnp.float64)
    g64, rs64 = jnp.float64(cfg.gamma), jnp.float64(cfg.immediate_reward_std)
    taue = jax.random.uniform(jax.random.PRNGKey(cfg.eval_iqn_seed), (cfg.num_taus_iqn_eval,), jnp.float32)
    atoms = c51_atoms(cfg)
    taqr = jnp.linspace(1 / (2 * cfg.num_taus_iqn_eval), 1 - 1 / (2 * cfg.num_taus_iqn_eval), cfg.num_taus_iqn_eval, jnp.float32)
    phis = [d.phi for d in ACTION_DISTS]

    def one(pi, pq, p5, pm, xs, k):
        oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1, a1 = xs[None, :].astype(jnp.float32), oh[None, :]
        qs = iqn.apply({"params": pi}, x1, a1, taue[None, :])[0].astype(jnp.float64)
        Fi, Fir = empirical_cdf(jnp.sort(qs), xe), empirical_cdf(jnp.sort(qs), xr)
        phii = empirical_cf_from_samples(qs.astype(jnp.complex128), te.astype(jnp.complex128))
        qq = qrn.apply({"params": pq}, x1, a1, taqr[None, :])[0].astype(jnp.float64)
        Fq, Fqr = empirical_cdf(jnp.sort(qq), xe), empirical_cdf(jnp.sort(qq), xr)
        phiq = empirical_cf_from_samples(qq.astype(jnp.complex128), te.astype(jnp.complex128))
        lc = c51.apply({"params": p5}, x1, a1)[0]
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
        Ft, Ftr = gil_pelaez_cdf(ptp, tp, xe), gil_pelaez_cdf(ptp, tp, xr)
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
        er_undefined = (k == 1) or (k == 3)
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

        return {
            "phi_mse_iqn": cf_mse_vs_truth(phii, pt),
            "phi_mse_qr_dqn": cf_mse_vs_truth(phiq, pt),
            "phi_mse_c51": cf_mse_vs_truth(phi5, pt),
            "phi_mse_mog_cqn": cf_mse_vs_truth(phim, pt),
            "ks_iqn": ks_on_grid(Fi, Ft),
            "ks_qr_dqn": ks_on_grid(Fq, Ft),
            "ks_c51": ks_on_grid(F5, Ft),
            "ks_mog_cqn": ks_on_grid(Fm, Ft),
            "w1_iqn": w1_cdf(Fi, Ft, xe),
            "w1_qr_dqn": w1_cdf(Fq, Ft, xe),
            "w1_c51": w1_cdf(F5, Ft, xe),
            "w1_mog_cqn": w1_cdf(Fm, Ft, xe),
            "cvar_err_iqn": jnp.abs(cvar_from_cdf(Fir, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_qr_dqn": jnp.abs(cvar_from_cdf(Fqr, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_c51": jnp.abs(cvar_from_cdf(F5r, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_mog_cqn": jnp.abs(cvar_from_cdf(Fmr, xr, cfg.cvar_alpha) - cv_t),
            "er_err_iqn": er_err_iqn,
            "er_err_qr_dqn": er_err_qr_dqn,
            "er_err_c51": er_err_c51,
            "er_err_mog_cqn": er_err_mog_cqn,
            "phi_iqn": phii, "phi_qr_dqn": phiq, "phi_c51": phi5, "phi_mog_cqn": phim, "phi_true": pt,
            "F_iqn": Fi, "F_qr_dqn": Fq, "F_c51": F5, "F_mog_cqn": Fm, "F_true": Ft,
            "pdf_true": pdf_t, "pdf_iqn": pdfi, "pdf_qr_dqn": pdfq, "pdf_c51": pdf5, "pdf_mog_cqn": pdfm,
            "mean_err_iqn": mei[0], "mean_err_qr_dqn": meq[0], "mean_err_c51": me5[0], "mean_err_mog_cqn": mem[0],
            "var_err_iqn": mei[1], "var_err_qr_dqn": meq[1], "var_err_c51": me5[1], "var_err_mog_cqn": mem[1],
            "mv_err_iqn": mei[2], "mv_err_qr_dqn": meq[2], "mv_err_c51": me5[2], "mv_err_mog_cqn": mem[2],
        }

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
    "phi_qr_dqn",
    "phi_c51",
    "phi_mog_cqn",
    "F_true",
    "F_iqn",
    "F_qr_dqn",
    "F_c51",
    "F_mog_cqn",
    "pdf_iqn",
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


def save_msgpack(cfg: DAConfig, *params):
    os.makedirs(os.path.dirname(cfg.msgpack_filename()), exist_ok=True)
    with open(cfg.msgpack_filename(), "wb") as f:
        f.write(flax.serialization.to_bytes(dict(zip(("iqn", "qr_dqn", "c51", "mog_cqn"), params))))


def train_and_export(cfg: DAConfig, tag: str, wb: bool):
    rng = jax.random.PRNGKey(cfg.seed)
    ri, rd, rt, ru = jax.random.split(rng, 4)
    print(f"Dataset... offset={cfg.dataset_seed_offset}", flush=True)
    ds = build_dataset(jax.random.fold_in(rd, cfg.dataset_seed_offset), cfg)
    jax.tree.map(lambda x: x.block_until_ready(), ds)
    si, sq, s5, sm, *nets = create_train_states(ri, cfg)
    run = make_train_chunk(cfg, nets[0], nets[1], nets[2], nets[3], ds)
    nc = cfg.total_steps // cfg.log_every
    t0 = time.perf_counter()
    rng = ru
    last_losses: dict[str, float] | None = None
    for c in range(nc):
        tc = time.perf_counter()
        si, sq, s5, sm, rng, ls = run(si, sq, s5, sm, rng)
        jax.tree.map(lambda x: x.block_until_ready(), ls)
        st = (c + 1) * cfg.log_every
        last_losses = {
            "train/loss_iqn_final": float(ls[0][-1]),
            "train/loss_qr_dqn_final": float(ls[1][-1]),
            "train/loss_c51_final": float(ls[2][-1]),
            "train/loss_mog_cqn_final": float(ls[3][-1]),
        }
        print(
            f"  {st:>7}  iqn={last_losses['train/loss_iqn_final']:.4f} "
            f"qr={last_losses['train/loss_qr_dqn_final']:.4f} "
            f"c51={last_losses['train/loss_c51_final']:.4f} "
            f"mog={last_losses['train/loss_mog_cqn_final']:.4f}  {time.perf_counter()-tc:.1f}s",
            flush=True,
        )
    print(f"train {time.perf_counter()-t0:.1f}s", flush=True)
    xt = jax.random.fold_in(rt, cfg.test_seed_offset)
    xv = jax.random.normal(xt, (cfg.n_test, cfg.state_dim), jnp.float32).at[0].set(
        jax.random.normal(PANEL_ANCHOR_PRNG_KEY, (cfg.state_dim,), jnp.float32)
    )
    er, te, xe = evaluate_test_set(cfg, tuple(nets), (si.params, sq.params, s5.params, sm.params), xv)
    jax.tree.map(lambda x: x.block_until_ready(), er)
    payload, pm = _collect_eval_payload(cfg, tag, er, te, xe, panel_index=0)
    save_msgpack(cfg, si.params, sq.params, s5.params, sm.params)
    if wb:
        import wandb

        if last_losses:
            wandb.summary.update(last_losses)
        _wandb_log_eval_metrics(pm, tag, step=cfg.total_steps)
        _save_eval_payload_to_wandb(payload)


def parse_args(a=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment-tag", default="DistAnalysis")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--dataset-size", type=int)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    return p.parse_args(a)


def main(a=None):
    args = parse_args(a)
    cfg = DAConfig(seed=args.seed)
    if args.total_steps:
        cfg = replace(cfg, total_steps=args.total_steps)
    if args.dataset_size:
        cfg = replace(cfg, dataset_size=args.dataset_size)
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
            config=asdict(cfg),
        )
    try:
        train_and_export(cfg, tag, use_wandb)
    finally:
        if use_wandb:
            import wandb

            if wandb.run:
                wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
