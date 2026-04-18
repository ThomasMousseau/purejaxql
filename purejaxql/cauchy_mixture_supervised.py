"""
Supervised Cauchy-mixture IQN vs characteristic-function network (local toy experiment).

Ground truth: equal mixture of Cauchy(mu=-5, gamma=0.5) and Cauchy(mu=5, gamma=0.5).

IQN: cosine tau embedding -> MLP trunk -> scalar. CF net: scalar t -> Linear(embed_dim)+ReLU
-> same trunk -> 2D head on the unit disk. Each training step uses one shared reward batch R.

Eval uses a **fixed** RNG seed for IQN quantile levels tau so empirical CF curves across
checkpoints reflect learning, not independent Monte Carlo noise.

Run: python -m purejaxql.cauchy_mixture_supervised
Outputs under figures/ with suffix from TrainConfig.figure_tag; metrics also saved as CSV.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training import train_state

import flax.linen as nn


# --- Mixture parameters (ground truth) ---
MIX_PROB = jnp.float32(0.5)
MU1 = jnp.float32(-5.0)
GAMMA1 = jnp.float32(0.5)
MU2 = jnp.float32(5.0)
GAMMA2 = jnp.float32(0.5)

# Filename suffix for figures / npz (|mu|=5, gamma=2.0)
FIG_FILE_TAG = "_5-mu_05-gamma_v3"


@dataclass(frozen=True)
class TrainConfig:
    
    seed: int = 0
    total_steps: int = 100_000
    checkpoint_steps: tuple[int, ...] = (1_000, 10_000, 100_000)
    log_every: int = 500

    batch_size: int = 16_384
    num_tau: int = 128
    num_cf_freq: int = 128
    t_min: float = -25.0
    t_max: float = 25.0

    embed_dim: int = 64
    hidden_dim: int = 256
    trunk_depth: int = 3
    huber_kappa: float = 1.0

    lr_iqn: float = 3e-4
    lr_cf: float = 3e-4

    # Eval IQN empirical CF: fixed τ draws so curves are comparable across checkpoints (see module docstring)
    eval_iqn_seed: int = 42

    # Metrics / Gil–Pelaez on a wide x-grid (accurate KS / W1 integral)
    eval_t_max: float = 50.0
    eval_num_t: int = 401
    eval_x_min: float = -50.0
    eval_x_max: float = 50.0
    eval_num_x: int = 801
    eval_num_tau_iqn: int = 50_000

    # Zoom for **plots** only (tails omitted for clarity; Cauchy has heavy tails)
    plot_t_max: float = 25.0
    plot_x_min: float = -14.0
    plot_x_max: float = 14.0
    gil_pelaez_t_delta: float = 1e-4
    gil_pelaez_t_max: float = 50.0
    gil_pelaez_num_t: int = 8001

    figures_dir: str = "figures"
    figure_tag: str = FIG_FILE_TAG

    def npz_filename(self) -> str:
        return f"purejaxql/cauchy_mixture_run{self.figure_tag}.npz"


def sample_cauchy_mixture(rng: jax.Array, n: int, dtype=jnp.float32) -> jax.Array:
    """Bernoulli mixture of two Cauchys; inverse-CDF sampling."""
    rng_a, rng_b = jax.random.split(rng)
    pick = jax.random.bernoulli(rng_a, MIX_PROB, (n,))
    u = jax.random.uniform(rng_b, (n,), dtype=dtype)
    x = jnp.tan(jnp.pi * (u - 0.5))
    r1 = MU1 + GAMMA1 * x
    r2 = MU2 + GAMMA2 * x
    return jnp.where(pick, r1, r2)


def phi_true(t: jax.Array, dtype=jnp.complex128) -> jax.Array:
    """Closed-form mixture CF; t any shape."""
    t = jnp.asarray(t, dtype=dtype)
    abs_t = jnp.abs(t)
    p1 = MIX_PROB * jnp.exp(1j * t * MU1 - GAMMA1 * abs_t)
    p2 = (1.0 - MIX_PROB) * jnp.exp(1j * t * MU2 - GAMMA2 * abs_t)
    return p1 + p2


def cdf_true(x: jax.Array, dtype=jnp.float64) -> jax.Array:
    """Closed-form mixture CDF."""
    x = jnp.asarray(x, dtype=dtype)
    f1 = (jnp.arctan((x - MU1) / GAMMA1) / jnp.pi + 0.5)
    f2 = (jnp.arctan((x - MU2) / GAMMA2) / jnp.pi + 0.5)
    return MIX_PROB * f1 + (1.0 - MIX_PROB) * f2


def pdf_true(x: jax.Array, dtype=jnp.float64) -> jax.Array:
    """Closed-form mixture density (sum of Cauchy PDFs)."""
    x = jnp.asarray(x, dtype=dtype)
    p1 = 1.0 / (jnp.pi * GAMMA1 * (1.0 + ((x - MU1) / GAMMA1) ** 2))
    p2 = 1.0 / (jnp.pi * GAMMA2 * (1.0 + ((x - MU2) / GAMMA2) ** 2))
    return MIX_PROB * p1 + (1.0 - MIX_PROB) * p2


def _huber(u: jax.Array, kappa: float) -> jax.Array:
    abs_u = jnp.abs(u)
    quad = 0.5 * u**2
    lin = kappa * (abs_u - 0.5 * kappa)
    return jnp.where(abs_u <= kappa, quad, lin)


def quantile_huber_loss(
    q_preds: jax.Array,
    rewards: jax.Array,
    taus: jax.Array,
    kappa: float,
) -> jax.Array:
    """Pairwise quantile Huber loss (IQN-style); q_preds (M,), rewards (N,), taus (M,)."""
    u = rewards[None, :] - q_preds[:, None]
    hub = _huber(u, kappa)
    ind = (u < 0).astype(jnp.float32)
    rho = jnp.abs(taus[:, None] - ind) * hub / kappa
    return jnp.mean(rho)


class MLPTrunk(nn.Module):
    hidden_dim: int
    depth: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for i in range(self.depth):
            x = nn.Dense(self.hidden_dim, name=f"trunk_{i}")(x)
            x = nn.relu(x)
        return x


class IQN1D(nn.Module):
    """Cosine tau embedding -> shared trunk -> scalar quantile."""

    embed_dim: int
    hidden_dim: int
    trunk_depth: int = 3

    @nn.compact
    def __call__(self, tau: jax.Array) -> jax.Array:
        tau = jnp.clip(tau, 1e-6, 1.0 - 1e-6)
        idx = jnp.arange(1, self.embed_dim + 1, dtype=tau.dtype)
        emb = jnp.cos(jnp.pi * tau[:, None] * idx[None, :])
        h = MLPTrunk(hidden_dim=self.hidden_dim, depth=self.trunk_depth)(emb)
        q = nn.Dense(1, name="head")(h).squeeze(-1)
        return q


class CF1D(nn.Module):
    """Scalar t -> Dense(embed_dim) + ReLU -> same trunk -> 2D -> unit disk."""

    embed_dim: int
    hidden_dim: int
    trunk_depth: int = 3

    @nn.compact
    def __call__(self, t: jax.Array) -> jax.Array:
        x = jnp.asarray(t, jnp.float32)[..., None]
        x = nn.Dense(self.embed_dim, name="t_proj")(x)
        x = nn.relu(x)
        h = MLPTrunk(hidden_dim=self.hidden_dim, depth=self.trunk_depth)(x)
        raw = nn.Dense(2, name="head")(h)
        a, b = raw[..., 0], raw[..., 1]
        z_r = jnp.tanh(a)
        z_i = jnp.tanh(b)
        mag = jnp.sqrt(z_r**2 + z_i**2) + 1e-8
        scale = jnp.minimum(1.0, 1.0 / mag)
        return (z_r + 1j * z_i) * scale


def create_train_states(
    rng: jax.Array,
    cfg: TrainConfig,
) -> tuple[train_state.TrainState, train_state.TrainState, IQN1D, CF1D]:
    rng_i, rng_c = jax.random.split(rng)
    iqn = IQN1D(cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth)
    cf = CF1D(cfg.embed_dim, cfg.hidden_dim, cfg.trunk_depth)
    tau0 = jnp.ones((2,), jnp.float32) * 0.5
    t0 = jnp.zeros((2,), jnp.float32)
    viqn = iqn.init(rng_i, tau0)
    vcf = cf.init(rng_c, t0)
    tx_i = optax.adam(cfg.lr_iqn)
    tx_c = optax.adam(cfg.lr_cf)
    si = train_state.TrainState.create(
        apply_fn=iqn.apply,
        params=viqn["params"],
        tx=tx_i,
    )
    sc = train_state.TrainState.create(
        apply_fn=cf.apply,
        params=vcf["params"],
        tx=tx_c,
    )
    return si, sc, iqn, cf


def make_train_step(cfg: TrainConfig, iqn: IQN1D, cf: CF1D):
    """JIT train step sharing one reward batch."""

    def step(
        state_i: train_state.TrainState,
        state_c: train_state.TrainState,
        rng: jax.Array,
        rewards: jax.Array,
    ):
        rng, k1, k2 = jax.random.split(rng, 3)
        taus = jax.random.uniform(k1, (cfg.num_tau,), dtype=rewards.dtype)
        ts = jax.random.uniform(k2, (cfg.num_cf_freq,), dtype=rewards.dtype)
        ts = ts * (cfg.t_max - cfg.t_min) + cfg.t_min

        def loss_iqn(params):
            q = iqn.apply({"params": params}, taus)
            return quantile_huber_loss(q, rewards, taus, cfg.huber_kappa)

        def loss_cf(params):
            phi_p = cf.apply({"params": params}, ts)
            phi_e = jnp.mean(
                jnp.exp(1j * ts[:, None] * rewards[None, :]),
                axis=1,
            )
            d = phi_p - phi_e
            return 0.5 * jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)

        li, gi = jax.value_and_grad(loss_iqn)(state_i.params)
        lc, gc = jax.value_and_grad(loss_cf)(state_c.params)
        state_i = state_i.apply_gradients(grads=gi)
        state_c = state_c.apply_gradients(grads=gc)
        return state_i, state_c, li, lc

    return jax.jit(step)


def empirical_cf_from_samples(samples: jax.Array, t_grid: jax.Array) -> jax.Array:
    """Empirical CF (complex64/128) at frequencies t_grid from 1D samples."""
    # samples (M,), t (T,)
    return jnp.mean(
        jnp.exp(1j * t_grid[:, None] * samples[None, :]),
        axis=1,
    )


def empirical_cdf(samples: jax.Array, x_grid: jax.Array) -> jax.Array:
    """Right-continuous empirical CDF at grid points."""
    s = jnp.sort(samples)
    n = samples.shape[0]
    idx = jnp.searchsorted(s, x_grid, side="right")
    return idx.astype(jnp.float64) / jnp.maximum(n, 1)


def gil_pelaez_cdf(
    phi_positive: jax.Array,
    t_pos: jax.Array,
    x_grid: jax.Array,
) -> jax.Array:
    """
    F(x) = 1/2 - (1/pi) * integral_delta^inf Im(exp(-i t x) phi(t)) / t dt
    phi_positive: phi(t) on t_pos (t > 0), complex128
    t_pos: strictly increasing, first point ~ delta > 0
    """
    #TODO: check if this is correct
    phi_positive = phi_positive - 1j * jnp.imag(phi_positive[0])
    
    x = x_grid[:, None]  # (X, 1)
    t = t_pos[None, :]  # (1, T)
    integrand = jnp.imag(jnp.exp(-1j * t * x) * phi_positive[None, :]) / t
    integral = jnp.trapezoid(integrand, t, axis=1)
    return 0.5 - integral / jnp.pi


def cf_mse_vs_truth(phi_hat: jax.Array, phi_t: jax.Array) -> jax.Array:
    d = phi_hat - phi_t
    return jnp.mean(jnp.real(d) ** 2 + jnp.imag(d) ** 2)


def ks_on_grid(f_hat: jax.Array, f_true: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(f_hat - f_true))


def wasserstein_1_cdf_l1(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    """
    1-Wasserstein distance between distributions on R with CDFs F and G:
      W1 = ∫ |F(x) − G(x)| dx
    Trapezoidal rule on x_grid (approximation; extend grid wide for negligible tail bias).
    """
    return jnp.trapezoid(jnp.abs(f_hat - f_true), x_grid)


def make_eval_checkpoint_jit(cfg: TrainConfig, iqn: IQN1D, cf: CF1D):
    """JIT-compiled eval; closure holds module structure."""

    @jax.jit
    def eval_jit(params_i, params_c):
        dtype_r = jnp.float64
        t_eval = jnp.linspace(-cfg.eval_t_max, cfg.eval_t_max, cfg.eval_num_t, dtype=dtype_r)
        x_eval = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, dtype=dtype_r)
        phi_t = phi_true(t_eval, dtype=jnp.complex128)

        # Fixed τ grid across checkpoints → empirical CF differences reflect learning, not fresh MC noise
        taus = jax.random.uniform(
            jax.random.PRNGKey(cfg.eval_iqn_seed),
            (cfg.eval_num_tau_iqn,),
            dtype=jnp.float32,
        )
        q_samples = iqn.apply({"params": params_i}, taus).astype(dtype_r)

        phi_iqn = empirical_cf_from_samples(
            q_samples.astype(jnp.complex128),
            t_eval.astype(jnp.complex128),
        )

        phi_cf = cf.apply({"params": params_c}, t_eval.astype(jnp.float32)).astype(jnp.complex128)

        f_true = cdf_true(x_eval)
        f_iqn = empirical_cdf(q_samples, x_eval)

        t_gp = jnp.linspace(
            cfg.gil_pelaez_t_delta,
            cfg.gil_pelaez_t_max,
            cfg.gil_pelaez_num_t,
            dtype=dtype_r,
        )
        phi_gp = cf.apply({"params": params_c}, t_gp.astype(jnp.float32)).astype(jnp.complex128)
        f_cf = gil_pelaez_cdf(phi_gp, t_gp, x_eval)

        mse_iqn = cf_mse_vs_truth(phi_iqn, phi_t)
        mse_cf = cf_mse_vs_truth(phi_cf, phi_t)
        ks_iqn = ks_on_grid(f_iqn, f_true)
        ks_cf = ks_on_grid(f_cf, f_true)
        w1_iqn = wasserstein_1_cdf_l1(f_iqn, f_true, x_eval)
        w1_cf = wasserstein_1_cdf_l1(f_cf, f_true, x_eval)
        return (
            t_eval,
            x_eval,
            phi_t,
            phi_iqn,
            phi_cf,
            f_true,
            f_iqn,
            f_cf,
            mse_iqn,
            mse_cf,
            ks_iqn,
            ks_cf,
            w1_iqn,
            w1_cf,
        )

    return eval_jit


def evaluate_checkpoint_to_dict(
    eval_jit,
    state_i: train_state.TrainState,
    state_c: train_state.TrainState,
) -> dict:
    out = eval_jit(state_i.params, state_c.params)
    (
        t_eval,
        x_eval,
        phi_t,
        phi_iqn,
        phi_cf,
        f_true,
        f_iqn,
        f_cf,
        mse_iqn,
        mse_cf,
        ks_iqn,
        ks_cf,
        w1_iqn,
        w1_cf,
    ) = out
    return {
        "t_eval": np.asarray(t_eval),
        "x_eval": np.asarray(x_eval),
        "phi_true": np.asarray(phi_t),
        "phi_iqn": np.asarray(phi_iqn),
        "phi_cf": np.asarray(phi_cf),
        "f_true": np.asarray(f_true),
        "f_iqn": np.asarray(f_iqn),
        "f_cf": np.asarray(f_cf),
        "mse_iqn": float(mse_iqn),
        "mse_cf": float(mse_cf),
        "ks_iqn": float(ks_iqn),
        "ks_cf": float(ks_cf),
        "w1_iqn": float(w1_iqn),
        "w1_cf": float(w1_cf),
    }


def plot_frequency_figure(
    checkpoint_results: list[dict],
    steps: list[int],
    out_path: str,
    plot_t_max: float,
) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 3.8), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step, res in zip(axes, steps, checkpoint_results):
        t = res["t_eval"]
        m = np.abs(t) <= plot_t_max
        ax.plot(t[m], np.real(res["phi_true"])[m], color="black", lw=2.0, label=r"$\mathrm{Re}\,\varphi_{\mathrm{true}}$")
        ax.plot(t[m], np.real(res["phi_cf"])[m], color="C0", lw=1.5, label="CF net")
        ax.plot(t[m], np.real(res["phi_iqn"])[m], color="C1", lw=1.5, alpha=0.9, label="IQN empirical")
        ax.set_title(f"{step:,} steps")
        ax.set_xlabel(r"$t$")
        ax.set_xlim(-plot_t_max, plot_t_max)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\varphi(t)$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        r"(IQN empirical CF uses the same $\tau$ grid at every checkpoint — curve changes reflect learning, not resampling noise)",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_figure(
    checkpoint_results: list[dict],
    steps: list[int],
    out_path: str,
    plot_x_min: float,
    plot_x_max: float,
) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4.0), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step, res in zip(axes, steps, checkpoint_results):
        x = res["x_eval"]
        m = (x >= plot_x_min) & (x <= plot_x_max)
        ax.plot(x[m], res["f_true"][m], color="black", lw=2.0, label=r"$F_{\mathrm{true}}$")
        ax.plot(x[m], res["f_cf"][m], color="C0", lw=1.5, label="CF net (Gil–Pelaez)")
        ax.plot(x[m], res["f_iqn"][m], color="C1", lw=1.5, alpha=0.9, label="IQN empirical")
        ax.set_title(f"{step:,} steps")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(plot_x_min, plot_x_max)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.03,
            0.97,
            f"KS  IQN {res['ks_iqn']:.3g}   CF {res['ks_cf']:.3g}\n"
            f"W₁  IQN {res['w1_iqn']:.3g}   CF {res['w1_cf']:.3g}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="0.8"),
        )
    axes[0].set_ylabel(r"$F(x)$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        r"$W_1=\int |F_{\mathrm{est}}-F_{\mathrm{true}}|\,dx$ on full eval grid (KS is pointwise max $\|F_{\mathrm{est}}-F_{\mathrm{true}}\|_\infty$)",
        fontsize=8,
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _checkpoint_col_label(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def _fmt_metric(v: float) -> str:
    """Compact numeric cell (3 significant digits)."""
    return f"{float(v):.3g}"


def plot_mixture_pdf_figure(
    out_path: str,
    rng: jax.Array,
    x_min: float,
    x_max: float,
) -> None:
    """True mixture PDF + histogram of samples (zoomed window; Cauchy has heavy tails)."""
    x = jnp.linspace(x_min, x_max, 1200)
    y = pdf_true(x)
    rng, rk = jax.random.split(rng)
    samp = sample_cauchy_mixture(rk, 80_000)
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    ax.hist(
        np.asarray(samp),
        bins=90,
        range=(x_min, x_max),
        density=True,
        color="0.88",
        edgecolor="0.75",
        linewidth=0.25,
        label="80k samples",
        zorder=1,
    )
    ax.plot(np.asarray(x), np.asarray(y), color="black", lw=2.2, label="Mixture PDF", zorder=2)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$x$ (reward)")
    ax.set_ylabel("density")
    ax.set_title(
        r"Ground truth: $\frac{1}{2}\mathrm{Cauchy}(-5,0.5)+\frac{1}{2}\mathrm{Cauchy}(5,0.5)$"
    )
    ax.text(
        0.02,
        0.98,
        f"zoom: [{x_min:.0f}, {x_max:.0f}] — tails omitted (heavy-tailed Cauchy)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        color="0.35",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table_figure(
    checkpoints_data: dict[int, dict],
    ordered_steps: list[int],
    out_path: str,
) -> None:
    """PNG summary table: rows = method/metric, cols = checkpoints."""
    col_labels = [_checkpoint_col_label(s) for s in ordered_steps]
    row_labels = [
        "IQN  φ MSE",
        "CF   φ MSE",
        "IQN  KS",
        "CF   KS",
        "IQN  W₁",
        "CF   W₁",
    ]
    keys = ("mse_iqn", "mse_cf", "ks_iqn", "ks_cf", "w1_iqn", "w1_cf")
    cell_text = [
        [_fmt_metric(checkpoints_data[s][k]) for s in ordered_steps] for k in keys
    ]

    fig, ax = plt.subplots(figsize=(6.5, 2.65))
    ax.axis("off")
    fig.subplots_adjust(top=0.88)
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
        rowLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.05, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col < 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("0.85")
    fig.suptitle("vs truth (lower is better) — same CSV on disk", fontsize=11, y=0.98)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(
    checkpoints_data: dict[int, dict],
    ordered_steps: list[int],
    out_path: str,
) -> None:
    """Machine-readable metrics (preferred over screenshot for numbers)."""
    fieldnames = [
        "step",
        "phi_mse_iqn",
        "phi_mse_cf",
        "ks_iqn",
        "ks_cf",
        "w1_iqn",
        "w1_cf",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in ordered_steps:
            r = checkpoints_data[s]
            w.writerow(
                {
                    "step": s,
                    "phi_mse_iqn": r["mse_iqn"],
                    "phi_mse_cf": r["mse_cf"],
                    "ks_iqn": r["ks_iqn"],
                    "ks_cf": r["ks_cf"],
                    "w1_iqn": r["w1_iqn"],
                    "w1_cf": r["w1_cf"],
                }
            )


def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    os.makedirs(cfg.figures_dir, exist_ok=True)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, rk = jax.random.split(rng)
    state_i, state_c, iqn, cf = create_train_states(rk, cfg)
    train_step = make_train_step(cfg, iqn, cf)
    eval_jit = make_eval_checkpoint_jit(cfg, iqn, cf)

    ck_set = set(cfg.checkpoint_steps)
    checkpoints_data: dict[int, dict] = {}

    t0 = time.perf_counter()
    step_times: list[float] = []

    for step in range(1, cfg.total_steps + 1):
        rng, rk = jax.random.split(rng)
        rewards = sample_cauchy_mixture(rk, cfg.batch_size)
        rng, tk = jax.random.split(rng)

        t_step = time.perf_counter()
        state_i, state_c, li, lc = train_step(state_i, state_c, tk, rewards)
        step_times.append(time.perf_counter() - t_step)

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            mean_step = float(np.mean(step_times[-cfg.log_every :]))
            print(
                f"step={step:,} loss_iqn={float(li):.6f} loss_cf={float(lc):.6f} "
                f"sps={sps:.1f} step_ms={mean_step*1000:.2f}",
                flush=True,
            )

        if step in ck_set:
            checkpoints_data[step] = evaluate_checkpoint_to_dict(eval_jit, state_i, state_c)
            r = checkpoints_data[step]
            print(
                f"[checkpoint {step:,}] φ-MSE iqn={r['mse_iqn']:.6e} cf={r['mse_cf']:.6e} | "
                f"KS iqn={r['ks_iqn']:.6e} cf={r['ks_cf']:.6e} | "
                f"W1 iqn={r['w1_iqn']:.6e} cf={r['w1_cf']:.6e}",
                flush=True,
            )

    total_time = time.perf_counter() - t0
    print(f"Done {cfg.total_steps:,} steps in {total_time:.1f}s ({cfg.total_steps/total_time:.1f} steps/s).")

    ordered_steps = sorted(checkpoints_data.keys())
    results_list = [checkpoints_data[s] for s in ordered_steps]

    tag = cfg.figure_tag
    plot_frequency_figure(
        results_list,
        ordered_steps,
        os.path.join(cfg.figures_dir, f"cauchy_mixture_frequency{tag}.png"),
        cfg.plot_t_max,
    )
    plot_spatial_figure(
        results_list,
        ordered_steps,
        os.path.join(cfg.figures_dir, f"cauchy_mixture_spatial{tag}.png"),
        cfg.plot_x_min,
        cfg.plot_x_max,
    )
    metrics_png = os.path.join(cfg.figures_dir, f"cauchy_mixture_metrics{tag}.png")
    metrics_csv = os.path.join(cfg.figures_dir, f"cauchy_mixture_metrics{tag}.csv")
    plot_metrics_table_figure(checkpoints_data, ordered_steps, metrics_png)
    save_metrics_csv(checkpoints_data, ordered_steps, metrics_csv)
    print(f"Saved {metrics_csv}")

    rng_pdf = jax.random.PRNGKey(cfg.seed + 9_001)
    plot_mixture_pdf_figure(
        os.path.join(cfg.figures_dir, f"cauchy_mixture_pdf{tag}.png"),
        rng_pdf,
        cfg.plot_x_min,
        cfg.plot_x_max,
    )

    save_dict = {"steps": np.array(ordered_steps)}
    for i, s in enumerate(ordered_steps):
        pre = f"ck{i}_"
        for k, v in checkpoints_data[s].items():
            save_dict[pre + k] = v
    npz_out = cfg.npz_filename()
    np.savez(npz_out, **save_dict)
    print(f"Saved {npz_out}")


if __name__ == "__main__":
    main()
