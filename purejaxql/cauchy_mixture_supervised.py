"""
Supervised Cauchy-mixture IQN vs characteristic-function network (local toy experiment).

Ground truth: equal mixture of Cauchy(mu=-5, gamma=0.5) and Cauchy(mu=5, gamma=0.5).

IQN: cosine tau embedding -> MLP trunk -> scalar. CQN: scalar t -> Linear(embed_dim)+ReLU
-> same trunk -> 2D head on the unit disk. Each training step uses one shared reward batch R.

Eval uses a **fixed** RNG seed for IQN quantile levels tau so empirical CF curves across
checkpoints reflect learning, not independent Monte Carlo noise.

Run: python -m purejaxql.cauchy_mixture_supervised

Replot checkpoints only (no training), same PDFs + PNGs + metrics CSV::

    python -m purejaxql.cauchy_mixture_supervised --replot purejaxql/cauchy_mixture_run_TAG.npz

Outputs under figures/: vector PDF + 300 dpi PNG per figure; metrics CSV alongside.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, replace
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training import train_state

import flax.linen as nn

# --- Publication plotting: system sans-serif stack (SF on macOS via -apple-system), vector PDF ---
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

# GT nearly black; CQN saturated purple. Tiered lw (gap widened ~10%) keeps overlays readable.
COLORS = {
    "truth": "#0A0A0A",
    "cqn": "#7C3AED",
    "iqn": "#56B4E9",
}
LW_GT, LW_CQN, LW_IQN = 2.15, 1.41, 1.05  # GT > purple > blue (alignment overlay)


def minimal_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", color="#E0E0E0", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save_figure_pdf_png(fig: plt.Figure, pdf_path: str, *, png_dpi: int = 300) -> None:
    """Save the same layout as PDF and as a raster PNG (`pdf_path` may end with .pdf or any name)."""
    stem = os.path.splitext(pdf_path)[0]
    # Extra padding avoids tight-bbox clipping (e.g. first-column title ``1k``, y-axis labels).
    fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{stem}.png", format="png", bbox_inches="tight", pad_inches=0.08, dpi=png_dpi)


def _legend_below_panels(
    fig: plt.Figure,
    axes: list,
    ncol: int,
    *,
    y_anchor: float = 0.12,
    label_order: tuple[str, ...] | None = None,
) -> None:
    """Legend centered below subplot area (main title via fig.suptitle at top)."""
    handles, labels = axes[-1].get_legend_handles_labels()
    if label_order is not None:
        by_label = dict(zip(labels, handles))
        handles = [by_label[L] for L in label_order if L in by_label]
        labels = [L for L in label_order if L in by_label]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y_anchor),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize=8,
        labelcolor="#2E2E2E",
    )


# --- Mixture parameters (ground truth) ---
MIX_PROB = jnp.float32(0.5)
MU1 = jnp.float32(-5.0)
GAMMA1 = jnp.float32(0.5)
MU2 = jnp.float32(5.0)
GAMMA2 = jnp.float32(0.5)

# Filename suffix for figures / npz (|mu|=5, gamma=0.5)
FIG_FILE_TAG = "_5-mu_05-gamma_finale"


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


_CKPT_SCALAR_KEYS = ("mse_iqn", "mse_cf", "ks_iqn", "ks_cf", "w1_iqn", "w1_cf")
_CKPT_ARRAY_KEYS = ("t_eval", "x_eval", "phi_true", "phi_iqn", "phi_cf", "f_true", "f_iqn", "f_cf")


def checkpoints_dict_from_npz(npz_path: str) -> tuple[dict[int, dict], list[int]]:
    """Rebuild checkpoint dicts produced by ``main()``'s ``np.savez`` (``ck{i}_*`` keys)."""
    z = np.load(npz_path, allow_pickle=False)
    if "steps" not in z.files:
        raise ValueError(f"No 'steps' array in {npz_path}")
    ordered_steps = [int(x) for x in np.asarray(z["steps"]).flatten()]
    checkpoints_data: dict[int, dict] = {}
    for i, s in enumerate(ordered_steps):
        pre = f"ck{i}_"
        d: dict = {}
        for k in _CKPT_SCALAR_KEYS:
            d[k] = float(np.asarray(z[pre + k]))
        for k in _CKPT_ARRAY_KEYS:
            d[k] = np.asarray(z[pre + k])
        checkpoints_data[s] = d
    return checkpoints_data, ordered_steps


def figure_tag_from_npz_filename(npz_path: str) -> str | None:
    """Parse ``.../cauchy_mixture_run{tag}.npz`` basename → ``tag`` (includes leading ``_`` if present)."""
    base = os.path.basename(npz_path)
    prefix, suffix = "cauchy_mixture_run", ".npz"
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix) : -len(suffix)]
    return None


def _checkpoint_col_label(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def _fmt_metric(v: float) -> str:
    """Compact numeric cell (3 significant digits)."""
    return f"{float(v):.3g}"


def plot_frequency_figure(
    checkpoint_results: list[dict],
    steps: list[int],
    out_path: str,
    plot_t_max: float,
) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(3.35 * len(steps), 2.6), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    # z-order: black (GT) back, purple (CQN) mid, blue (IQN) front
    z_gt, z_cqn, z_iqn = 1, 2, 3
    lg_gt = r"GT ($\mathrm{Re}\,\varphi$)"
    lg_cqn, lg_iqn = "CQN", "IQN (empirical)"
    legend_order = (lg_gt, lg_cqn, lg_iqn)

    for ax, step, res in zip(axes, steps, checkpoint_results):
        t = res["t_eval"]
        m = np.abs(t) <= plot_t_max
        ax.plot(
            t[m],
            np.real(res["phi_true"])[m],
            color=COLORS["truth"],
            lw=LW_GT,
            label=lg_gt,
            zorder=z_gt,
        )
        ax.plot(
            t[m],
            np.real(res["phi_cf"])[m],
            color=COLORS["cqn"],
            lw=LW_CQN,
            label=lg_cqn,
            zorder=z_cqn,
        )
        ax.plot(
            t[m],
            np.real(res["phi_iqn"])[m],
            color=COLORS["iqn"],
            lw=LW_IQN,
            alpha=0.95,
            label=lg_iqn,
            zorder=z_iqn,
        )
        minimal_style(ax)
        ax.set_title(_checkpoint_col_label(step), fontsize=9, color="#6E6E6E")
        # SP-style: abscissa is the CF “frequency” parameter (ω; also written W in some signal texts)
        ax.set_xlabel(r"$\omega$ ($W$)")
        ax.set_xlim(-plot_t_max, plot_t_max)
    axes[0].set_ylabel(r"$\mathrm{Re}\,\varphi(\omega)$")
    fig.suptitle("Frequency Domain", fontsize=11, y=0.97, color="#2E2E2E")
    _legend_below_panels(fig, axes, ncol=3, y_anchor=0.13, label_order=legend_order)
    fig.subplots_adjust(bottom=0.27, top=0.83, left=0.11, right=0.98, wspace=0.12)
    _save_figure_pdf_png(fig, out_path)
    plt.close(fig)


def plot_spatial_figure(
    checkpoint_results: list[dict],
    steps: list[int],
    out_path: str,
    plot_x_min: float,
    plot_x_max: float,
) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(3.35 * len(steps), 2.75), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    z_gt, z_cqn, z_iqn = 1, 2, 3
    lg_gt, lg_cqn, lg_iqn = "GT", "CQN (Gil–Pelaez)", "IQN (empirical)"
    legend_order = (lg_gt, lg_cqn, lg_iqn)

    for ax, step, res in zip(axes, steps, checkpoint_results):
        x = res["x_eval"]
        m = (x >= plot_x_min) & (x <= plot_x_max)
        ax.plot(x[m], res["f_true"][m], color=COLORS["truth"], lw=LW_GT, label=lg_gt, zorder=z_gt)
        ax.plot(
            x[m],
            res["f_cf"][m],
            color=COLORS["cqn"],
            lw=LW_CQN,
            label=lg_cqn,
            zorder=z_cqn,
        )
        ax.plot(
            x[m],
            res["f_iqn"][m],
            color=COLORS["iqn"],
            lw=LW_IQN,
            alpha=0.95,
            label=lg_iqn,
            zorder=z_iqn,
        )
        minimal_style(ax)
        ax.set_title(_checkpoint_col_label(step), fontsize=9, color="#6E6E6E")
        ax.set_xlabel(r"$x$")
        ax.set_xlim(plot_x_min, plot_x_max)
        ax.text(
            0.03,
            0.97,
            f"KS  IQN {res['ks_iqn']:.3g}   CQN {res['ks_cf']:.3g}\n"
            f"W₁  IQN {res['w1_iqn']:.3g}   CQN {res['w1_cf']:.3g}",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#DDDDDD", linewidth=0.6),
        )
    axes[0].set_ylabel(r"$F(x)$")
    fig.suptitle("Spatial Domain", fontsize=11, y=0.97, color="#2E2E2E")
    _legend_below_panels(fig, axes, ncol=3, y_anchor=0.13, label_order=legend_order)
    fig.subplots_adjust(bottom=0.27, top=0.82, left=0.11, right=0.98, wspace=0.12)
    _save_figure_pdf_png(fig, out_path)
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.hist(
        np.asarray(samp),
        bins=72,
        range=(x_min, x_max),
        density=True,
        color="#E8E8E8",
        edgecolor="#CCCCCC",
        linewidth=0.2,
        label=r"Samples ($n{=}80\mathrm{k}$)",
        zorder=1,
    )
    ax.plot(np.asarray(x), np.asarray(y), color=COLORS["truth"], lw=LW_GT, label=r"Truth PDF", zorder=3)
    minimal_style(ax)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    
    fig.suptitle("Cauchy Distribution", fontsize=11, y=0.96, color="#2E2E2E")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor="#2E2E2E",
    )
    fig.subplots_adjust(bottom=0.22, top=0.86, left=0.12, right=0.96)
    _save_figure_pdf_png(fig, out_path)
    plt.close(fig)


def plot_metrics_table_figure(
    checkpoints_data: dict[int, dict],
    ordered_steps: list[int],
    out_path: str,
) -> None:
    """Compact metrics table PDF (numbers also in companion CSV)."""
    col_labels = [_checkpoint_col_label(s) for s in ordered_steps]
    row_labels = [
        "IQN  φ MSE",
        "CQN  φ MSE",
        "IQN  KS",
        "CQN  KS",
        "IQN  W₁",
        "CQN  W₁",
    ]
    keys = ("mse_iqn", "mse_cf", "ks_iqn", "ks_cf", "w1_iqn", "w1_cf")
    cell_text = [
        [_fmt_metric(checkpoints_data[s][k]) for s in ordered_steps] for k in keys
    ]

    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    ax.axis("off")
    fig.subplots_adjust(top=0.85, bottom=0.06)
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
    tbl.set_fontsize(9)
    tbl.scale(1.02, 1.38)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col < 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.6)
    fig.suptitle("Metrics", fontsize=11, y=0.96, color="#2E2E2E")
    _save_figure_pdf_png(fig, out_path)
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


def export_plots_and_metrics(
    cfg: TrainConfig,
    checkpoints_data: dict[int, dict],
    ordered_steps: list[int],
) -> None:
    """Write frequency/spatial/PDF figures (PDF + PNG each), metrics table, and CSV."""
    os.makedirs(cfg.figures_dir, exist_ok=True)
    results_list = [checkpoints_data[s] for s in ordered_steps]
    tag = cfg.figure_tag
    plot_frequency_figure(
        results_list,
        ordered_steps,
        os.path.join(cfg.figures_dir, f"cauchy_mixture_frequency{tag}.pdf"),
        cfg.plot_t_max,
    )
    plot_spatial_figure(
        results_list,
        ordered_steps,
        os.path.join(cfg.figures_dir, f"cauchy_mixture_spatial{tag}.pdf"),
        cfg.plot_x_min,
        cfg.plot_x_max,
    )
    metrics_pdf = os.path.join(cfg.figures_dir, f"cauchy_mixture_metrics{tag}.pdf")
    metrics_csv = os.path.join(cfg.figures_dir, f"cauchy_mixture_metrics{tag}.csv")
    plot_metrics_table_figure(checkpoints_data, ordered_steps, metrics_pdf)
    save_metrics_csv(checkpoints_data, ordered_steps, metrics_csv)
    print(f"Saved {metrics_csv}")

    rng_pdf = jax.random.PRNGKey(cfg.seed + 9_001)
    plot_mixture_pdf_figure(
        os.path.join(cfg.figures_dir, f"cauchy_mixture_pdf{tag}.pdf"),
        rng_pdf,
        cfg.plot_x_min,
        cfg.plot_x_max,
    )


def replot_from_npz(
    npz_path: str,
    cfg: TrainConfig | None = None,
    *,
    infer_figure_tag: bool = True,
) -> None:
    """Regenerate figures from a saved ``np.savez`` archive (no training)."""
    checkpoints_data, ordered_steps = checkpoints_dict_from_npz(npz_path)
    c = cfg or TrainConfig()
    if infer_figure_tag:
        inferred = figure_tag_from_npz_filename(npz_path)
        if inferred is not None:
            c = replace(c, figure_tag=inferred)
    export_plots_and_metrics(c, checkpoints_data, ordered_steps)
    print(f"Replotted from {npz_path} → {c.figures_dir}/ (PDF + PNG + metrics CSV)")


def _cli_train_config(args: argparse.Namespace) -> TrainConfig:
    c = TrainConfig()
    if args.figures_dir:
        c = replace(c, figures_dir=args.figures_dir)
    if args.figure_tag is not None:
        c = replace(c, figure_tag=args.figure_tag)
    return c


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
    export_plots_and_metrics(cfg, checkpoints_data, ordered_steps)

    save_dict = {"steps": np.array(ordered_steps)}
    for i, s in enumerate(ordered_steps):
        pre = f"ck{i}_"
        for k, v in checkpoints_data[s].items():
            save_dict[pre + k] = v
    npz_out = cfg.npz_filename()
    np.savez(npz_out, **save_dict)
    print(f"Saved {npz_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cauchy-mixture IQN vs CQN supervised toy.")
    parser.add_argument(
        "--replot",
        default=None,
        metavar="NPZ",
        help="Load arrays from an np.savez checkpoint and regenerate figures + CSV (no JAX training).",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Output directory for figures (default: TrainConfig.figures_dir).",
    )
    parser.add_argument(
        "--figure-tag",
        default=None,
        help="Suffix in filenames, e.g. _5-mu_05-gamma_v3. With --replot, omit to infer from NPZ name.",
    )
    cli = parser.parse_args()

    if cli.replot:
        cfg = _cli_train_config(cli)
        replot_from_npz(cli.replot, cfg, infer_figure_tag=cli.figure_tag is None)
    else:
        main(_cli_train_config(cli))
