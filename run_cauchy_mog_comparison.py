#!/usr/bin/env python3
"""Compare CF-loss fits across MoG / MoCauchy / MoGamma on a 2-step synthetic MDP."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import jax
from jax import config as jax_config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

jax_config.update("jax_enable_x64", True)


_HERE = Path(__file__).resolve().parent
_FIG_ROOT = _HERE / "figures" / "cauchy_mog_comparison"


@dataclass(frozen=True)
class Config:
    seed: int = 0
    gamma: float = 0.9
    reward_std: float = 0.2
    num_components: int = 51
    train_steps: int = 3000
    eval_every: int = 100
    lr: float = 7.5e-3
    t_max: float = 25.0
    num_t: int = 256
    x_min: float = -16.0
    x_max: float = 20.0
    num_x: int = 1200
    truth_samples: int = 300_000
    panel_samples: int = 12_000
    artifacts_dir: Path | None = None


def _default_config_yaml_path() -> Path:
    return _HERE / "purejaxql" / "config" / "analysis" / "cauchy_mog_comparison.yaml"


def load_config_yaml(path: Path | None, overrides: dict) -> Config:
    """Load Config from YAML (OmegaConf), then apply non-None CLI overrides."""
    from omegaconf import OmegaConf

    base = Config()
    known = {f.name for f in fields(Config)} - {"artifacts_dir"}
    merged: dict = {}
    p = path or _default_config_yaml_path()
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


def _config_for_logging(cfg: Config) -> dict:
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


def _softplus(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    return jax.nn.softplus(x) + eps


def _weights(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


def normal_cdf(z: jax.Array) -> jax.Array:
    return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))


def empirical_cdf(sorted_samples: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.mean(sorted_samples[:, None] <= x_grid[None, :], axis=0)


def w1_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid(jnp.abs(f_hat - f_true), x_grid)


def cramer_l2_sq_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid((f_hat - f_true) ** 2, x_grid)


def cf_l2_sq_over_omega2(phi_hat: jax.Array, phi_true: jax.Array, omega_grid: jax.Array, eps: float = 1e-4) -> jax.Array:
    w = 1.0 / (omega_grid**2 + eps)
    return jnp.mean(w * jnp.abs(phi_hat - phi_true) ** 2)


def sample_gaussian_mixture(key: jax.Array, n: int, weights: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    key_i, key_e = jax.random.split(key)
    idx = jax.random.categorical(key_i, jnp.log(weights), shape=(n,))
    eps = jax.random.normal(key_e, (n,))
    return mu[idx] + sigma[idx] * eps


def sample_cauchy_mixture(key: jax.Array, n: int, weights: jax.Array, loc: jax.Array, scale: jax.Array) -> jax.Array:
    key_i, key_u = jax.random.split(key)
    idx = jax.random.categorical(key_i, jnp.log(weights), shape=(n,))
    u = jax.random.uniform(key_u, (n,), minval=1e-6, maxval=1.0 - 1e-6)
    z = jnp.tan(jnp.pi * (u - 0.5))
    return loc[idx] + scale[idx] * z


def sample_gamma_mixture(
    key: jax.Array,
    n: int,
    weights: jax.Array,
    shape: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    key_i, key_g = jax.random.split(key)
    idx = jax.random.categorical(key_i, jnp.log(weights), shape=(n,))
    k_comp = shape[idx]
    th_comp = scale[idx]
    g = jax.random.gamma(key_g, k_comp, shape=(n,))
    return th_comp * g


def phi_gaussian_mixture(t: jax.Array, weights: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    expo = 1j * mu[:, None] * t[None, :] - 0.5 * (sigma[:, None] ** 2) * (t[None, :] ** 2)
    return jnp.sum(weights[:, None] * jnp.exp(expo), axis=0)


def cdf_gaussian_mixture(x: jax.Array, weights: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    z = (x[None, :] - mu[:, None]) / sigma[:, None]
    return jnp.sum(weights[:, None] * normal_cdf(z), axis=0)


def pdf_gaussian_mixture(x: jax.Array, weights: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    z = (x[None, :] - mu[:, None]) / sigma[:, None]
    p = jnp.exp(-0.5 * z**2) / (sigma[:, None] * jnp.sqrt(2.0 * jnp.pi))
    return jnp.sum(weights[:, None] * p, axis=0)


def phi_cauchy_mixture(t: jax.Array, weights: jax.Array, loc: jax.Array, scale: jax.Array) -> jax.Array:
    expo = 1j * loc[:, None] * t[None, :] - scale[:, None] * jnp.abs(t[None, :])
    return jnp.sum(weights[:, None] * jnp.exp(expo), axis=0)


def cdf_cauchy_mixture(x: jax.Array, weights: jax.Array, loc: jax.Array, scale: jax.Array) -> jax.Array:
    vals = jnp.arctan((x[None, :] - loc[:, None]) / scale[:, None]) / jnp.pi + 0.5
    return jnp.sum(weights[:, None] * vals, axis=0)


def pdf_cauchy_mixture(x: jax.Array, weights: jax.Array, loc: jax.Array, scale: jax.Array) -> jax.Array:
    z = (x[None, :] - loc[:, None]) / scale[:, None]
    p = 1.0 / (jnp.pi * scale[:, None] * (1.0 + z**2))
    return jnp.sum(weights[:, None] * p, axis=0)


def phi_gamma_mixture(
    t: jax.Array,
    weights: jax.Array,
    shape: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    z = (1.0 - 1j * scale[:, None] * t[None, :]).astype(jnp.complex128)
    comp = jnp.power(z, -shape[:, None])
    return jnp.sum(weights[:, None] * comp, axis=0)


def cdf_gamma_mixture(
    x: jax.Array,
    weights: jax.Array,
    shape: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    rat = jnp.maximum(x[None, :], 0.0) / jnp.maximum(scale[:, None], jnp.finfo(jnp.float64).tiny)
    vals = jnp.where(x[None, :] > 0, jax.scipy.special.gammainc(shape[:, None], rat), 0.0)
    return jnp.sum(weights[:, None] * vals, axis=0)


def pdf_gamma_mixture(
    x: jax.Array,
    weights: jax.Array,
    shape: jax.Array,
    scale: jax.Array,
) -> jax.Array:
    xp = jnp.maximum(x[None, :], jnp.finfo(jnp.float64).tiny)
    logp = (
        (shape[:, None] - 1.0) * jnp.log(xp)
        - xp / scale[:, None]
        - jax.scipy.special.gammaln(shape[:, None])
        - shape[:, None] * jnp.log(scale[:, None])
    )
    vals = jnp.where(x[None, :] > 0, jnp.exp(logp), 0.0)
    return jnp.sum(weights[:, None] * vals, axis=0)


TARGETS = {
    "gaussian_action": {
        "family": "mog",
        "weights": jnp.array([0.58, 0.42], dtype=jnp.float64),
        "mu": jnp.array([-2.4, 2.2], dtype=jnp.float64),
        "sigma": jnp.array([0.55, 1.1], dtype=jnp.float64),
        "plot_range": (-8.0, 8.0),
    },
    "cauchy_action": {
        "family": "moc",
        "weights": jnp.array([0.47, 0.53], dtype=jnp.float64),
        "loc": jnp.array([-3.5, 2.8], dtype=jnp.float64),
        "scale": jnp.array([0.38, 1.1], dtype=jnp.float64),
        "plot_range": (-15.0, 15.0),
    },
    "gamma_action": {
        "family": "mogamma",
        "weights": jnp.array([0.36, 0.64], dtype=jnp.float64),
        "shape": jnp.array([0.75, 2.2], dtype=jnp.float64),
        "scale": jnp.array([0.55, 2.1], dtype=jnp.float64),
        "plot_range": (-1.0, 22.0),
    },
}

MODEL_ORDER = ("mog", "moc", "mogamma")
MODEL_NAME = {"mog": "MoG", "moc": "MoCauchy", "mogamma": "MoGamma"}
MODEL_COLOR = {"truth": "#222222", "mog": "#4ecb8d", "moc": "#ff8c00", "mogamma": "#f1c40f"}
MODEL_STYLE = {
    "truth": {"lw": 3.2, "ls": "-", "alpha": 0.95, "zorder": 1},
    "mog": {"lw": 1.55, "ls": "-", "alpha": 0.95, "zorder": 2},
    "moc": {"lw": 1.45, "ls": "-", "alpha": 0.95, "zorder": 3},
    "mogamma": {"lw": 1.5, "ls": "-", "alpha": 0.95, "zorder": 4},
}
METRIC_ORDER = ("w1", "cramer_l2", "cf_l2_w")
METRIC_LABELS = {
    "w1": r"Wasserstein-1 $\downarrow$",
    "cramer_l2": r"Cramér-$L_2^2$ $\downarrow$",
    "cf_l2_w": r"CF-$\ell_2^2/\omega^2$ $\downarrow$",
}


def target_phi_return(target: dict, t: jax.Array, gamma: float, reward_std: float) -> jax.Array:
    tg = gamma * t
    if target["family"] == "mog":
        phi_next = phi_gaussian_mixture(tg, target["weights"], target["mu"], target["sigma"])
    elif target["family"] == "moc":
        phi_next = phi_cauchy_mixture(tg, target["weights"], target["loc"], target["scale"])
    else:
        phi_next = phi_gamma_mixture(tg, target["weights"], target["shape"], target["scale"])
    phi_reward = jnp.exp(-0.5 * (reward_std**2) * (t**2))
    return phi_reward * phi_next


def sample_target_return(key: jax.Array, target: dict, n: int, gamma: float, reward_std: float) -> jax.Array:
    key_x, key_r = jax.random.split(key)
    if target["family"] == "mog":
        x1 = sample_gaussian_mixture(key_x, n, target["weights"], target["mu"], target["sigma"])
    elif target["family"] == "moc":
        x1 = sample_cauchy_mixture(key_x, n, target["weights"], target["loc"], target["scale"])
    else:
        x1 = sample_gamma_mixture(key_x, n, target["weights"], target["shape"], target["scale"])
    r0 = reward_std * jax.random.normal(key_r, (n,))
    return r0 + gamma * x1


def init_params(family: str, k: int, key: jax.Array) -> dict[str, jax.Array]:
    k1, k2, k3 = jax.random.split(key, 3)
    params: dict[str, jax.Array] = {
        "logits": jax.random.normal(k1, (k,)) * 0.1,
    }
    if family == "mog":
        params["mu"] = jax.random.normal(k2, (k,)) * 2.0
        params["log_sigma"] = -0.2 + jax.random.normal(k3, (k,)) * 0.15
    elif family == "moc":
        params["loc"] = jax.random.normal(k2, (k,)) * 2.5
        params["log_scale"] = -0.5 + jax.random.normal(k3, (k,)) * 0.2
    else:
        k4, k5 = jax.random.split(k3)
        params["log_shape"] = jnp.log(2.0 + jax.random.uniform(k4, (k,)))
        params["log_scale"] = jnp.log(0.8 + 0.4 * jax.random.uniform(k5, (k,)))
    return params


def model_phi_return(family: str, params: dict[str, jax.Array], t: jax.Array) -> jax.Array:
    w = _weights(params["logits"])
    if family == "mog":
        return phi_gaussian_mixture(t, w, params["mu"], _softplus(params["log_sigma"]))
    if family == "moc":
        return phi_cauchy_mixture(t, w, params["loc"], _softplus(params["log_scale"]))
    return phi_gamma_mixture(
        t,
        w,
        _softplus(params["log_shape"]),
        _softplus(params["log_scale"]),
    )


def model_cdf(family: str, params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    w = _weights(params["logits"])
    if family == "mog":
        return cdf_gaussian_mixture(x, w, params["mu"], _softplus(params["log_sigma"]))
    if family == "moc":
        return cdf_cauchy_mixture(x, w, params["loc"], _softplus(params["log_scale"]))
    return cdf_gamma_mixture(
        x,
        w,
        _softplus(params["log_shape"]),
        _softplus(params["log_scale"]),
    )


def model_pdf(family: str, params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    w = _weights(params["logits"])
    if family == "mog":
        return pdf_gaussian_mixture(x, w, params["mu"], _softplus(params["log_sigma"]))
    if family == "moc":
        return pdf_cauchy_mixture(x, w, params["loc"], _softplus(params["log_scale"]))
    return pdf_gamma_mixture(
        x,
        w,
        _softplus(params["log_shape"]),
        _softplus(params["log_scale"]),
    )


def train_one_model(
    family: str,
    target_phi: jax.Array,
    t_grid: jax.Array,
    cfg: Config,
    key: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, np.ndarray]]:
    params = init_params(family, cfg.num_components, key)
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)
    eps = 1e-3

    @jax.jit
    def step(p, s):
        def loss_fn(pp):
            ph = model_phi_return(family, pp, t_grid)
            return jnp.mean(jnp.abs(ph - target_phi) ** 2 / (t_grid**2 + eps))

        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s_new = tx.update(grads, s, p)
        p_new = optax.apply_updates(p, updates)
        return p_new, s_new, loss

    hist_steps, hist_loss = [], []
    for st in range(1, cfg.train_steps + 1):
        params, opt_state, loss = step(params, opt_state)
        if st % cfg.eval_every == 0 or st == 1 or st == cfg.train_steps:
            hist_steps.append(st)
            hist_loss.append(float(loss))
    return params, {"step": np.asarray(hist_steps), "cf_loss": np.asarray(hist_loss)}


def make_panels_and_table(
    out_dir: Path,
    cfg: Config,
    x_grid: jax.Array,
    t_grid: jax.Array,
    truth_data: dict,
    fit_data: dict,
) -> None:
    def _smooth_curve_np(y: np.ndarray, passes: int = 2) -> np.ndarray:
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

    rows = list(TARGETS.keys())
    cols = list(MODEL_ORDER)

    csv_path = out_dir / "cauchy_mog_comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target_action", "model_family", "w1", "cramer_l2", "cf_l2_w"])
        for r in rows:
            for c in cols:
                m = fit_data[r][c]["metrics"]
                w.writerow([r, c, float(m["w1"]), float(m["cramer_l2"]), float(m["cf_l2_w"])])

    fig_tbl, axes_tbl = plt.subplots(len(rows), 1, figsize=(8.4, 2.4 * len(rows) + 0.8))
    if len(rows) == 1:
        axes_tbl = [axes_tbl]
    fig_tbl.subplots_adjust(top=0.92, bottom=0.06, hspace=0.38)
    for ax, action_name in zip(axes_tbl, rows):
        ax.axis("off")
        cell_text = []
        values = np.zeros((len(METRIC_ORDER), len(cols)), dtype=np.float64)
        for mi, metric_key in enumerate(METRIC_ORDER):
            row_text = []
            for ci, model_key in enumerate(cols):
                val = float(fit_data[action_name][model_key]["metrics"][metric_key])
                values[mi, ci] = val
                row_text.append(f"{val:.4g}")
            cell_text.append(row_text)
        best_idx = np.argmin(values, axis=1)
        table = ax.table(
            cellText=cell_text,
            rowLabels=[METRIC_LABELS[m] for m in METRIC_ORDER],
            colLabels=[MODEL_NAME[c] for c in cols],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.06, 1.45)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#CCCCCC")
            if r == 0 or c < 0:
                cell.set_text_props(fontweight="bold")
            if r >= 1 and c >= 0 and best_idx[r - 1] == c:
                cell.set_text_props(fontweight="bold")
        ax.text(
            0.5,
            1.03,
            action_name.replace("_", " ").title(),
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    fig_tbl.suptitle("Per-action metrics (rows) across model families (columns)")
    fig_tbl.savefig(out_dir / "cauchy_mog_comparison_metrics.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_tbl)

    fig, axes = plt.subplots(len(rows), 4, figsize=(14.2, 2.8 * len(rows)))
    x_np = np.asarray(x_grid)
    t_np = np.asarray(t_grid)

    for i, r in enumerate(rows):
        rng = TARGETS[r]["plot_range"]
        mask = (x_np >= rng[0]) & (x_np <= rng[1])
        ax_hist, ax_phi, ax_pdf, ax_cdf = axes[i]
        top_model = {
            "gaussian_action": "mog",
            "cauchy_action": "moc",
            "gamma_action": "mogamma",
        }[r]
        draw_order = [c for c in cols if c != top_model] + [top_model]
        st_truth = MODEL_STYLE["truth"]
        ax_hist.plot(
            x_np[mask],
            _smooth_curve_np(np.asarray(truth_data[r]["pdf"]))[mask],
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        ax_hist.set_title("Density" if i == 0 else "")
        ax_hist.set_xlim(*rng)

        ax_phi.plot(
            t_np,
            np.real(np.asarray(truth_data[r]["phi"])),
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        for c in draw_order:
            st = MODEL_STYLE[c]
            ax_phi.plot(
                t_np,
                np.real(np.asarray(fit_data[r][c]["phi"])),
                color=MODEL_COLOR[c],
                lw=st["lw"],
                ls=st["ls"],
                alpha=st["alpha"],
                zorder=6 if c == top_model else st["zorder"],
                label=MODEL_NAME[c],
            )
        ax_phi.set_title("Re $\\varphi$" if i == 0 else "")

        ax_pdf.plot(
            x_np[mask],
            _smooth_curve_np(np.asarray(truth_data[r]["pdf"]))[mask],
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        for c in draw_order:
            st = MODEL_STYLE[c]
            ax_pdf.plot(
                x_np[mask],
                _smooth_curve_np(np.asarray(fit_data[r][c]["pdf"]))[mask],
                color=MODEL_COLOR[c],
                lw=st["lw"],
                ls=st["ls"],
                alpha=st["alpha"],
                zorder=st["zorder"],
                label=MODEL_NAME[c],
            )
        ax_pdf.set_xlim(*rng)
        ax_pdf.set_title("PDF" if i == 0 else "")

        ax_cdf.plot(
            x_np[mask],
            np.asarray(truth_data[r]["cdf"])[mask],
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        for c in draw_order:
            st = MODEL_STYLE[c]
            ax_cdf.plot(
                x_np[mask],
                np.asarray(fit_data[r][c]["cdf"])[mask],
                color=MODEL_COLOR[c],
                lw=st["lw"],
                ls=st["ls"],
                alpha=st["alpha"],
                zorder=st["zorder"],
                label=MODEL_NAME[c],
            )
        ax_cdf.set_xlim(*rng)
        ax_cdf.set_ylim(-0.02, 1.02)
        ax_cdf.set_title("CDF" if i == 0 else "")

        for ax in (ax_hist, ax_phi, ax_pdf, ax_cdf):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle="--", color="#e1e1e1", alpha=0.7)
            ax.set_axisbelow(True)
        ax_hist.set_ylabel(r.replace("_", "\n").title(), fontsize=9, fontweight="bold")

    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)
    #fig.suptitle("2-step MDP return fitting with CF loss: family-matched models win", y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.09, wspace=0.28, hspace=0.35)
    fig.savefig(out_dir / "cauchy_mog_comparison_panels.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # Extra compact panels: Density + Re phi only (2x4 and 4x2).
    fig_2x4, axes_2x4 = plt.subplots(2, len(rows), figsize=(12.8, 5.1))
    for col_i, r in enumerate(rows):
        rng = TARGETS[r]["plot_range"]
        mask = (x_np >= rng[0]) & (x_np <= rng[1])
        ax_density = axes_2x4[0, col_i]
        ax_phi = axes_2x4[1, col_i]
        top_model = {
            "gaussian_action": "mog",
            "cauchy_action": "moc",
            "gamma_action": "mogamma",
        }[r]
        draw_order = [c for c in cols if c != top_model] + [top_model]
        st_truth = MODEL_STYLE["truth"]

        ax_density.plot(
            x_np[mask],
            _smooth_curve_np(np.asarray(truth_data[r]["pdf"]))[mask],
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        ax_density.set_title(r.replace("_action", "").replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax_density.set_xlim(*rng)
        ax_density.spines["top"].set_visible(False)
        ax_density.spines["right"].set_visible(False)
        ax_density.grid(True, linestyle="--", color="#e1e1e1", alpha=0.7)
        ax_density.set_axisbelow(True)

        ax_phi.plot(
            t_np,
            np.real(np.asarray(truth_data[r]["phi"])),
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        for c in draw_order:
            st = MODEL_STYLE[c]
            ax_phi.plot(
                t_np,
                np.real(np.asarray(fit_data[r][c]["phi"])),
                color=MODEL_COLOR[c],
                lw=st["lw"],
                ls=st["ls"],
                alpha=st["alpha"],
                zorder=6 if c == top_model else st["zorder"],
                label=MODEL_NAME[c],
            )
        ax_phi.set_xlim(t_np.min(), t_np.max())
        ax_phi.spines["top"].set_visible(False)
        ax_phi.spines["right"].set_visible(False)
        ax_phi.grid(True, linestyle="--", color="#e1e1e1", alpha=0.7)
        ax_phi.set_axisbelow(True)

    axes_2x4[0, 0].set_ylabel("Density", fontweight="bold")
    axes_2x4[1, 0].set_ylabel(r"Re $\varphi$", fontweight="bold")
    axes_2x4[1, 0].legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
    fig_2x4.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.10, wspace=0.22, hspace=0.23)
    fig_2x4.savefig(out_dir / "cauchy_mog_comparison_panels_density_rephi_2x4.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig_2x4)

    fig_4x2, axes_4x2 = plt.subplots(len(rows), 2, figsize=(7.5, 8.4))
    axes_4x2[0, 0].set_title("Density", fontweight="bold")
    axes_4x2[0, 1].set_title(r"Re $\varphi$", fontweight="bold")
    for row_i, r in enumerate(rows):
        rng = TARGETS[r]["plot_range"]
        mask = (x_np >= rng[0]) & (x_np <= rng[1])
        ax_density, ax_phi = axes_4x2[row_i]
        top_model = {
            "gaussian_action": "mog",
            "cauchy_action": "moc",
            "gamma_action": "mogamma",
        }[r]
        draw_order = [c for c in cols if c != top_model] + [top_model]
        st_truth = MODEL_STYLE["truth"]

        ax_density.plot(
            x_np[mask],
            _smooth_curve_np(np.asarray(truth_data[r]["pdf"]))[mask],
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        ax_density.set_xlim(*rng)
        ax_density.spines["top"].set_visible(False)
        ax_density.spines["right"].set_visible(False)
        ax_density.grid(True, linestyle="--", color="#e1e1e1", alpha=0.7)
        ax_density.set_axisbelow(True)

        ax_phi.plot(
            t_np,
            np.real(np.asarray(truth_data[r]["phi"])),
            color=MODEL_COLOR["truth"],
            lw=st_truth["lw"],
            ls=st_truth["ls"],
            alpha=st_truth["alpha"],
            zorder=st_truth["zorder"],
            label="Truth",
        )
        for c in draw_order:
            st = MODEL_STYLE[c]
            ax_phi.plot(
                t_np,
                np.real(np.asarray(fit_data[r][c]["phi"])),
                color=MODEL_COLOR[c],
                lw=st["lw"],
                ls=st["ls"],
                alpha=st["alpha"],
                zorder=6 if c == top_model else st["zorder"],
                label=MODEL_NAME[c],
            )
        ax_phi.set_xlim(t_np.min(), t_np.max())
        ax_phi.spines["top"].set_visible(False)
        ax_phi.spines["right"].set_visible(False)
        ax_phi.grid(True, linestyle="--", color="#e1e1e1", alpha=0.7)
        ax_phi.set_axisbelow(True)
        ax_density.text(
            -0.34,
            0.5,
            r.replace("_action", "").replace("_", " ").title(),
            transform=ax_density.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )

    axes_4x2[0, 1].legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
    fig_4x2.subplots_adjust(left=0.15, right=0.99, top=0.96, bottom=0.08, wspace=0.18, hspace=0.28)
    fig_4x2.savefig(out_dir / "cauchy_mog_comparison_panels_density_rephi_4x2.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig_4x2)

    fig_curve, axes_curve = plt.subplots(1, len(rows), figsize=(4.4 * len(rows), 3.3))
    if len(rows) == 1:
        axes_curve = [axes_curve]
    for i, r in enumerate(rows):
        ax = axes_curve[i]
        for c in cols:
            h = fit_data[r][c]["history"]
            st = MODEL_STYLE[c]
            ax.plot(
                h["step"],
                h["cf_loss"],
                color=MODEL_COLOR[c],
                lw=max(1.4, st["lw"] - 0.2),
                ls=st["ls"],
                alpha=st["alpha"],
                zorder=st["zorder"],
                label=MODEL_NAME[c],
            )
        ax.set_yscale("log")
        ax.set_title(r.replace("_", " ").title())
        ax.set_xlabel("Train step")
        if i == 0:
            ax.set_ylabel("CF weighted loss")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", color="#e1e1e1", alpha=0.7)
        ax.set_axisbelow(True)
    axes_curve[-1].legend(frameon=False, loc="upper right")
    fig_curve.suptitle("CF training curves")
    fig_curve.tight_layout()
    fig_curve.savefig(out_dir / "cauchy_mog_comparison_training_curves.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig_curve)


def run_experiment(cfg: Config, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t_grid = jnp.linspace(-cfg.t_max, cfg.t_max, cfg.num_t, dtype=jnp.float64)
    x_grid = jnp.linspace(cfg.x_min, cfg.x_max, cfg.num_x, dtype=jnp.float64)
    key = jax.random.PRNGKey(cfg.seed)

    truth_data: dict = {}
    fit_data: dict = {k: {} for k in TARGETS}

    for target_name, target in TARGETS.items():
        key, k_samp, k_panel = jax.random.split(key, 3)
        samples_truth = sample_target_return(k_samp, target, cfg.truth_samples, cfg.gamma, cfg.reward_std)
        sorted_truth = jnp.sort(samples_truth)
        cdf_truth = empirical_cdf(sorted_truth, x_grid)
        phi_truth = target_phi_return(target, t_grid, cfg.gamma, cfg.reward_std)
        # Numerical derivative of empirical CDF for a smooth-enough truth PDF view.
        dx = x_grid[1] - x_grid[0]
        pdf_truth = jnp.gradient(cdf_truth, dx)
        samples_panel = sample_target_return(k_panel, target, cfg.panel_samples, cfg.gamma, cfg.reward_std)
        truth_data[target_name] = {
            "samples_panel": samples_panel,
            "cdf": cdf_truth,
            "phi": phi_truth,
            "pdf": jnp.maximum(pdf_truth, 0.0),
        }

        for model_name in MODEL_ORDER:
            key, k_model = jax.random.split(key)
            params, history = train_one_model(model_name, phi_truth, t_grid, cfg, k_model)
            phi_hat = model_phi_return(model_name, params, t_grid)
            cdf_hat = model_cdf(model_name, params, x_grid)
            pdf_hat = model_pdf(model_name, params, x_grid)
            metrics = {
                "w1": w1_cdf(cdf_hat, cdf_truth, x_grid),
                "cramer_l2": cramer_l2_sq_cdf(cdf_hat, cdf_truth, x_grid),
                "cf_l2_w": cf_l2_sq_over_omega2(phi_hat, phi_truth, t_grid),
            }
            fit_data[target_name][model_name] = {
                "params": params,
                "history": history,
                "phi": phi_hat,
                "cdf": cdf_hat,
                "pdf": pdf_hat,
                "metrics": metrics,
            }
            print(
                f"[{target_name:15s}] {MODEL_NAME[model_name]:9s} "
                f"W1={float(metrics['w1']):.4f}  "
                f"Cramer={float(metrics['cramer_l2']):.4f}  "
                f"CF-L2={float(metrics['cf_l2_w']):.4f}",
                flush=True,
            )

    make_panels_and_table(output_dir, cfg, x_grid, t_grid, truth_data, fit_data)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="2-step MDP comparison: MoG vs MoCauchy vs MoGamma under CF loss.")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML overriding defaults (default: purejaxql/config/analysis/cauchy_mog_comparison.yaml).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--num-components", type=int, default=None, help="Mixture components (fixed to 51 for fair comparison).")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--reward-std", type=float, default=None)
    p.add_argument("--t-max", type=float, default=None)
    p.add_argument("--num-t", type=int, default=None)
    p.add_argument("--x-min", type=float, default=None)
    p.add_argument("--x-max", type=float, default=None)
    p.add_argument("--num-x", type=int, default=None)
    p.add_argument("--truth-samples", type=int, default=None)
    p.add_argument("--panel-samples", type=int, default=None)
    p.add_argument("--figures-dir", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "seed": args.seed,
        "train_steps": args.train_steps,
        "eval_every": args.eval_every,
        "num_components": args.num_components,
        "lr": args.lr,
        "gamma": args.gamma,
        "reward_std": args.reward_std,
        "t_max": args.t_max,
        "num_t": args.num_t,
        "x_min": args.x_min,
        "x_max": args.x_max,
        "num_x": args.num_x,
        "truth_samples": args.truth_samples,
        "panel_samples": args.panel_samples,
    }
    cfg = load_config_yaml(args.config, overrides)
    root = args.figures_dir if args.figures_dir is not None else _FIG_ROOT
    run_dir = root / dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cfg = replace(cfg, artifacts_dir=run_dir)
    print(f"Artifacts will be written to: {run_dir}", flush=True)
    _write_yaml(run_dir / "cauchy_mog_comparison_experiment_config.yaml", _config_for_logging(cfg))
    run_experiment(cfg, run_dir)


if __name__ == "__main__":
    main()
