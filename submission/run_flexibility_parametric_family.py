#!/usr/bin/env python3
"""Offline contextual bandit benchmark parallel to ``run_distribution_analysis.py``.

Same dataset and Bellman CF targets as distribution_analysis; **only** the three parametric mixture
φTD heads from ``run_cauchy_mog_comparison.py`` (MoG / MoCauchy / MoGamma). CTD and QTD are **not**
trained or evaluated here (use ``run_distribution_analysis.py`` for those baselines).

Uses the same ``distribution.yaml`` defaults as distribution_analysis for comparable wall-clock.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import flax.linen as nn
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

from run_cauchy_mog_comparison import model_cdf, model_pdf, model_phi_return, _softplus
from run_distribution_analysis import (
    ACTION_DISTS,
    DAConfig,
    MK,
    NUM_ACTIONS,
    PANEL_ANCHOR_PRNG_KEY,
    _INVERSE_2PI,
    _config_for_logging,
    _huber,
    _dense_ln_relu,
    analytic_bellman_moments,
    build_dataset,
    cf_l2_sq_over_omega2,
    cf_mse_vs_truth,
    cramer_l2_sq_cdf,
    cvar_from_cdf,
    entropic_risk_from_cdf,
    gil_pelaez_cdf,
    ks_on_grid,
    load_da_config_yaml,
    sample_frequencies,
    w1_cdf,
    _write_yaml,
)

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parent


def _artifact_purejaxql_dir() -> Path:
    return _REPO_ROOT / "purejaxql"


# --- Three φTD mixture families only (no CTD/QTD in this experiment). ---
FP_ORDER = ("phi_mog", "phi_moc", "phi_mogamma")
LEGACY_ALGO_ALIASES_FP = {
    "phi_mog": "phitd_mog",
    "phi_moc": "phitd_cauchy",
    "phi_mogamma": "phitd_mogamma",
}


class PhiMoGNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        logits = nn.Dense(self.num_components, name="logits")(h)
        mu = nn.Dense(self.num_components, name="mu")(h)
        log_sigma = nn.Dense(self.num_components, name="log_sigma")(h)
        return logits, mu, log_sigma


class PhiMoCNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        logits = nn.Dense(self.num_components, name="logits")(h)
        loc = nn.Dense(self.num_components, name="loc")(h)
        log_scale = nn.Dense(self.num_components, name="log_scale")(h)
        return logits, loc, log_scale


class PhiMoGammaNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        logits = nn.Dense(self.num_components, name="logits")(h)
        log_shape = nn.Dense(self.num_components, name="log_shape")(h)
        log_scale = nn.Dense(self.num_components, name="log_scale")(h)
        return logits, log_shape, log_scale


def _opt(lr: float, clip: float):
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def _params_mog(logits, mu, log_sigma) -> dict[str, jax.Array]:
    return {"logits": logits, "mu": mu, "log_sigma": log_sigma}


def _params_moc(logits, loc, log_scale) -> dict[str, jax.Array]:
    return {"logits": logits, "loc": loc, "log_scale": log_scale}


def _params_mogamma(logits, log_shape, log_scale) -> dict[str, jax.Array]:
    return {"logits": logits, "log_shape": log_shape, "log_scale": log_scale}


def _mixture_mean_mog(logits: jax.Array, mu: jax.Array, log_sigma: jax.Array) -> jax.Array:
    w = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(w * mu, axis=-1)


def _mixture_mean_moc(logits: jax.Array, loc: jax.Array, log_scale: jax.Array) -> jax.Array:
    """MoCauchy has no finite mean; return NaN for compatibility with Cauchy truth moments."""
    return jnp.full(loc.shape[:-1], jnp.nan, dtype=loc.dtype)


def _mixture_mean_mogamma(logits: jax.Array, log_shape: jax.Array, log_scale: jax.Array) -> jax.Array:
    w = jax.nn.softmax(logits, axis=-1)
    k = _softplus(log_shape)
    th = _softplus(log_scale)
    comp_m = k * th
    return jnp.sum(w * comp_m, axis=-1)


def create_train_states_fp(rng: jax.Array, cfg: DAConfig):
    rmog, rmoc, rmg = jax.random.split(rng, 3)
    k = cfg.num_dirac_components
    phi_mog = PhiMoGNet(cfg.hidden_dim, cfg.trunk_layers, k)
    phi_moc = PhiMoCNet(cfg.hidden_dim, cfg.trunk_layers, k)
    phi_mogamma = PhiMoGammaNet(cfg.hidden_dim, cfg.trunk_layers, k)
    x0, a0 = jnp.zeros((2, cfg.state_dim), jnp.float32), jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    tx = _opt(cfg.lr, cfg.max_grad_norm)
    vm1, vm2, vm3 = (
        phi_mog.init(rmog, x0, a0),
        phi_moc.init(rmoc, x0, a0),
        phi_mogamma.init(rmg, x0, a0),
    )
    return (
        train_state.TrainState.create(apply_fn=phi_mog.apply, params=vm1["params"], tx=tx),
        train_state.TrainState.create(apply_fn=phi_moc.apply, params=vm2["params"], tx=tx),
        train_state.TrainState.create(apply_fn=phi_mogamma.apply, params=vm3["params"], tx=tx),
        phi_mog,
        phi_moc,
        phi_mogamma,
    )


def make_train_chunk_fp(
    cfg: DAConfig,
    phi_mog: PhiMoGNet,
    phi_moc: PhiMoCNet,
    phi_mogamma: PhiMoGammaNet,
    ds,
):
    xa, aa, ra, zn = ds
    n, aux_w = xa.shape[0], jnp.float32(cfg.aux_mean_loss_weight)
    g = jnp.float32(cfg.gamma)
    phis = [d.phi for d in ACTION_DISTS]

    def step(c, _):
        smog, smoc, smoga, rng = c
        rng, kb, ko = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        xb, ab, rb, znb = xa[idx], aa[idx], ra[idx], zn[idx]
        oh = jax.nn.one_hot(ab, NUM_ACTIONS, dtype=jnp.float32)
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

        gw = g * om
        fr = jnp.exp(1j * om.astype(jnp.complex64) * rb[:, None].astype(jnp.complex64))

        def ph1(ai, wi, xi):
            return jax.lax.switch(ai, phis, wi.astype(jnp.float64), xi.astype(jnp.float64))

        fz = jax.vmap(ph1)(ab, gw, xb).astype(jnp.complex64)
        target_cf = fr * fz
        if cfg.close_to_theory:

            def cf_phi_mean(residual: jnp.ndarray) -> jnp.ndarray:
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2) * _INVERSE_2PI

        else:
            w2 = jnp.maximum(jnp.square(om), jnp.float32(cfg.cf_loss_omega_eps**2))

            def cf_phi_mean(residual: jnp.ndarray) -> jnp.ndarray:
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2 / w2) * _INVERSE_2PI

        def lphimog(pp):
            lg, mu, ls = phi_mog.apply({"params": pp}, xb, oh)

            def row(lgi, mui, lsi, omi):
                return model_phi_return("mog", _params_mog(lgi, mui, lsi), omi)

            pred = jax.vmap(row)(lg, mu, ls, om)
            d = pred - target_cf
            mp = _mixture_mean_mog(lg, mu, ls)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        def lphimoc(pp):
            lg, loc, lsc = phi_moc.apply({"params": pp}, xb, oh)

            def row(lgi, loci, lsci, omi):
                return model_phi_return("moc", _params_moc(lgi, loci, lsci), omi)

            pred = jax.vmap(row)(lg, loc, lsc, om)
            d = pred - target_cf
            mp = _mixture_mean_moc(lg, loc, lsc)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        def lphimg(pp):
            lg, lsh, lsc = phi_mogamma.apply({"params": pp}, xb, oh)

            def row(lgi, lshi, lsci, omi):
                return model_phi_return("mogamma", _params_mogamma(lgi, lshi, lsci), omi)

            pred = jax.vmap(row)(lg, lsh, lsc, om)
            d = pred - target_cf
            mp = _mixture_mean_mogamma(lg, lsh, lsc)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        v1, g1 = jax.value_and_grad(lphimog)(smog.params)
        v2, g2 = jax.value_and_grad(lphimoc)(smoc.params)
        v3, g3 = jax.value_and_grad(lphimg)(smoga.params)
        return (
            smog.apply_gradients(grads=g1),
            smoc.apply_gradients(grads=g2),
            smoga.apply_gradients(grads=g3),
            rng,
        ), (v1, v2, v3)

    scan_len = int(cfg.parseval_every)

    @jax.jit
    def run(smog, smoc, smoga, rng):
        (smog, smoc, smoga, rng), loss = jax.lax.scan(step, (smog, smoc, smoga, rng), None, scan_len)
        return smog, smoc, smoga, rng, loss

    return run


def _make_eval_fp(
    cfg: DAConfig,
    phi_mog: PhiMoGNet,
    phi_moc: PhiMoCNet,
    phi_mogamma: PhiMoGammaNet,
):
    wm = cfg.omega_max
    te = jnp.linspace(-wm, wm, cfg.eval_num_t, jnp.float64)
    xe = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, jnp.float64)
    xr = jnp.linspace(cfg.risk_x_min, cfg.risk_x_max, cfg.risk_num_x, jnp.float64)
    tp = jnp.linspace(cfg.gp_t_delta, wm, cfg.gp_num_t, jnp.float64)
    g64, rs64 = jnp.float64(cfg.gamma), jnp.float64(cfg.immediate_reward_std)
    phis = [d.phi for d in ACTION_DISTS]
    te_c = te.astype(jnp.complex128)

    def one(pm1, pm2, pm3, xs, k):
        oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1, a1 = xs[None, :].astype(jnp.float32), oh[None, :]

        lg1, mu1, ls1 = phi_mog.apply({"params": pm1}, x1, a1)
        pm_mog = _params_mog(lg1[0], mu1[0], ls1[0])
        phimog = model_phi_return("mog", pm_mog, te_c)
        Fmog = model_cdf("mog", pm_mog, xe)
        Fmogr = model_cdf("mog", pm_mog, xr)
        pdfmog = model_pdf("mog", pm_mog, xe)

        lg2, loc2, lsc2 = phi_moc.apply({"params": pm2}, x1, a1)
        pm_moc = _params_moc(lg2[0], loc2[0], lsc2[0])
        phimoc = model_phi_return("moc", pm_moc, te_c)
        Fmoc = model_cdf("moc", pm_moc, xe)
        Fmocr = model_cdf("moc", pm_moc, xr)
        pdfmoc = model_pdf("moc", pm_moc, xe)

        lg3, lsh3, lsc3 = phi_mogamma.apply({"params": pm3}, x1, a1)
        pm_mg = _params_mogamma(lg3[0], lsh3[0], lsc3[0])
        phimg = model_phi_return("mogamma", pm_mg, te_c)
        Fmg = model_cdf("mogamma", pm_mg, xe)
        Fmgr = model_cdf("mogamma", pm_mg, xr)
        pdfmg = model_pdf("mogamma", pm_mg, xe)

        pr = jnp.exp(-0.5 * (rs64**2) * te**2)
        pz = jax.lax.switch(k, phis, g64 * te, xs.astype(jnp.float64))
        pt = pr * pz

        prp = jnp.exp(-0.5 * (rs64**2) * tp**2)
        pzp = jax.lax.switch(k, phis, g64 * tp, xs.astype(jnp.float64))
        ptp = prp * pzp
        Ft = gil_pelaez_cdf(ptp, tp, xe)
        Ftr = gil_pelaez_cdf(ptp, tp, xr)
        pdf_t = jnp.clip(jnp.gradient(Ft, xe), 0, None)

        mtn, vtn, mvtn, ertn = analytic_bellman_moments(
            k, xs, g64, rs64, jnp.float64(cfg.mv_lambda), jnp.float64(cfg.er_beta)
        )

        cv_t = cvar_from_cdf(Ftr, xr, cfg.cvar_alpha)

        er_err_mog = jnp.abs(entropic_risk_from_cdf(Fmogr, xr, cfg.er_beta) - ertn)
        er_err_moc = jnp.abs(entropic_risk_from_cdf(Fmocr, xr, cfg.er_beta) - ertn)
        er_err_mg = jnp.abs(entropic_risk_from_cdf(Fmgr, xr, cfg.er_beta) - ertn)

        def mer(mp, vp, mf, vf):
            return jnp.abs(mp - mf), jnp.abs(vp - vf), jnp.abs((mp - cfg.mv_lambda * vp) - mvtn)

        wv = jax.nn.softmax(lg1[0], axis=-1)
        sig = _softplus(ls1[0])
        mmog = jnp.sum(wv * mu1[0])
        vmog = jnp.sum(wv * (sig**2 + mu1[0] ** 2)) - mmog**2
        memog = mer(mmog.astype(jnp.float64), vmog.astype(jnp.float64), mtn, vtn)

        memoc = mer(_mixture_mean_moc(lg2[0], loc2[0], lsc2[0]), jnp.float64(jnp.nan), mtn, vtn)

        w3 = jax.nn.softmax(lg3[0], axis=-1)
        k3, th3 = _softplus(lsh3[0]), _softplus(lsc3[0])
        m_comp = k3 * th3
        v_comp = k3 * th3**2
        mm_g = jnp.sum(w3 * m_comp)
        vm_g = jnp.sum(w3 * (v_comp + m_comp**2)) - mm_g**2
        memg = mer(mm_g.astype(jnp.float64), vm_g.astype(jnp.float64), mtn, vtn)

        cf_l2_mog = cf_l2_sq_over_omega2(phimog.astype(jnp.complex128), pt, te, cfg.gp_t_delta)
        cf_l2_moc = cf_l2_sq_over_omega2(phimoc.astype(jnp.complex128), pt, te, cfg.gp_t_delta)
        cf_l2_mg = cf_l2_sq_over_omega2(phimg.astype(jnp.complex128), pt, te, cfg.gp_t_delta)

        out = {
            "phi_mse_phi_mog": cf_mse_vs_truth(phimog, pt),
            "phi_mse_phi_moc": cf_mse_vs_truth(phimoc, pt),
            "phi_mse_phi_mogamma": cf_mse_vs_truth(phimg, pt),
            "ks_phi_mog": ks_on_grid(Fmog, Ft),
            "ks_phi_moc": ks_on_grid(Fmoc, Ft),
            "ks_phi_mogamma": ks_on_grid(Fmg, Ft),
            "w1_phi_mog": w1_cdf(Fmog, Ft, xe),
            "w1_phi_moc": w1_cdf(Fmoc, Ft, xe),
            "w1_phi_mogamma": w1_cdf(Fmg, Ft, xe),
            "cramer_l2_phi_mog": cramer_l2_sq_cdf(Fmog, Ft, xe),
            "cramer_l2_phi_moc": cramer_l2_sq_cdf(Fmoc, Ft, xe),
            "cramer_l2_phi_mogamma": cramer_l2_sq_cdf(Fmg, Ft, xe),
            "cf_l2_w_phi_mog": cf_l2_mog,
            "cf_l2_w_phi_moc": cf_l2_moc,
            "cf_l2_w_phi_mogamma": cf_l2_mg,
            "cvar_err_phi_mog": jnp.abs(cvar_from_cdf(Fmogr, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_phi_moc": jnp.abs(cvar_from_cdf(Fmocr, xr, cfg.cvar_alpha) - cv_t),
            "cvar_err_phi_mogamma": jnp.abs(cvar_from_cdf(Fmgr, xr, cfg.cvar_alpha) - cv_t),
            "er_err_phi_mog": er_err_mog,
            "er_err_phi_moc": er_err_moc,
            "er_err_phi_mogamma": er_err_mg,
            "phi_phi_mog": phimog,
            "phi_phi_moc": phimoc,
            "phi_phi_mogamma": phimg,
            "F_phi_mog": Fmog,
            "F_phi_moc": Fmoc,
            "F_phi_mogamma": Fmg,
            "phi_true": pt,
            "F_true": Ft,
            "pdf_true": pdf_t,
            "pdf_phi_mog": pdfmog,
            "pdf_phi_moc": pdfmoc,
            "pdf_phi_mogamma": pdfmg,
            "mean_err_phi_mog": memog[0],
            "mean_err_phi_moc": memoc[0],
            "mean_err_phi_mogamma": memg[0],
            "var_err_phi_mog": memog[1],
            "var_err_phi_moc": memoc[1],
            "var_err_phi_mogamma": memg[1],
            "mv_err_phi_mog": memog[2],
            "mv_err_phi_moc": memoc[2],
            "mv_err_phi_mogamma": memg[2],
        }

        for m in MK:
            out[f"{m}_phitd_mog"] = out[f"{m}_phi_mog"]
            out[f"{m}_phitd_cauchy"] = out[f"{m}_phi_moc"]
            out[f"{m}_phitd_mogamma"] = out[f"{m}_phi_mogamma"]

        return out

    @jax.jit
    def ev(pm1, pm2, pm3, xs):
        P = [one(pm1, pm2, pm3, xs, kk) for kk in range(NUM_ACTIONS)]
        return {k: jnp.stack([p[k] for p in P]) for k in P[0]}

    return ev, te, xe


def evaluate_test_set_fp(cfg: DAConfig, nets, params, xt):
    ev, te, xe = _make_eval_fp(cfg, *nets)
    ech = jax.jit(jax.vmap(ev, (None, None, None, 0)))
    n, cs = xt.shape[0], cfg.eval_chunk_size
    assert n % cs == 0
    ch = xt.reshape(n // cs, cs, -1)
    parts = [ech(*params, ch[i]) for i in range(ch.shape[0])]
    return {k: jnp.concatenate([p[k] for p in parts]) for k in parts[0]}, np.asarray(te), np.asarray(xe)


FP_PANEL_EXPORT_KEYS = (
    "pdf_true",
    "phi_true",
    "F_true",
    "phi_phi_mog",
    "phi_phi_moc",
    "phi_phi_mogamma",
    "F_phi_mog",
    "F_phi_moc",
    "F_phi_mogamma",
    "pdf_phi_mog",
    "pdf_phi_moc",
    "pdf_phi_mogamma",
)


def _collect_eval_payload_fp(cfg: DAConfig, tag: str, er: dict, te: np.ndarray, xe: np.ndarray, *, panel_index: int = 0) -> tuple[dict, dict]:
    pm = {f"{m}_{a}": np.asarray(er[f"{m}_{a}"]) for a in FP_ORDER for m in MK if f"{m}_{a}" in er}
    panel_data: dict[str, list] = {}
    for k in FP_PANEL_EXPORT_KEYS:
        if k not in er:
            continue
        arr = np.asarray(er[k])[panel_index]
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        panel_data[f"panel_{k}"] = arr.tolist()

    def _means_local(pm_dict: dict, algo: str):
        return np.stack([np.mean(np.asarray(pm_dict[f"{m}_{algo}"]), 0) for m in MK])

    metric_means = {}
    ok = [a for a in FP_ORDER if all(f"{m}_{a}" in pm for m in MK)]
    for a in ok:
        metric_means[a] = _means_local(pm, a).tolist()
    for new_name, old_name in LEGACY_ALGO_ALIASES_FP.items():
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
        "experiment": "flexibility_parametric_family",
    }
    return payload, pm


def _save_parseval_training_plot_fp(
    cfg: DAConfig, tag: str, history: dict[str, list[float]], output_dir: Path
) -> dict[str, str] | None:
    if not history.get("step"):
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"warning: parseval plot skipped (matplotlib import failed: {exc})", flush=True)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"flexibility_parametric_family_parseval_training_phi_mog_seed{cfg.seed}"
    csv_path = stem.with_suffix(".csv")

    steps = np.asarray(history["step"], dtype=np.float64)

    def _as_seed_matrix(values: list[float] | list[list[float]]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            return arr[None, :]
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Expected 1D or 2D parseval history, got shape {arr.shape}.")

    cr_matrix = _as_seed_matrix(history["cramer_l2_phi_mog"])
    cf_matrix = _as_seed_matrix(history["cf_l2_w_phi_mog"])
    y_cr = np.clip(np.nan_to_num(np.nanmean(cr_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_cf = np.clip(np.nan_to_num(np.nanmean(cf_matrix, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    y_cr_std = np.nan_to_num(np.nanstd(cr_matrix, axis=0, ddof=0), nan=0.0, posinf=0.0, neginf=0.0)
    y_cf_std = np.nan_to_num(np.nanstd(cf_matrix, axis=0, ddof=0), nan=0.0, posinf=0.0, neginf=0.0)
    cr_color = "#5E2B97"
    cf_color = "#A06CD5"

    configure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(6.1, 3.5))
    ax.fill_between(steps, np.clip(y_cr - y_cr_std, 0.0, None), y_cr + y_cr_std, color=cr_color, alpha=0.2, lw=0)
    ax.fill_between(steps, np.clip(y_cf - y_cf_std, 0.0, None), y_cf + y_cf_std, color=cf_color, alpha=0.2, lw=0)
    ax.plot(steps, y_cr, color=cr_color, lw=2.0, label=r"Cramer $L_2^2$ (CDF)")
    ax.plot(steps, y_cf, color=cf_color, lw=2.0, ls="--", label=r"CF $L_2^2/\omega^2$")
    pos = np.concatenate([y_cr[np.isfinite(y_cr)], y_cf[np.isfinite(y_cf)]])
    pos = pos[pos > 0]
    if pos.size and float(np.nanmin(pos)) > 0:
        ax.set_yscale("log")
    style_axes_wandb_curve(ax)
    ax.set_title("Parseval anchor (φTD-MoG)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 1])
    png_out, pdf_out = save_figure_png_and_pdf(fig, stem, dpi_png=300, dpi_pdf=300)
    plt.close(fig)

    fieldnames = (
        "step",
        "cramer_l2_phi_mog",
        "cf_l2_w_phi_mog",
        "cramer_l2_phi_mog_std",
        "cf_l2_w_phi_mog_std",
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, st in enumerate(steps.tolist()):
            writer.writerow(
                {
                    "step": st,
                    "cramer_l2_phi_mog": y_cr[i],
                    "cf_l2_w_phi_mog": y_cf[i],
                    "cramer_l2_phi_mog_std": y_cr_std[i],
                    "cf_l2_w_phi_mog_std": y_cf_std[i],
                }
            )

    return {"png": png_out, "pdf": pdf_out, "csv": str(csv_path)}


# --- Plotting (parallel structure to run_distribution_analysis plot blocks) ---
FP_PLOT_TITLES = {
    "phi_mog": r"$\varphi\text{TD-Gaussian}$",
    "phi_moc": r"$\varphi\text{TD-Cauchy}$",
    "phi_mogamma": r"$\varphi\text{TD-Gamma}$",
}
FP_PLOT_VISIBLE_ALGOS = ("phi_mog", "phi_moc", "phi_mogamma")
FP_PLOT_COLORS = {
    "truth": "#0A0A0A",
    "phi_mog": algo_color("phitd_mog"),
    "phi_moc": algo_color("phitd_cauchy"),
    "phi_mogamma": algo_color("phitd_mogamma"),
}
FP_PLOT_LW = {
    "truth": 2.15,
    "phi_mog": 1.12,
    "phi_moc": 1.12,
    "phi_mogamma": 1.12,
}


def _fp_plot_row_labels(row_keys: tuple[str, ...]) -> list[str]:
    from run_distribution_analysis import PLOT_ROW_LBL

    return [PLOT_ROW_LBL.get(k, k) for k in row_keys]


def _fp_plot_resolve_row_keys(payloads: list[dict]) -> tuple[str, ...]:
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


def _fp_plot_resolve_algo_key(metric_means: dict, algo: str) -> str | None:
    if algo in metric_means:
        return algo
    legacy = LEGACY_ALGO_ALIASES_FP.get(algo)
    if legacy and legacy in metric_means:
        return legacy
    return None


def _fp_plot_resolve_panel_key(data: dict[str, np.ndarray], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in data:
            return key
    return None


def _fp_plot_fmt_disp(x: float) -> str:
    return "-" if np.isnan(float(x)) else f"{float(x):.3g}"


def _fp_plot_save_fig(fig, path: str, *, png_dpi: int = 300) -> None:
    stem = os.path.splitext(path)[0]
    save_figure_png_and_pdf(fig, stem, dpi_png=png_dpi, dpi_pdf=png_dpi, pad_inches=0.08)


def _fp_plot_metric_tables(stderr: dict[str, tuple[np.ndarray, np.ndarray]], row_keys: tuple[str, ...], path: str) -> None:
    import matplotlib.pyplot as plt

    algos = [a for a in FP_PLOT_VISIBLE_ALGOS if a in stderr]
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
    row_labels = _fp_plot_row_labels(row_keys)
    win_bg = "#D2F4D9"

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
                    row.append(_fp_plot_fmt_disp(mv))
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
                cell.set_facecolor(win_bg)
        ax.text(0.5, 1.02, FP_PLOT_TITLES[algo], transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")

    _fp_plot_save_fig(fig, path)
    plt.close(fig)


def _fp_plot_smooth_curve(y: np.ndarray, passes: int = 2) -> np.ndarray:
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


def _fp_plot_minimal_style(ax) -> None:
    style_axes_panel(ax)


def _fp_plot_panels(mean_p: dict[str, np.ndarray], se_p: dict[str, np.ndarray], te: np.ndarray, xe: np.ndarray, cf_hw: float, path: str) -> None:
    import matplotlib.pyplot as plt

    cf_series = (
        (("panel_phi_true",), "truth", "Truth"),
        (("panel_phi_phi_mog",), "phi_mog", FP_PLOT_TITLES["phi_mog"]),
        (("panel_phi_phi_moc",), "phi_moc", FP_PLOT_TITLES["phi_moc"]),
        (("panel_phi_phi_mogamma",), "phi_mogamma", FP_PLOT_TITLES["phi_mogamma"]),
    )
    pdf_series = (
        (("panel_pdf_true",), "truth", "Truth"),
        (("panel_pdf_phi_mog",), "phi_mog", FP_PLOT_TITLES["phi_mog"]),
        (("panel_pdf_phi_moc",), "phi_moc", FP_PLOT_TITLES["phi_moc"]),
        (("panel_pdf_phi_mogamma",), "phi_mogamma", FP_PLOT_TITLES["phi_mogamma"]),
    )
    cdf_series = (
        (("panel_F_true",), "truth", "Truth"),
        (("panel_F_phi_mog",), "phi_mog", FP_PLOT_TITLES["phi_mog"]),
        (("panel_F_phi_moc",), "phi_moc", FP_PLOT_TITLES["phi_moc"]),
        (("panel_F_phi_mogamma",), "phi_mogamma", FP_PLOT_TITLES["phi_mogamma"]),
    )

    fig, axes = plt.subplots(len(ACTION_DISTS), 4, figsize=(13.5, 2.45 * len(ACTION_DISTS)))
    for row, dist in enumerate(ACTION_DISTS):
        axd, axcf, axp, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        if "panel_pdf_true" in mean_p:
            y_truth = np.asarray(mean_p["panel_pdf_true"][row], dtype=np.float64)
            axd.plot(xe[xmask], y_truth[xmask], color=FP_PLOT_COLORS["truth"], lw=FP_PLOT_LW["truth"])
        _fp_plot_minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axd.set_title("Density")

        tmask = np.abs(te) <= cf_hw
        for candidates, ckey, label in cf_series:
            pkey = _fp_plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = np.real(mean_p[pkey][row])
            ys = np.real(se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row])
            axcf.fill_between(te[tmask], (y - ys)[tmask], (y + ys)[tmask], color=FP_PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axcf.plot(te[tmask], y[tmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
        _fp_plot_minimal_style(axcf)
        axcf.set_xlim(-cf_hw, cf_hw)
        if row == 0:
            axcf.set_title(r"Re $\varphi$")

        for candidates, ckey, label in pdf_series:
            pkey = _fp_plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            if ckey == "truth":
                y = np.asarray(mean_p[pkey][row], dtype=np.float64)
                ys = np.asarray(se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], dtype=np.float64)
            else:
                y = _fp_plot_smooth_curve(mean_p[pkey][row])
                ys = _fp_plot_smooth_curve(se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], passes=1)
            axp.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=FP_PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axp.plot(xe[xmask], y[xmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
        _fp_plot_minimal_style(axp)
        axp.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axp.set_title("PDF")

        for candidates, ckey, label in cdf_series:
            pkey = _fp_plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][row]
            ys = se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row]
            axc.fill_between(
                xe[xmask],
                (y - ys)[xmask],
                (y + ys)[xmask],
                color=FP_PLOT_COLORS[ckey],
                alpha=0.2,
                lw=0,
            )
            axc.plot(xe[xmask], y[xmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
        _fp_plot_minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axc.set_title("CDF")

    axes[0, 3].legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.10, wspace=0.24, hspace=0.22)
    _fp_plot_save_fig(fig, path)
    plt.close(fig)


def _fp_plot_density_cdf_only_panels(
    mean_p: dict[str, np.ndarray],
    se_p: dict[str, np.ndarray],
    xe: np.ndarray,
    path_2x4: str,
    path_4x2: str,
) -> None:
    import matplotlib.pyplot as plt

    def _title_case_name(name: str) -> str:
        return name

    density_truth_key = "panel_pdf_true"
    cdf_series = (
        (("panel_F_true",), "truth", "Truth"),
        (("panel_F_phi_mog",), "phi_mog", FP_PLOT_TITLES["phi_mog"]),
        (("panel_F_phi_moc",), "phi_moc", FP_PLOT_TITLES["phi_moc"]),
        (("panel_F_phi_mogamma",), "phi_mogamma", FP_PLOT_TITLES["phi_mogamma"]),
    )

    fig, axes = plt.subplots(2, len(ACTION_DISTS), figsize=(12.8, 5.1))
    for col_i, dist in enumerate(ACTION_DISTS):
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        ax_density = axes[0, col_i]
        ax_cdf = axes[1, col_i]
        if density_truth_key in mean_p:
            ax_density.plot(xe[xmask], np.asarray(mean_p[density_truth_key][col_i], dtype=np.float64)[xmask], color=FP_PLOT_COLORS["truth"], lw=FP_PLOT_LW["truth"])
        ax_density.set_title(dist.name.split()[0] if " " in dist.name else dist.name, fontsize=9, fontweight="bold")
        ax_density.set_xlim(dist.plot_x_min, dist.plot_x_max)
        _fp_plot_minimal_style(ax_density)

        for candidates, ckey, label in cdf_series:
            pkey = _fp_plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][col_i]
            ys = se_p.get(pkey, np.zeros_like(mean_p[pkey]))[col_i]
            ax_cdf.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=FP_PLOT_COLORS[ckey], alpha=0.2, lw=0)
            ax_cdf.plot(xe[xmask], y[xmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
        ax_cdf.set_xlim(dist.plot_x_min, dist.plot_x_max)
        _fp_plot_minimal_style(ax_cdf)

    axes[0, 0].set_ylabel("Density", fontweight="bold")
    axes[1, 0].set_ylabel("CDF", fontweight="bold")
    axes[1, 0].legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.10, wspace=0.22, hspace=0.23)
    _fp_plot_save_fig(fig, path_2x4)
    plt.close(fig)

    fig, axes = plt.subplots(len(ACTION_DISTS), 2, figsize=(7.4, 9.4))
    axes[0, 0].set_title("Density", fontweight="bold")
    axes[0, 1].set_title("CDF", fontweight="bold")
    for row, dist in enumerate(ACTION_DISTS):
        axd, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        y_truth = np.asarray(mean_p[density_truth_key][row], dtype=np.float64)
        axd.plot(xe[xmask], y_truth[xmask], color=FP_PLOT_COLORS["truth"], lw=FP_PLOT_LW["truth"], label="Truth")
        _fp_plot_minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        for candidates, ckey, label in cdf_series:
            pkey = _fp_plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][row]
            ys = se_p.get(pkey, np.zeros_like(mean_p[pkey]))[row]
            axc.fill_between(xe[xmask], (y - ys)[xmask], (y + ys)[xmask], color=FP_PLOT_COLORS[ckey], alpha=0.2, lw=0)
            axc.plot(xe[xmask], y[xmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
        _fp_plot_minimal_style(axc)
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
    _fp_plot_save_fig(fig, path_4x2)
    plt.close(fig)


def _fp_plot_aggregate_payloads(
    payloads: list[dict], row_keys: tuple[str, ...]
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    if not payloads:
        raise ValueError("No payloads to aggregate.")

    metric_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for algo in FP_ORDER:
        mats = []
        for payload in payloads:
            metric_means = payload.get("metric_means", {})
            algo_key = _fp_plot_resolve_algo_key(metric_means, algo)
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


def _fp_plot_write_metrics_csv(path: str, metric_stats: dict[str, tuple[np.ndarray, np.ndarray]], row_keys: tuple[str, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["algorithm", "metric"] + [f"{x.short}_mean" for x in ACTION_DISTS] + [f"{x.short}_se" for x in ACTION_DISTS]
        writer.writerow(header)
        for algo in FP_ORDER:
            if algo not in metric_stats:
                continue
            mm, sm = metric_stats[algo]
            for row_idx, row_key in enumerate(row_keys):
                writer.writerow([algo, row_key, *[float(x) for x in mm[row_idx]], *[float(x) for x in sm[row_idx]]])


def plot_flexibility_parametric_family_local(payloads: list[dict], output_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    if not payloads:
        raise ValueError("No payloads to plot.")
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_keys = _fp_plot_resolve_row_keys(payloads)
    metric_stats, panel_mean, panel_se, t_eval, x_eval, omega_max = _fp_plot_aggregate_payloads(payloads, row_keys)

    metrics_png_base = str(output_dir / "flexibility_parametric_family_metrics_combined.png")
    panels_png_base = str(output_dir / "flexibility_parametric_family_panels.png")
    panels_density_cdf_2x4_base = str(output_dir / "flexibility_parametric_family_panels_density_cdf_2x4.png")
    panels_density_cdf_4x2_base = str(output_dir / "flexibility_parametric_family_panels_density_cdf_4x2.png")
    metrics_csv = str(output_dir / "flexibility_parametric_family_metrics_combined.csv")

    _fp_plot_metric_tables(metric_stats, row_keys, metrics_png_base)
    cf_hw = float(omega_max)
    _fp_plot_panels(panel_mean, panel_se, t_eval, x_eval, cf_hw, panels_png_base)
    _fp_plot_density_cdf_only_panels(
        panel_mean,
        panel_se,
        x_eval,
        panels_density_cdf_2x4_base,
        panels_density_cdf_4x2_base,
    )
    _fp_plot_write_metrics_csv(metrics_csv, metric_stats, row_keys)

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


def _wandb_log_eval_metrics_fp(pm: dict, tag: str, *, step: int) -> None:
    import wandb

    ok = [a for a in FP_ORDER if all(f"{m}_{a}" in pm for m in MK)]
    wandb.summary["tag"] = tag
    wandb.summary["experiment_tag"] = tag
    wandb.summary["algos_in_metrics"] = ",".join(ok) if ok else ""
    out: dict = {}
    for a in ok:
        marr = np.stack([np.mean(np.asarray(pm[f"{m}_{a}"]), 0) for m in MK])
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
    wandb.log(out, step=step)


def train_and_export_fp(
    cfg: DAConfig,
    tag: str,
    wb: bool,
    *,
    figures_run_dir: Path,
    skip_plots: bool = False,
) -> None:
    figures_run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(figures_run_dir / "flexibility_parametric_family_experiment_config.yaml", _config_for_logging(cfg))

    rng = jax.random.PRNGKey(cfg.seed)
    ri, rd, rt, ru = jax.random.split(rng, 4)
    print(f"Dataset... offset={cfg.dataset_seed_offset}", flush=True)
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
    smog, smoc, smoga, *nets = create_train_states_fp(ri, cfg)
    parseval_eval, _, _ = _make_eval_fp(cfg, *nets)
    anchor_x = jax.random.normal(PANEL_ANCHOR_PRNG_KEY, (cfg.state_dim,), jnp.float32)
    parseval_history: dict[str, list[float]] = {
        "step": [],
        "cramer_l2_phi_mog": [],
        "cf_l2_w_phi_mog": [],
    }
    run = make_train_chunk_fp(cfg, *nets, ds)
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
        smog, smoc, smoga, rng, ls = run(smog, smoc, smoga, rng)
        jax.tree.map(lambda x: x.block_until_ready(), ls)
        st = (c + 1) * pe
        last_losses = {
            "train/loss_phi_mog_final": float(ls[0][-1]),
            "train/loss_phi_moc_final": float(ls[1][-1]),
            "train/loss_phi_mogamma_final": float(ls[2][-1]),
        }
        last_losses["train/loss_phitd_mog_final"] = last_losses["train/loss_phi_mog_final"]
        last_losses["train/loss_phitd_cauchy_final"] = last_losses["train/loss_phi_moc_final"]
        last_losses["train/loss_phitd_mogamma_final"] = last_losses["train/loss_phi_mogamma_final"]
        if st % int(cfg.log_every) == 0 or st == int(cfg.total_steps) or c == 0:
            print(
                f"  {st:>7}  phi_mog={last_losses['train/loss_phi_mog_final']:.4f} "
                f"phi_moc={last_losses['train/loss_phi_moc_final']:.4f} "
                f"phi_mogamma={last_losses['train/loss_phi_mogamma_final']:.4f}  {time.perf_counter()-tc:.1f}s",
                flush=True,
            )

        pv = parseval_eval(smog.params, smoc.params, smoga.params, anchor_x)
        jax.tree.map(lambda x: x.block_until_ready(), pv)
        parseval_history["step"].append(float(st))
        cr_m = float(jnp.mean(pv["cramer_l2_phi_mog"]))
        cf_m = float(jnp.mean(pv["cf_l2_w_phi_mog"]))
        parseval_history["cramer_l2_phi_mog"].append(cr_m)
        parseval_history["cf_l2_w_phi_mog"].append(cf_m)
        pv_log = {
            "train/parseval/phi_mog/cramer_l2_anchor": cr_m,
            "train/parseval/phi_mog/cf_l2_w_anchor": cf_m,
        }
        if wb and wandb_mod is not None:
            wandb_mod.log(pv_log, step=st)

    print(f"train {time.perf_counter()-t0:.1f}s", flush=True)
    parseval_plot_paths = None
    if not skip_plots:
        parseval_plot_paths = _save_parseval_training_plot_fp(cfg, tag, parseval_history, figures_run_dir)
    xt = jax.random.fold_in(rt, cfg.test_seed_offset)
    xv = jax.random.normal(xt, (cfg.n_test, cfg.state_dim), jnp.float32).at[0].set(
        jax.random.normal(PANEL_ANCHOR_PRNG_KEY, (cfg.state_dim,), jnp.float32)
    )
    er, te, xe = evaluate_test_set_fp(
        cfg,
        tuple(nets),
        (smog.params, smoc.params, smoga.params),
        xv,
    )
    jax.tree.map(lambda x: x.block_until_ready(), er)

    payload, pm = _collect_eval_payload_fp(cfg, tag, er, te, xe, panel_index=0)
    payload["gamma"] = float(cfg.gamma)
    payload["immediate_reward_std"] = float(cfg.immediate_reward_std)

    if not skip_plots:
        try:
            plot_paths = plot_flexibility_parametric_family_local([payload], figures_run_dir)
            print(f"Saved figures: {plot_paths}", flush=True)
        except Exception as exc:
            print(f"warning: post-run plotting failed: {exc}", flush=True)
    if wb:
        if last_losses:
            wandb_mod.summary.update(last_losses)
        if parseval_plot_paths:
            for key, p in parseval_plot_paths.items():
                if key == "csv":
                    continue
                wandb_mod.save(p, policy="now")
            wandb_mod.summary["parseval_train_plot_png"] = parseval_plot_paths["png"]
            wandb_mod.summary["parseval_train_plot_pdf"] = parseval_plot_paths.get("pdf", "")
            wandb_mod.summary["parseval_train_metrics_csv"] = parseval_plot_paths["csv"]
        _wandb_log_eval_metrics_fp(pm, tag, step=cfg.total_steps)
        wandb_mod.summary["flexibility_parametric_family_artifacts_dir"] = str(figures_run_dir)


def _default_config_yaml_path() -> Path:
    return _artifact_purejaxql_dir() / "config" / "analysis" / "distribution.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Offline bandit + parametric φTD families (MoG/MoC/MoGamma): same protocol as distribution_analysis.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML for DAConfig (default: purejaxql/config/analysis/distribution.yaml).",
    )
    p.add_argument("--figures-dir", type=Path, default=None, help="Root for run output (default: figures/flexibility_parametric_family).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--experiment-tag", default="FlexParamFamily")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--parseval-every", type=int, default=None)
    p.add_argument("--dataset-size", type=int)
    p.add_argument("--omega-laplace-scale", type=float)
    p.add_argument("--num-qtd-quantiles", type=int)
    p.add_argument("--num-ctd-atoms", type=int)
    p.add_argument(
        "--num-mixture-components",
        type=int,
        dest="num_dirac_components",
        help="Components K for each φTD mixture head (same field as num_dirac_components in YAML).",
    )
    p.add_argument(
        "--num-dirac-components",
        type=int,
        dest="num_dirac_components",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--close-to-theory",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="close_to_theory",
        help="φTD: Pareto ω + unweighted CF MSE vs half-Laplacian ω + CF MSE/ω².",
    )
    p.add_argument("--omega-min-pareto", type=float, default=None, dest="omega_min_pareto")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    figures_root = args.figures_dir if args.figures_dir is not None else (_REPO_ROOT / "figures" / "flexibility_parametric_family")

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
        "num_dirac_components": args.num_dirac_components,
    }
    if getattr(args, "close_to_theory", None) is not None:
        overrides["close_to_theory"] = args.close_to_theory
    if getattr(args, "omega_min_pareto", None) is not None:
        overrides["omega_min_pareto"] = args.omega_min_pareto
    base_cfg = load_da_config_yaml(args.config or _default_config_yaml_path(), overrides)
    cfg = DAConfig(**{f.name: getattr(base_cfg, f.name) for f in fields(DAConfig)})
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
        train_and_export_fp(cfg, tag, use_wandb, figures_run_dir=run_dir, skip_plots=args.no_plots)
    finally:
        if use_wandb:
            import wandb

            if wandb.run:
                wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
