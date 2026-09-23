#!/usr/bin/env python3
"""Offline contextual bandit: parametric φTD families + Categorical / Quantile.

Truth action distributions (same as distribution_analysis):
  Gaussian, Cauchy mixture, MoG, Gamma.

φTD algorithms benchmarked:
  MoG (Gaussian), MoCauchy, MoGamma, Categorical, Quantile.

Logged metrics: W1, Cramér ℓ₂², CF ℓ₂²/ω² only.
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

from paper_plots import (
    configure_matplotlib,
    pdf_path_for_png_stem,
    save_figure_png_and_pdf,
    style_axes_panel,
    style_axes_wandb_curve,
)
from plot_colors import algo_color
from purejaxql.utils.mog_cf import build_categorical_cf, build_quantile_cf
from run_distribution_analysis import (
    ACTION_DISTS,
    DAConfig,
    MK,
    NUM_ACTIONS,
    CTDNet,
    QTDNet,
    _INVERSE_2PI,
    _config_for_logging,
    _dense_ln_relu,
    _huber,
    _panel_anchor_prng_key,
    _write_yaml,
    build_dataset,
    cf_l2_sq_over_omega2,
    cramer_l2_sq_cdf,
    ctd_atoms,
    discrete_return_cdf,
    discrete_return_cf,
    empirical_cdf,
    empirical_cf_from_samples,
    gil_pelaez_cdf,
    load_da_config_yaml,
    sample_frequencies,
    w1_cdf,
)

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parent


def _artifact_purejaxql_dir() -> Path:
    return _REPO_ROOT / "purejaxql"


FP_ORDER = ("phi_mog", "phi_moc", "phi_mogamma", "phi_cat", "phi_qt")
LEGACY_ALGO_ALIASES_FP = {
    "phi_mog": "phitd_mog",
    "phi_moc": "phitd_cauchy",
    "phi_mogamma": "phitd_mogamma",
    "phi_cat": "phitd_fcm",
    "phi_qt": "phitd_fqm",
}


def _softplus(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    return jax.nn.softplus(x) + eps


def normal_cdf(z: jax.Array) -> jax.Array:
    return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))


def phi_gaussian_mixture(t, weights, mu, sigma):
    expo = 1j * mu[:, None] * t[None, :] - 0.5 * (sigma[:, None] ** 2) * (t[None, :] ** 2)
    return jnp.sum(weights[:, None] * jnp.exp(expo), axis=0)


def cdf_gaussian_mixture(x, weights, mu, sigma):
    z = (x[None, :] - mu[:, None]) / sigma[:, None]
    return jnp.sum(weights[:, None] * normal_cdf(z), axis=0)


def pdf_gaussian_mixture(x, weights, mu, sigma):
    z = (x[None, :] - mu[:, None]) / sigma[:, None]
    p = jnp.exp(-0.5 * z**2) / (sigma[:, None] * jnp.sqrt(2.0 * jnp.pi))
    return jnp.sum(weights[:, None] * p, axis=0)


def phi_cauchy_mixture(t, weights, loc, scale):
    expo = 1j * loc[:, None] * t[None, :] - scale[:, None] * jnp.abs(t[None, :])
    return jnp.sum(weights[:, None] * jnp.exp(expo), axis=0)


def cdf_cauchy_mixture(x, weights, loc, scale):
    vals = jnp.arctan((x[None, :] - loc[:, None]) / scale[:, None]) / jnp.pi + 0.5
    return jnp.sum(weights[:, None] * vals, axis=0)


def pdf_cauchy_mixture(x, weights, loc, scale):
    z = (x[None, :] - loc[:, None]) / scale[:, None]
    p = 1.0 / (jnp.pi * scale[:, None] * (1.0 + z**2))
    return jnp.sum(weights[:, None] * p, axis=0)


def phi_gamma_mixture(t, weights, shape, scale):
    z = (1.0 - 1j * scale[:, None] * t[None, :]).astype(jnp.complex128)
    comp = jnp.power(z, -shape[:, None])
    return jnp.sum(weights[:, None] * comp, axis=0)


def cdf_gamma_mixture(x, weights, shape, scale):
    rat = jnp.maximum(x[None, :], 0.0) / jnp.maximum(scale[:, None], jnp.finfo(jnp.float64).tiny)
    vals = jnp.where(x[None, :] > 0, jax.scipy.special.gammainc(shape[:, None], rat), 0.0)
    return jnp.sum(weights[:, None] * vals, axis=0)


def pdf_gamma_mixture(x, weights, shape, scale):
    xp = jnp.maximum(x[None, :], jnp.finfo(jnp.float64).tiny)
    logp = (
        (shape[:, None] - 1.0) * jnp.log(xp)
        - xp / scale[:, None]
        - jax.scipy.special.gammaln(shape[:, None])
        - shape[:, None] * jnp.log(scale[:, None])
    )
    vals = jnp.where(x[None, :] > 0, jnp.exp(logp), 0.0)
    return jnp.sum(weights[:, None] * vals, axis=0)


def model_phi_return(family: str, params: dict, t: jax.Array) -> jax.Array:
    w = jax.nn.softmax(params["logits"], axis=-1)
    if family == "mog":
        return phi_gaussian_mixture(t, w, params["mu"], _softplus(params["log_sigma"]))
    if family == "moc":
        return phi_cauchy_mixture(t, w, params["loc"], _softplus(params["log_scale"]))
    return phi_gamma_mixture(t, w, _softplus(params["log_shape"]), _softplus(params["log_scale"]))


def model_cdf(family: str, params: dict, x: jax.Array) -> jax.Array:
    w = jax.nn.softmax(params["logits"], axis=-1)
    if family == "mog":
        return cdf_gaussian_mixture(x, w, params["mu"], _softplus(params["log_sigma"]))
    if family == "moc":
        return cdf_cauchy_mixture(x, w, params["loc"], _softplus(params["log_scale"]))
    return cdf_gamma_mixture(x, w, _softplus(params["log_shape"]), _softplus(params["log_scale"]))


def model_pdf(family: str, params: dict, x: jax.Array) -> jax.Array:
    w = jax.nn.softmax(params["logits"], axis=-1)
    if family == "mog":
        return pdf_gaussian_mixture(x, w, params["mu"], _softplus(params["log_sigma"]))
    if family == "moc":
        return pdf_cauchy_mixture(x, w, params["loc"], _softplus(params["log_scale"]))
    return pdf_gamma_mixture(x, w, _softplus(params["log_shape"]), _softplus(params["log_scale"]))


class PhiMoGNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_components, name="logits")(h), nn.Dense(self.num_components, name="mu")(h), nn.Dense(
            self.num_components, name="log_sigma"
        )(h)


class PhiMoCNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_components, name="logits")(h), nn.Dense(self.num_components, name="loc")(h), nn.Dense(
            self.num_components, name="log_scale"
        )(h)


class PhiMoGammaNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return (
            nn.Dense(self.num_components, name="logits")(h),
            nn.Dense(self.num_components, name="log_shape")(h),
            nn.Dense(self.num_components, name="log_scale")(h),
        )


def _opt(lr: float, clip: float):
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def _params_mog(logits, mu, log_sigma):
    return {"logits": logits, "mu": mu, "log_sigma": log_sigma}


def _params_moc(logits, loc, log_scale):
    return {"logits": logits, "loc": loc, "log_scale": log_scale}


def _params_mogamma(logits, log_shape, log_scale):
    return {"logits": logits, "log_shape": log_shape, "log_scale": log_scale}


def _mixture_mean_mog(logits, mu, log_sigma):
    return jnp.sum(jax.nn.softmax(logits, -1) * mu, -1)


def _mixture_mean_moc(logits, loc, log_scale):
    return jnp.full(loc.shape[:-1], jnp.nan, dtype=loc.dtype)


def _mixture_mean_mogamma(logits, log_shape, log_scale):
    w = jax.nn.softmax(logits, -1)
    return jnp.sum(w * _softplus(log_shape) * _softplus(log_scale), -1)


def create_train_states_fp(rng: jax.Array, cfg: DAConfig):
    # Exact same 3-way split as the original MoG/MoC/MoGamma run (seed-matched);
    # Cat/Quant keys are folded in so they do not perturb those inits.
    rmog, rmoc, rmg = jax.random.split(rng, 3)
    rcat = jax.random.fold_in(rng, 11)
    rqt = jax.random.fold_in(rng, 13)
    k = cfg.num_dirac_components
    phi_mog = PhiMoGNet(cfg.hidden_dim, cfg.trunk_layers, k)
    phi_moc = PhiMoCNet(cfg.hidden_dim, cfg.trunk_layers, k)
    phi_mogamma = PhiMoGammaNet(cfg.hidden_dim, cfg.trunk_layers, k)
    phi_cat = CTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    phi_qt = QTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_tau)
    x0 = jnp.zeros((2, cfg.state_dim), jnp.float32)
    a0 = jnp.zeros((2, NUM_ACTIONS), jnp.float32)
    tx = _opt(cfg.lr, cfg.max_grad_norm)
    nets_keys = (
        (phi_mog, rmog),
        (phi_moc, rmoc),
        (phi_mogamma, rmg),
        (phi_cat, rcat),
        (phi_qt, rqt),
    )
    states = []
    nets = []
    for net, key in nets_keys:
        v = net.init(key, x0, a0)
        states.append(train_state.TrainState.create(apply_fn=net.apply, params=v["params"], tx=tx))
        nets.append(net)
    return (*states, *nets)


def make_train_chunk_fp(cfg, phi_mog, phi_moc, phi_mogamma, phi_cat, phi_qt, ds):
    xa, aa, ra, zn = ds
    n, aux_w = xa.shape[0], jnp.float32(cfg.aux_mean_loss_weight)
    g = jnp.float32(cfg.gamma)
    atoms = ctd_atoms(cfg)
    phis = [d.phi for d in ACTION_DISTS]

    def step(c, _):
        smog, smoc, smoga, scat, sqt, rng = c
        rng, kb, ko = jax.random.split(rng, 3)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        xb, ab, rb, znb = xa[idx], aa[idx], ra[idx], zn[idx]
        oh = jax.nn.one_hot(ab, NUM_ACTIONS, dtype=jnp.float32)
        if cfg.close_to_theory:
            om = sample_frequencies(
                ko, cfg.batch_size * cfg.num_omega, float(cfg.omega_clip),
                scale=None, distribution="pareto_1", omega_min=float(cfg.omega_min_pareto),
            ).reshape(cfg.batch_size, cfg.num_omega)
        else:
            om = sample_frequencies(
                ko, cfg.batch_size * cfg.num_omega, float(cfg.omega_clip),
                scale=float(cfg.omega_laplace_scale), distribution="half_laplacian",
            ).reshape(cfg.batch_size, cfg.num_omega)
        tgt = rb + g * znb
        gw = g * om
        fr = jnp.exp(1j * om.astype(jnp.complex64) * rb[:, None].astype(jnp.complex64))

        def ph1(ai, wi, xi):
            return jax.lax.switch(ai, phis, wi.astype(jnp.float64), xi.astype(jnp.float64))

        target_cf = fr * jax.vmap(ph1)(ab, gw, xb).astype(jnp.complex64)
        # Training CF loss matches May-4 flexibility run (no 1/2π); eval metric still uses Parseval scale.
        if cfg.close_to_theory:
            def cf_phi_mean(residual):
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2)
        else:
            w2 = jnp.maximum(jnp.square(om), jnp.float32(cfg.cf_loss_omega_eps**2))

            def cf_phi_mean(residual):
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2 / w2)

        def lphimog(pp):
            lg, mu, ls = phi_mog.apply({"params": pp}, xb, oh)
            pred = jax.vmap(lambda a, b, c, o: model_phi_return("mog", _params_mog(a, b, c), o))(lg, mu, ls, om)
            mp = _mixture_mean_mog(lg, mu, ls)
            return cf_phi_mean(pred - target_cf) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        def lphimoc(pp):
            lg, loc, lsc = phi_moc.apply({"params": pp}, xb, oh)
            pred = jax.vmap(lambda a, b, c, o: model_phi_return("moc", _params_moc(a, b, c), o))(lg, loc, lsc, om)
            # MoCauchy has no finite mean — CF only.
            return cf_phi_mean(pred - target_cf)

        def lphimg(pp):
            lg, lsh, lsc = phi_mogamma.apply({"params": pp}, xb, oh)
            pred = jax.vmap(lambda a, b, c, o: model_phi_return("mogamma", _params_mogamma(a, b, c), o))(lg, lsh, lsc, om)
            mp = _mixture_mean_mogamma(lg, lsh, lsc)
            return cf_phi_mean(pred - target_cf) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        def lphicat(pp):
            logits = phi_cat.apply({"params": pp}, xb, oh)
            probs = jax.nn.softmax(logits, -1)
            pred = jax.vmap(lambda p, o: build_categorical_cf(p, atoms, o))(probs, om)
            mexp = jnp.sum(probs * atoms, -1)
            return cf_phi_mean(pred - target_cf) + aux_w * jnp.mean(_huber(mexp - tgt, cfg.huber_kappa))

        def lphiqt(pp):
            qv = phi_qt.apply({"params": pp}, xb, oh)
            pred = jax.vmap(lambda q, o: build_quantile_cf(q, o))(qv, om)
            mq = jnp.mean(qv, -1)
            return cf_phi_mean(pred - target_cf) + aux_w * jnp.mean(_huber(mq - tgt, cfg.huber_kappa))

        losses_grads = [
            jax.value_and_grad(lphimog)(smog.params),
            jax.value_and_grad(lphimoc)(smoc.params),
            jax.value_and_grad(lphimg)(smoga.params),
            jax.value_and_grad(lphicat)(scat.params),
            jax.value_and_grad(lphiqt)(sqt.params),
        ]
        states = [smog, smoc, smoga, scat, sqt]
        new_states = [s.apply_gradients(grads=g) for s, (_, g) in zip(states, losses_grads)]
        vals = tuple(v for v, _ in losses_grads)
        return (*new_states, rng), vals

    scan_len = int(cfg.parseval_every)

    @jax.jit
    def run(smog, smoc, smoga, scat, sqt, rng):
        (smog, smoc, smoga, scat, sqt, rng), loss = jax.lax.scan(
            step, (smog, smoc, smoga, scat, sqt, rng), None, scan_len
        )
        return smog, smoc, smoga, scat, sqt, rng, loss

    return run


def _make_eval_fp(cfg, phi_mog, phi_moc, phi_mogamma, phi_cat, phi_qt):
    wm = cfg.omega_max
    te = jnp.linspace(-wm, wm, cfg.eval_num_t, jnp.float64)
    xe = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, jnp.float64)
    tp = jnp.linspace(cfg.gp_t_delta, wm, cfg.gp_num_t, jnp.float64)
    g64, rs64 = jnp.float64(cfg.gamma), jnp.float64(cfg.immediate_reward_std)
    atoms = ctd_atoms(cfg).astype(jnp.float64)
    phis = [d.phi for d in ACTION_DISTS]
    te_c = te.astype(jnp.complex128)

    def one(pm1, pm2, pm3, pc, pqt, xs, k):
        oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1, a1 = xs[None, :].astype(jnp.float32), oh[None, :]

        lg1, mu1, ls1 = phi_mog.apply({"params": pm1}, x1, a1)
        pm_mog = _params_mog(lg1[0], mu1[0], ls1[0])
        phimog = model_phi_return("mog", pm_mog, te_c)
        Fmog = model_cdf("mog", pm_mog, xe)
        pdfmog = model_pdf("mog", pm_mog, xe)

        lg2, loc2, lsc2 = phi_moc.apply({"params": pm2}, x1, a1)
        pm_moc = _params_moc(lg2[0], loc2[0], lsc2[0])
        phimoc = model_phi_return("moc", pm_moc, te_c)
        Fmoc = model_cdf("moc", pm_moc, xe)
        pdfmoc = model_pdf("moc", pm_moc, xe)

        lg3, lsh3, lsc3 = phi_mogamma.apply({"params": pm3}, x1, a1)
        pm_mg = _params_mogamma(lg3[0], lsh3[0], lsc3[0])
        phimg = model_phi_return("mogamma", pm_mg, te_c)
        Fmg = model_cdf("mogamma", pm_mg, xe)
        pdfmg = model_pdf("mogamma", pm_mg, xe)

        lc = phi_cat.apply({"params": pc}, x1, a1)[0]
        pc2 = jax.nn.softmax(lc).astype(jnp.float64)
        Fcat = discrete_return_cdf(pc2, atoms, xe)
        phicat = discrete_return_cf(pc2, atoms, te)
        pdfcat = jnp.clip(jnp.gradient(Fcat, xe), 0, None)

        qq = phi_qt.apply({"params": pqt}, x1, a1)[0].astype(jnp.float64)
        Fqt = empirical_cdf(jnp.sort(qq), xe)
        phiqt = empirical_cf_from_samples(qq.astype(jnp.complex128), te_c)
        pdfqt = jnp.clip(jnp.gradient(Fqt, xe), 0, None)

        pr = jnp.exp(-0.5 * (rs64**2) * te**2)
        pz = jax.lax.switch(k, phis, g64 * te, xs.astype(jnp.float64))
        pt = pr * pz
        ptp = jnp.exp(-0.5 * (rs64**2) * tp**2) * jax.lax.switch(k, phis, g64 * tp, xs.astype(jnp.float64))
        Ft = gil_pelaez_cdf(ptp, tp, xe)
        pdf_t = jnp.clip(jnp.gradient(Ft, xe), 0, None)

        out = {
            "w1_phi_mog": w1_cdf(Fmog, Ft, xe),
            "w1_phi_moc": w1_cdf(Fmoc, Ft, xe),
            "w1_phi_mogamma": w1_cdf(Fmg, Ft, xe),
            "w1_phi_cat": w1_cdf(Fcat, Ft, xe),
            "w1_phi_qt": w1_cdf(Fqt, Ft, xe),
            "cramer_l2_phi_mog": cramer_l2_sq_cdf(Fmog, Ft, xe),
            "cramer_l2_phi_moc": cramer_l2_sq_cdf(Fmoc, Ft, xe),
            "cramer_l2_phi_mogamma": cramer_l2_sq_cdf(Fmg, Ft, xe),
            "cramer_l2_phi_cat": cramer_l2_sq_cdf(Fcat, Ft, xe),
            "cramer_l2_phi_qt": cramer_l2_sq_cdf(Fqt, Ft, xe),
            "cf_l2_w_phi_mog": cf_l2_sq_over_omega2(phimog.astype(jnp.complex128), pt, te, cfg.gp_t_delta),
            "cf_l2_w_phi_moc": cf_l2_sq_over_omega2(phimoc.astype(jnp.complex128), pt, te, cfg.gp_t_delta),
            "cf_l2_w_phi_mogamma": cf_l2_sq_over_omega2(phimg.astype(jnp.complex128), pt, te, cfg.gp_t_delta),
            "cf_l2_w_phi_cat": cf_l2_sq_over_omega2(phicat.astype(jnp.complex128), pt, te, cfg.gp_t_delta),
            "cf_l2_w_phi_qt": cf_l2_sq_over_omega2(phiqt.astype(jnp.complex128), pt, te, cfg.gp_t_delta),
            "phi_phi_mog": phimog,
            "phi_phi_moc": phimoc,
            "phi_phi_mogamma": phimg,
            "phi_phi_cat": phicat,
            "phi_phi_qt": phiqt,
            "F_phi_mog": Fmog,
            "F_phi_moc": Fmoc,
            "F_phi_mogamma": Fmg,
            "F_phi_cat": Fcat,
            "F_phi_qt": Fqt,
            "phi_true": pt,
            "F_true": Ft,
            "pdf_true": pdf_t,
            "pdf_phi_mog": pdfmog,
            "pdf_phi_moc": pdfmoc,
            "pdf_phi_mogamma": pdfmg,
            "pdf_phi_cat": pdfcat,
            "pdf_phi_qt": pdfqt,
        }
        for m in MK:
            out[f"{m}_phitd_mog"] = out[f"{m}_phi_mog"]
            out[f"{m}_phitd_cauchy"] = out[f"{m}_phi_moc"]
            out[f"{m}_phitd_mogamma"] = out[f"{m}_phi_mogamma"]
            out[f"{m}_phitd_fcm"] = out[f"{m}_phi_cat"]
            out[f"{m}_phitd_fqm"] = out[f"{m}_phi_qt"]
        return out

    @jax.jit
    def ev(pm1, pm2, pm3, pc, pqt, xs):
        P = [one(pm1, pm2, pm3, pc, pqt, xs, kk) for kk in range(NUM_ACTIONS)]
        return {k: jnp.stack([p[k] for p in P]) for k in P[0]}

    return ev, te, xe


def evaluate_test_set_fp(cfg, nets, params, xt):
    ev, te, xe = _make_eval_fp(cfg, *nets)
    ech = jax.jit(jax.vmap(ev, (None, None, None, None, None, 0)))
    n, cs = xt.shape[0], cfg.eval_chunk_size
    assert n % cs == 0
    ch = xt.reshape(n // cs, cs, -1)
    parts = [ech(*params, ch[i]) for i in range(ch.shape[0])]
    return {k: jnp.concatenate([p[k] for p in parts]) for k in parts[0]}, np.asarray(te), np.asarray(xe)


FP_PANEL_EXPORT_KEYS = (
    "pdf_true", "phi_true", "F_true",
    "phi_phi_mog", "phi_phi_moc", "phi_phi_mogamma", "phi_phi_cat", "phi_phi_qt",
    "F_phi_mog", "F_phi_moc", "F_phi_mogamma", "F_phi_cat", "F_phi_qt",
    "pdf_phi_mog", "pdf_phi_moc", "pdf_phi_mogamma", "pdf_phi_cat", "pdf_phi_qt",
)


def _collect_eval_payload_fp(cfg, tag, er, te, xe, *, panel_index=0):
    pm = {f"{m}_{a}": np.asarray(er[f"{m}_{a}"]) for a in FP_ORDER for m in MK if f"{m}_{a}" in er}
    panel_data = {}
    for k in FP_PANEL_EXPORT_KEYS:
        if k not in er:
            continue
        arr = np.asarray(er[k])[panel_index]
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        panel_data[f"panel_{k}"] = arr.tolist()

    metric_means = {}
    ok = [a for a in FP_ORDER if all(f"{m}_{a}" in pm for m in MK)]
    for a in ok:
        metric_means[a] = np.stack([np.mean(np.asarray(pm[f"{m}_{a}"]), 0) for m in MK]).tolist()
    for new_name, old_name in LEGACY_ALGO_ALIASES_FP.items():
        if new_name in metric_means:
            metric_means[old_name] = metric_means[new_name]

    return {
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
    }, pm


def _write_metrics_w1_cramer_cf_csv(path: Path, metric_means: dict) -> None:
    """Compact CSV: algorithm × {w1, cramer_l2, cf_l2_w} × truth shorts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shorts = [d.short for d in ACTION_DISTS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "metric", *shorts])
        for algo in FP_ORDER:
            if algo not in metric_means:
                continue
            arr = np.asarray(metric_means[algo], dtype=np.float64)
            for mi, m in enumerate(MK):
                writer.writerow([algo, m, *[float(x) for x in arr[mi]]])


def _save_parseval_training_plot_fp(cfg, history, output_dir: Path):
    if not history.get("step"):
        return None
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"flexibility_parametric_family_parseval_training_phi_mog_seed{cfg.seed}"
    steps = np.asarray(history["step"], dtype=np.float64)
    y_w1 = np.asarray(history["w1_phi_mog"], dtype=np.float64)
    y_cr = np.asarray(history["cramer_l2_phi_mog"], dtype=np.float64)
    y_cf = np.asarray(history["cf_l2_w_phi_mog"], dtype=np.float64)

    configure_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(6.1, 3.5))
    ax.plot(steps, y_w1, color="#1F4E79", lw=2.0, label=r"$W_1$")
    ax.plot(steps, y_cr, color="#5E2B97", lw=2.0, label=r"Cramer $L_2^2$ (CDF)")
    ax.plot(steps, y_cf, color="#A06CD5", lw=2.0, ls="--", label=r"CF $L_2^2/\omega^2$")
    pos = np.concatenate([y_w1, y_cr, y_cf])
    pos = pos[np.isfinite(pos) & (pos > 0)]
    if pos.size:
        ax.set_yscale("log")
    style_axes_wandb_curve(ax)
    ax.set_title("Parseval anchor (φTD-Gaussian)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    png_out, pdf_out = save_figure_png_and_pdf(fig, stem, dpi_png=300, dpi_pdf=300)
    plt.close(fig)

    csv_path = stem.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "w1_phi_mog", "cramer_l2_phi_mog", "cf_l2_w_phi_mog"])
        w.writeheader()
        for i, st in enumerate(steps.tolist()):
            w.writerow({"step": st, "w1_phi_mog": y_w1[i], "cramer_l2_phi_mog": y_cr[i], "cf_l2_w_phi_mog": y_cf[i]})
    return {"png": png_out, "pdf": pdf_out, "csv": str(csv_path)}


FP_PLOT_TITLES = {
    "phi_mog": r"$\varphi\text{TD-Gaussian}$",
    "phi_moc": r"$\varphi\text{TD-Cauchy}$",
    "phi_mogamma": r"$\varphi\text{TD-Gamma}$",
    "phi_cat": r"$\varphi\text{TD-Categorical}$",
    "phi_qt": r"$\varphi\text{TD-Quantile}$",
}
FP_PLOT_COLORS = {
    "truth": "#0A0A0A",
    "phi_mog": algo_color("phitd_mog"),
    "phi_moc": algo_color("phitd_cauchy"),
    "phi_mogamma": algo_color("phitd_mogamma"),
    "phi_cat": algo_color("phitd_fcm"),
    "phi_qt": algo_color("phitd_fqm"),
}
FP_PLOT_LW = {k: (2.15 if k == "truth" else 1.12) for k in FP_PLOT_COLORS}


def _fp_plot_save_fig(fig, path: str, *, png_dpi: int = 300) -> None:
    save_figure_png_and_pdf(fig, os.path.splitext(path)[0], dpi_png=png_dpi, dpi_pdf=png_dpi, pad_inches=0.08)


def _fp_plot_metric_tables(metric_stats, row_keys, path):
    import matplotlib.pyplot as plt
    from run_distribution_analysis import PLOT_ROW_LBL

    algos = [a for a in FP_ORDER if a in metric_stats]
    means = np.stack([metric_stats[a][0] for a in algos])
    best = np.argmin(np.where(np.isfinite(means), means, np.inf), 0)
    best = np.where(~np.isfinite(means).any(0), -1, best)
    fig, axes = plt.subplots(len(algos), 1, figsize=(8.3, 2.0 * len(algos) + 0.7))
    if len(algos) == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.95, bottom=0.04, hspace=0.35)
    cols = [d.name for d in ACTION_DISTS]
    row_labels = [PLOT_ROW_LBL.get(k, k) for k in row_keys]
    for i, (ax, algo) in enumerate(zip(axes, algos)):
        ax.axis("off")
        cells = [[f"{means[i, r, c]:.4g}" for c in range(len(ACTION_DISTS))] for r in range(len(row_keys))]
        tbl = ax.table(cellText=cells, rowLabels=row_labels, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.08, 1.35)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0 or col < 0:
                cell.set_text_props(fontweight="bold")
            cell.set_edgecolor("#CCCCCC")
            if row >= 1 and col >= 0 and best[row - 1, col] == i:
                cell.set_facecolor("#D2F4D9")
        ax.text(0.5, 1.02, FP_PLOT_TITLES[algo], transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
    _fp_plot_save_fig(fig, path)
    plt.close(fig)


def _fp_plot_smooth_curve(y, passes=2):
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 5 or passes <= 0:
        return arr
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    out = arr
    for _ in range(passes):
        out = np.convolve(np.pad(out, (2, 2), mode="edge"), kernel, mode="valid")
    return out


def _fp_plot_panels(mean_p, se_p, te, xe, cf_hw, path):
    import matplotlib.pyplot as plt

    series = (
        (("panel_phi_true", "panel_pdf_true", "panel_F_true"), "truth", "Truth"),
        (("panel_phi_phi_mog", "panel_pdf_phi_mog", "panel_F_phi_mog"), "phi_mog", FP_PLOT_TITLES["phi_mog"]),
        (("panel_phi_phi_moc", "panel_pdf_phi_moc", "panel_F_phi_moc"), "phi_moc", FP_PLOT_TITLES["phi_moc"]),
        (("panel_phi_phi_mogamma", "panel_pdf_phi_mogamma", "panel_F_phi_mogamma"), "phi_mogamma", FP_PLOT_TITLES["phi_mogamma"]),
        (("panel_phi_phi_cat", "panel_pdf_phi_cat", "panel_F_phi_cat"), "phi_cat", FP_PLOT_TITLES["phi_cat"]),
        (("panel_phi_phi_qt", "panel_pdf_phi_qt", "panel_F_phi_qt"), "phi_qt", FP_PLOT_TITLES["phi_qt"]),
    )
    fig, axes = plt.subplots(len(ACTION_DISTS), 4, figsize=(13.5, 2.45 * len(ACTION_DISTS)))
    for row, dist in enumerate(ACTION_DISTS):
        axd, axcf, axp, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        tmask = np.abs(te) <= cf_hw
        if "panel_pdf_true" in mean_p:
            axd.plot(xe[xmask], mean_p["panel_pdf_true"][row][xmask], color=FP_PLOT_COLORS["truth"], lw=FP_PLOT_LW["truth"])
        style_axes_panel(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axd.set_title("Density")
            axcf.set_title(r"Re $\varphi$")
            axp.set_title("PDF")
            axc.set_title("CDF")
        for (phi_k, pdf_k, f_k), ckey, label in series:
            if phi_k in mean_p:
                y = np.real(mean_p[phi_k][row])
                axcf.plot(te[tmask], y[tmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
            if pdf_k in mean_p:
                y = mean_p[pdf_k][row] if ckey == "truth" else _fp_plot_smooth_curve(mean_p[pdf_k][row])
                axp.plot(xe[xmask], y[xmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
            if f_k in mean_p:
                axc.plot(xe[xmask], mean_p[f_k][row][xmask], color=FP_PLOT_COLORS[ckey], lw=FP_PLOT_LW[ckey], label=label)
        for ax in (axcf, axp, axc):
            style_axes_panel(ax)
        axcf.set_xlim(-cf_hw, cf_hw)
        axp.set_xlim(dist.plot_x_min, dist.plot_x_max)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)
    axes[0, 3].legend(loc="lower right", frameon=False, fontsize=6, ncol=1)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.08, wspace=0.24, hspace=0.22)
    _fp_plot_save_fig(fig, path)
    plt.close(fig)


def plot_flexibility_parametric_family_local(payloads, output_dir: Path):
    if not payloads:
        raise ValueError("No payloads to plot.")
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_keys = tuple(MK)
    metric_stats = {}
    for algo in FP_ORDER:
        mats = []
        for payload in payloads:
            mm = payload.get("metric_means", {})
            key = algo if algo in mm else LEGACY_ALGO_ALIASES_FP.get(algo)
            if key is None or key not in mm:
                continue
            mats.append(np.asarray(mm[key], dtype=np.float64))
        if mats:
            stack = np.stack(mats, 0)
            mean = np.nanmean(stack, 0)
            se = np.zeros_like(mean) if stack.shape[0] < 2 else np.nanstd(stack, 0, ddof=1) / np.sqrt(stack.shape[0])
            metric_stats[algo] = (mean, se)

    t_eval = np.asarray(payloads[0]["t_eval"], dtype=np.float64)
    x_eval = np.asarray(payloads[0]["x_eval"], dtype=np.float64)
    omega_max = float(payloads[0]["omega_max"])
    panel_mean, panel_se = {}, {}
    for key in sorted(payloads[0].get("panel", {})):
        vals = [np.asarray(p["panel"][key], dtype=np.float64) for p in payloads if key in p.get("panel", {})]
        if not vals:
            continue
        stack = np.stack(vals, 0)
        panel_mean[key] = np.nanmean(stack, 0)
        panel_se[key] = np.zeros_like(panel_mean[key]) if stack.shape[0] < 2 else np.nanstd(stack, 0, ddof=1) / np.sqrt(stack.shape[0])

    metrics_png = str(output_dir / "flexibility_parametric_family_metrics_combined.png")
    panels_png = str(output_dir / "flexibility_parametric_family_panels.png")
    metrics_csv = output_dir / "flexibility_parametric_family_metrics_combined.csv"
    w1cf_csv = output_dir / "flexibility_parametric_family_metrics_w1_cramer_cf.csv"

    _fp_plot_metric_tables(metric_stats, row_keys, metrics_png)
    _fp_plot_panels(panel_mean, panel_se, t_eval, x_eval, omega_max, panels_png)

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "metric"] + [f"{x.short}_mean" for x in ACTION_DISTS] + [f"{x.short}_se" for x in ACTION_DISTS])
        for algo in FP_ORDER:
            if algo not in metric_stats:
                continue
            mm, sm = metric_stats[algo]
            for ri, rk in enumerate(row_keys):
                w.writerow([algo, rk, *[float(x) for x in mm[ri]], *[float(x) for x in sm[ri]]])

    means_only = {a: metric_stats[a][0] for a in FP_ORDER if a in metric_stats}
    _write_metrics_w1_cramer_cf_csv(w1cf_csv, means_only)

    return {
        "metrics_png": metrics_png,
        "panels_png": panels_png,
        "metrics_csv": str(metrics_csv),
        "w1_cramer_cf_csv": str(w1cf_csv),
        "metrics_pdf": str(pdf_path_for_png_stem(Path(metrics_png).with_suffix(""))),
        "panels_pdf": str(pdf_path_for_png_stem(Path(panels_png).with_suffix(""))),
    }


def train_and_export_fp(cfg, tag, wb, *, figures_run_dir: Path, skip_plots: bool = False):
    figures_run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(figures_run_dir / "flexibility_parametric_family_experiment_config.yaml", _config_for_logging(cfg))

    rng = jax.random.PRNGKey(cfg.seed)
    ri, rd, rt, ru = jax.random.split(rng, 4)
    print(f"Dataset... offset={cfg.dataset_seed_offset}", flush=True)
    print("Algos: MoG / MoC / MoGamma / Categorical / Quantile", flush=True)
    print("Truths: Gauss / Cauchy / MoG / Gam", flush=True)
    ds = build_dataset(jax.random.fold_in(rd, cfg.dataset_seed_offset), cfg)
    jax.tree.map(lambda x: x.block_until_ready(), ds)

    smog, smoc, smoga, scat, sqt, *nets = create_train_states_fp(ri, cfg)
    parseval_eval, _, _ = _make_eval_fp(cfg, *nets)
    anchor_x = jax.random.normal(_panel_anchor_prng_key(), (cfg.state_dim,), jnp.float32)
    parseval_history = {"step": [], "w1_phi_mog": [], "cramer_l2_phi_mog": [], "cf_l2_w_phi_mog": []}
    run = make_train_chunk_fp(cfg, *nets, ds)
    pe = max(1, min(int(cfg.parseval_every), int(cfg.total_steps)))
    if int(cfg.total_steps) % pe != 0:
        raise ValueError(f"total_steps ({cfg.total_steps}) must be divisible by parseval_every ({pe})")
    nc = int(cfg.total_steps) // pe
    t0 = time.perf_counter()
    rng = ru
    last_losses = None
    for c in range(nc):
        tc = time.perf_counter()
        smog, smoc, smoga, scat, sqt, rng, ls = run(smog, smoc, smoga, scat, sqt, rng)
        jax.tree.map(lambda x: x.block_until_ready(), ls)
        st = (c + 1) * pe
        last_losses = {
            "train/loss_phi_mog_final": float(ls[0][-1]),
            "train/loss_phi_moc_final": float(ls[1][-1]),
            "train/loss_phi_mogamma_final": float(ls[2][-1]),
            "train/loss_phi_cat_final": float(ls[3][-1]),
            "train/loss_phi_qt_final": float(ls[4][-1]),
        }
        if st % int(cfg.log_every) == 0 or st == int(cfg.total_steps) or c == 0:
            print(
                f"  {st:>7}  mog={last_losses['train/loss_phi_mog_final']:.4f} "
                f"moc={last_losses['train/loss_phi_moc_final']:.4f} "
                f"mg={last_losses['train/loss_phi_mogamma_final']:.4f} "
                f"cat={last_losses['train/loss_phi_cat_final']:.4f} "
                f"qt={last_losses['train/loss_phi_qt_final']:.4f}  {time.perf_counter()-tc:.1f}s",
                flush=True,
            )
        pv = parseval_eval(smog.params, smoc.params, smoga.params, scat.params, sqt.params, anchor_x)
        jax.tree.map(lambda x: x.block_until_ready(), pv)
        parseval_history["step"].append(float(st))
        parseval_history["w1_phi_mog"].append(float(jnp.mean(pv["w1_phi_mog"])))
        parseval_history["cramer_l2_phi_mog"].append(float(jnp.mean(pv["cramer_l2_phi_mog"])))
        parseval_history["cf_l2_w_phi_mog"].append(float(jnp.mean(pv["cf_l2_w_phi_mog"])))

    print(f"train {time.perf_counter()-t0:.1f}s", flush=True)
    if not skip_plots:
        _save_parseval_training_plot_fp(cfg, parseval_history, figures_run_dir)

    xt = jax.random.fold_in(rt, cfg.test_seed_offset)
    xv = jax.random.normal(xt, (cfg.n_test, cfg.state_dim), jnp.float32).at[0].set(
        jax.random.normal(_panel_anchor_prng_key(), (cfg.state_dim,), jnp.float32)
    )
    er, te, xe = evaluate_test_set_fp(
        cfg, tuple(nets), (smog.params, smoc.params, smoga.params, scat.params, sqt.params), xv
    )
    jax.tree.map(lambda x: x.block_until_ready(), er)
    payload, pm = _collect_eval_payload_fp(cfg, tag, er, te, xe, panel_index=0)

    # Always write the compact W1/Cramer/CF CSV.
    _write_metrics_w1_cramer_cf_csv(
        figures_run_dir / "flexibility_parametric_family_metrics_w1_cramer_cf.csv",
        payload["metric_means"],
    )

    if not skip_plots:
        try:
            plot_paths = plot_flexibility_parametric_family_local([payload], figures_run_dir)
            print(f"Saved figures: {plot_paths}", flush=True)
        except Exception as exc:
            print(f"warning: post-run plotting failed: {exc}", flush=True)


def _default_config_yaml_path() -> Path:
    return _artifact_purejaxql_dir() / "config" / "analysis" / "flexibility_parametric_family.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Flexibility: MoG/MoC/MoGamma/Cat/Quant on Gauss/Cauchy/MoG/Gam.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--figures-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--experiment-tag", default="FlexParamFamily")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--parseval-every", type=int, default=None)
    p.add_argument("--dataset-size", type=int)
    p.add_argument("--omega-laplace-scale", type=float)
    p.add_argument("--num-qtd-quantiles", type=int)
    p.add_argument("--num-ctd-atoms", type=int)
    p.add_argument("--num-mixture-components", type=int, dest="num_dirac_components")
    p.add_argument("--num-dirac-components", type=int, dest="num_dirac_components", help=argparse.SUPPRESS)
    p.add_argument("--close-to-theory", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--omega-min-pareto", type=float, default=None)
    p.add_argument("--aux-mean-loss-weight", type=float, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    figures_root = args.figures_dir if args.figures_dir is not None else (_REPO_ROOT / "figures" / "flexibility_parametric_family")
    run_dir = figures_root / dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
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
        "close_to_theory": args.close_to_theory,
        "omega_min_pareto": args.omega_min_pareto,
        "aux_mean_loss_weight": args.aux_mean_loss_weight,
    }
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
