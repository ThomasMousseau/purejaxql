"""Phi-TD on Craftax with the same LSTM rollout/trunk as ``distributional_pqn_rnn_craftax``.

Uses **only** the characteristic-function TD(0) Bellman loss (no TD(λ) / PQN mean auxiliary),
matching ``phi_td_pqn_minatar.py``. For apples-to-apples benchmarks vs CTD/QTD:

  - ``FAMILY_DISTRIBUTION: categorical`` ↔ C51-style head + categorical CF (same support / projection
    algebra as ``ALG_VARIANT: ctd``).
  - ``FAMILY_DISTRIBUTION: quantile`` ↔ QR-DQN-style quantile head + empirical CF (same bootstrap
    structure as ``ALG_VARIANT: qtd``).

Other families (dirac, mog, cauchy, gamma, laplace, logistic) follow the same flat-batch CF mapping
as MinAtar Phi-TD, with ``(s,a)`` heads fed by the RNN latent at ``t`` and targets from ``t+1``.
"""

from __future__ import annotations

import copy
import os
import time
from functools import partial
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from omegaconf import OmegaConf

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.distributional_pqn_rnn_craftax import (
    ScannedRNN,
    Transition,
    apply_norm,
    project_categorical,
    select_action_values,
)
from purejaxql.utils.batch_renorm import BatchRenorm
from purejaxql.utils.craftax_wrappers import (
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from purejaxql.utils.mog_cf import (
    build_categorical_cf,
    build_cauchy_mixture_cf,
    build_dirac_mixture_cf,
    build_gamma_mixture_cf,
    build_laplace_mixture_cf,
    build_logistic_mixture_cf,
    build_mog_cf,
    build_quantile_cf,
    gamma_mixture_q_values,
    mog_q_values,
    sample_frequencies,
)


def phi_cf_flat_batch(
    out_cur: dict[str, jnp.ndarray],
    out_next: dict[str, jnp.ndarray],
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    actions: jnp.ndarray,
    omegas: jnp.ndarray,
    family: str,
    support: jnp.ndarray | None,
    cfg: dict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Flat batch of transitions (N = (T−1)·N_envs). Matches ``phi_td_pqn_minatar.phi_gradient_step``."""
    bsz = rewards.shape[0]
    bidx = jnp.arange(bsz)
    gam = float(cfg["GAMMA"])
    v_min = float(cfg.get("V_MIN", 0.0))
    v_max = float(cfg.get("V_MAX", 200.0))

    if family == "dirac":
        q_next = mog_q_values(out_next["pi"], out_next["mu"])
        next_actions = jnp.argmax(q_next, axis=-1)
        pi_sel = out_next["pi"][bidx, next_actions]
        mu_sel = out_next["mu"][bidx, next_actions]
        gammas = gam * (1.0 - dones)
        bellman_mu = rewards[:, None] + gammas[:, None] * mu_sel
        bellman_pi = pi_sel
        td_target = build_dirac_mixture_cf(
            bellman_pi[:, None, :],
            bellman_mu[:, None, :],
            omegas,
        )
        td_target = td_target[:, :, 0]
        online_phi = build_dirac_mixture_cf(out_cur["pi"], out_cur["mu"], omegas)
        online_phi_sel = online_phi[bidx, :, actions]
    elif family == "categorical":
        assert support is not None
        next_probs = jax.nn.softmax(out_next["logits"], axis=-1)
        next_q = jnp.sum(next_probs * support, axis=-1)
        next_actions = jnp.argmax(next_q, axis=-1)
        next_probs_sel = next_probs[bidx, next_actions]
        target_probs = project_categorical(
            next_probs_sel,
            rewards,
            dones.astype(jnp.float32),
            support,
            gam,
            v_min,
            v_max,
        )
        td_target = build_categorical_cf(jax.lax.stop_gradient(target_probs), support, omegas)
        probs = jax.nn.softmax(out_cur["logits"], axis=-1)
        sel_probs = select_action_values(probs, actions)
        online_phi_sel = build_categorical_cf(sel_probs, support, omegas)
    elif family == "quantile":
        nq = out_next["quantiles"]
        next_q = jnp.mean(nq, axis=-1)
        next_actions = jnp.argmax(next_q, axis=-1)
        sel_nq = nq[bidx, next_actions]
        target_q = rewards[:, None] + gam * (1.0 - dones.astype(jnp.float32)[:, None]) * sel_nq
        td_target = build_quantile_cf(jax.lax.stop_gradient(target_q), omegas)
        sel_q = select_action_values(out_cur["quantiles"], actions)
        online_phi_sel = build_quantile_cf(sel_q, omegas)
    elif family == "mog":
        q_next = mog_q_values(out_next["pi"], out_next["mu"])
        next_actions = jnp.argmax(q_next, axis=-1)
        pi_sel = out_next["pi"][bidx, next_actions]
        mu_sel = out_next["mu"][bidx, next_actions]
        sig_sel = out_next["sigma"][bidx, next_actions]
        gammas = gam * (1.0 - dones)
        bellman_mu = rewards[:, None] + gammas[:, None] * mu_sel
        bellman_sigma = gammas[:, None] * sig_sel
        td_target = build_mog_cf(
            pi_sel[:, None, :],
            bellman_mu[:, None, :],
            bellman_sigma[:, None, :],
            omegas,
        )
        td_target = td_target[:, :, 0]
        online_phi = build_mog_cf(out_cur["pi"], out_cur["mu"], out_cur["sigma"], omegas)
        online_phi_sel = online_phi[bidx, :, actions]
    elif family == "cauchy":
        q_next = mog_q_values(out_next["pi"], out_next["loc"])
        next_actions = jnp.argmax(q_next, axis=-1)
        pi_sel = out_next["pi"][bidx, next_actions]
        loc_sel = out_next["loc"][bidx, next_actions]
        scale_sel = out_next["scale"][bidx, next_actions]
        gammas = gam * (1.0 - dones)
        bellman_loc = rewards[:, None] + gammas[:, None] * loc_sel
        bellman_scale = gammas[:, None] * scale_sel
        td_target = build_cauchy_mixture_cf(
            pi_sel[:, None, :],
            bellman_loc[:, None, :],
            bellman_scale[:, None, :],
            omegas,
        )
        td_target = td_target[:, :, 0]
        online_phi = build_cauchy_mixture_cf(out_cur["pi"], out_cur["loc"], out_cur["scale"], omegas)
        online_phi_sel = online_phi[bidx, :, actions]
    elif family == "gamma":
        q_next = gamma_mixture_q_values(out_next["pi"], out_next["shape"], out_next["scale"])
        next_actions = jnp.argmax(q_next, axis=-1)
        pi_sel = out_next["pi"][bidx, next_actions]
        shape_sel = out_next["shape"][bidx, next_actions]
        scale_sel = out_next["scale"][bidx, next_actions]
        gammas = gam * (1.0 - dones)
        bellman_scale = gammas[:, None] * scale_sel
        phase_r = jnp.exp(1j * rewards[:, None] * omegas[None, :])
        mix_cf = build_gamma_mixture_cf(
            pi_sel[:, None, :],
            shape_sel[:, None, :],
            bellman_scale[:, None, :],
            omegas,
        )
        td_target = phase_r * mix_cf[:, :, 0]
        online_phi = build_gamma_mixture_cf(
            out_cur["pi"], out_cur["shape"], out_cur["scale"], omegas
        )
        online_phi_sel = online_phi[bidx, :, actions]
    elif family == "laplace":
        q_next = mog_q_values(out_next["pi"], out_next["loc"])
        next_actions = jnp.argmax(q_next, axis=-1)
        pi_sel = out_next["pi"][bidx, next_actions]
        loc_sel = out_next["loc"][bidx, next_actions]
        scale_sel = out_next["scale"][bidx, next_actions]
        gammas = gam * (1.0 - dones)
        bellman_loc = rewards[:, None] + gammas[:, None] * loc_sel
        bellman_scale = gammas[:, None] * scale_sel
        td_target = build_laplace_mixture_cf(
            pi_sel[:, None, :],
            bellman_loc[:, None, :],
            bellman_scale[:, None, :],
            omegas,
        )
        td_target = td_target[:, :, 0]
        online_phi = build_laplace_mixture_cf(out_cur["pi"], out_cur["loc"], out_cur["scale"], omegas)
        online_phi_sel = online_phi[bidx, :, actions]
    elif family == "logistic":
        q_next = mog_q_values(out_next["pi"], out_next["loc"])
        next_actions = jnp.argmax(q_next, axis=-1)
        pi_sel = out_next["pi"][bidx, next_actions]
        loc_sel = out_next["loc"][bidx, next_actions]
        scale_sel = out_next["scale"][bidx, next_actions]
        gammas = gam * (1.0 - dones)
        bellman_loc = rewards[:, None] + gammas[:, None] * loc_sel
        bellman_scale = gammas[:, None] * scale_sel
        td_target = build_logistic_mixture_cf(
            pi_sel[:, None, :],
            bellman_loc[:, None, :],
            bellman_scale[:, None, :],
            omegas,
        )
        td_target = td_target[:, :, 0]
        online_phi = build_logistic_mixture_cf(
            out_cur["pi"], out_cur["loc"], out_cur["scale"], omegas
        )
        online_phi_sel = online_phi[bidx, :, actions]
    else:
        raise ValueError(f"Unknown FAMILY_DISTRIBUTION: {family}")

    return online_phi_sel, td_target


class RNNPhiTDQNetwork(nn.Module):
    """Same MLP+LSTM trunk as ``RNNDistributionalQNetwork`` (without IQN); Phi heads like MinAtar."""

    family_distribution: str
    action_dim: int
    num_atoms: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    add_last_action: bool = False

    @nn.compact
    def encode_latent(self, hidden, obs, done, last_action, train: bool = False):
        x = obs
        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            _ = BatchRenorm(use_running_average=not train)(x)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = apply_norm(x, self.norm_type, train)
            x = nn.relu(x)
        if self.add_last_action:
            la = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, la], axis=-1)
        new_hidden = []
        for i in range(self.num_rnn_layers):
            hidden_aux, x = ScannedRNN()(hidden[i], (x, done))
            new_hidden.append(hidden_aux)
        return new_hidden, x

    @nn.compact
    def _decode_flat(self, latent: jnp.ndarray) -> dict[str, jnp.ndarray]:
        a, m = self.action_dim, self.num_atoms
        fam = str(self.family_distribution).lower()
        if fam == "dirac":
            pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
            pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
            mu = nn.Dense(a * m, name="head_mu")(latent).reshape(-1, a, m)
            return {"pi": pi, "mu": mu}
        if fam == "categorical":
            logits = nn.Dense(
                a * m,
                kernel_init=nn.initializers.zeros,
                name="head_categorical_logits",
            )(latent)
            return {"logits": logits.reshape(-1, a, m)}
        if fam == "quantile":
            qv = nn.Dense(
                a * m,
                kernel_init=nn.initializers.zeros,
                name="head_quantiles",
            )(latent)
            return {"quantiles": qv.reshape(-1, a, m)}
        if fam == "mog":
            pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
            pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
            mu = nn.Dense(a * m, name="head_mu")(latent).reshape(-1, a, m)
            sigma = jax.nn.softplus(nn.Dense(a * m, name="head_sigma")(latent).reshape(-1, a, m))
            return {"pi": pi, "mu": mu, "sigma": sigma}
        if fam == "cauchy":
            pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
            pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
            loc = nn.Dense(a * m, name="head_loc")(latent).reshape(-1, a, m)
            scale = jax.nn.softplus(
                nn.Dense(a * m, name="head_cauchy_scale")(latent).reshape(-1, a, m)
            )
            return {"pi": pi, "loc": loc, "scale": scale}
        if fam == "gamma":
            pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
            pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
            shape = jax.nn.softplus(
                nn.Dense(a * m, name="head_gamma_shape")(latent).reshape(-1, a, m)
            )
            scale = jax.nn.softplus(
                nn.Dense(a * m, name="head_gamma_scale")(latent).reshape(-1, a, m)
            )
            return {"pi": pi, "shape": shape, "scale": scale}
        if fam == "laplace":
            pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
            pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
            loc = nn.Dense(a * m, name="head_laplace_loc")(latent).reshape(-1, a, m)
            scale = jax.nn.softplus(
                nn.Dense(a * m, name="head_laplace_scale")(latent).reshape(-1, a, m)
            )
            return {"pi": pi, "loc": loc, "scale": scale}
        if fam == "logistic":
            pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
            pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
            loc = nn.Dense(a * m, name="head_logistic_loc")(latent).reshape(-1, a, m)
            scale = jax.nn.softplus(
                nn.Dense(a * m, name="head_logistic_scale")(latent).reshape(-1, a, m)
            )
            return {"pi": pi, "loc": loc, "scale": scale}
        raise ValueError(f"Unknown family_distribution: {self.family_distribution}")

    @nn.compact
    def __call__(self, hidden, obs, done, last_action, train: bool = False):
        new_hidden, latent = self.encode_latent(hidden, obs, done, last_action, train=train)
        t, b, d = latent.shape
        flat = latent.reshape(t * b, d)
        flat_out = self._decode_flat(flat)
        out = jax.tree_util.tree_map(lambda x: x.reshape(t, b, *x.shape[1:]), flat_out)
        return new_hidden, out

    def q_values(
        self,
        hidden,
        obs,
        done,
        last_action,
        train: bool = False,
        support: jnp.ndarray | None = None,
    ):
        new_hidden, out = self(hidden, obs, done, last_action, train=train)
        fam = str(self.family_distribution).lower()
        if fam == "dirac":
            q_vals = mog_q_values(out["pi"], out["mu"])
        elif fam == "mog":
            q_vals = mog_q_values(out["pi"], out["mu"])
        elif fam == "categorical":
            assert support is not None
            probs = jax.nn.softmax(out["logits"], axis=-1)
            q_vals = jnp.sum(probs * support, axis=-1)
        elif fam == "cauchy":
            q_vals = mog_q_values(out["pi"], out["loc"])
        elif fam == "gamma":
            q_vals = gamma_mixture_q_values(out["pi"], out["shape"], out["scale"])
        elif fam in ("laplace", "logistic"):
            q_vals = mog_q_values(out["pi"], out["loc"])
        elif fam == "quantile":
            q_vals = jnp.mean(out["quantiles"], axis=-1)
        else:
            q_vals = jnp.mean(out["quantiles"], axis=-1)
        return new_hidden, q_vals, out

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def _metrics_from_out(
    out: dict[str, jnp.ndarray],
    family: str,
    support: jnp.ndarray | None,
) -> dict[str, jnp.ndarray]:
    """Scalars for logging (mean over leading two dims T,B)."""
    fam = family.lower()
    flat_in = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), out)
    if fam == "dirac":
        q_all = mog_q_values(flat_in["pi"], flat_in["mu"])
    elif fam == "mog":
        q_all = mog_q_values(flat_in["pi"], flat_in["mu"])
    elif fam == "categorical":
        assert support is not None
        probs = jax.nn.softmax(flat_in["logits"], axis=-1)
        q_all = jnp.sum(probs * support, axis=-1)
    elif fam == "cauchy":
        q_all = mog_q_values(flat_in["pi"], flat_in["loc"])
    elif fam == "gamma":
        q_all = gamma_mixture_q_values(flat_in["pi"], flat_in["shape"], flat_in["scale"])
    elif fam in ("laplace", "logistic"):
        q_all = mog_q_values(flat_in["pi"], flat_in["loc"])
    else:
        q_all = jnp.mean(flat_in["quantiles"], axis=-1)
    q_gap = (q_all.max(axis=-1) - q_all.min(axis=-1)).mean()
    if fam in ("dirac", "mog", "cauchy", "gamma", "laplace", "logistic"):
        ent = -jnp.sum(flat_in["pi"] * jnp.log(flat_in["pi"] + 1e-8), axis=-1).mean()
    elif fam == "categorical":
        ent = -jnp.sum(
            jax.nn.softmax(flat_in["logits"], -1) * jax.nn.log_softmax(flat_in["logits"], -1),
            axis=-1,
        ).mean()
    else:
        ent = jnp.array(0.0, dtype=jnp.float32)
    return {"q_gap": q_gap, "dist_entropy": ent}


def make_train(config: dict):
    config = dict(config)
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = int(
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, (
        "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"
    )

    max_grad_norm = config["MAX_GRAD_NORM"]
    per_layer_gradient_clipping = config.get("PER_LAYER_GRADIENT_CLIPPING", False)

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])
    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    family = str(config.get("FAMILY_DISTRIBUTION", "categorical")).lower()
    num_atoms = int(config["NUM_ATOMS"])
    if family == "categorical":
        support = jnp.linspace(
            float(config.get("V_MIN", 0.0)),
            float(config.get("V_MAX", 200.0)),
            num_atoms,
        )
    else:
        support = None

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        return jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(rng_a, greedy_actions.shape, 0, q_vals.shape[-1]),
            greedy_actions,
        )

    def get_layer_grad_norms(grads):
        flat_grads = flatten_dict(grads)
        layer_sq_norms = {}
        for key, value in flat_grads.items():
            layer_key = "/".join(str(k) for k in key[:-1]) if len(key) > 1 else str(key[0])
            sq_norm = jnp.sum(jnp.square(value))
            if layer_key in layer_sq_norms:
                layer_sq_norms[layer_key] = layer_sq_norms[layer_key] + sq_norm
            else:
                layer_sq_norms[layer_key] = sq_norm
        return {k: jnp.sqrt(v) for k, v in layer_sq_norms.items()}

    def clip_grads_per_layer(grads):
        flat_grads = flatten_dict(grads)
        layer_sq_norms = {}
        for key, value in flat_grads.items():
            layer_key = key[:-1] if len(key) > 1 else key
            sq_norm = jnp.sum(jnp.square(value))
            if layer_key in layer_sq_norms:
                layer_sq_norms[layer_key] = layer_sq_norms[layer_key] + sq_norm
            else:
                layer_sq_norms[layer_key] = sq_norm

        max_norm = jnp.asarray(max_grad_norm, dtype=jnp.float32)
        eps = 1e-6
        layer_scales = {
            layer_key: jnp.minimum(1.0, max_norm / (jnp.sqrt(sq_norm) + eps))
            for layer_key, sq_norm in layer_sq_norms.items()
        }

        clipped_flat_grads = {}
        for key, value in flat_grads.items():
            layer_key = key[:-1] if len(key) > 1 else key
            clipped_flat_grads[key] = value * layer_scales[layer_key]

        clipped_grads = unflatten_dict(clipped_flat_grads)
        if isinstance(grads, FrozenDict):
            return freeze(clipped_grads)
        return clipped_grads

    def train(rng, seed_idx):
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            config["EPS_DECAY"] * config["NUM_UPDATES_DECAY"],
        )
        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(
                config["NUM_UPDATES_DECAY"] * config["NUM_MINIBATCHES"] * config["NUM_EPOCHS"]
            ),
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        network = RNNPhiTDQNetwork(
            family_distribution=family,
            action_dim=action_dim,
            num_atoms=num_atoms,
            hidden_size=config.get("HIDDEN_SIZE", 512),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
        )

        clip_transform = (
            optax.identity()
            if per_layer_gradient_clipping
            else optax.clip_by_global_norm(max_grad_norm)
        )
        tx = optax.chain(
            clip_transform,
            optax.radam(learning_rate=lr),
        )

        def create_agent(rng_init):
            init_x = (
                jnp.zeros((1, 1, *obs_shape)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            )
            init_hs = network.initialize_carry(1)
            variables = network.init(rng_init, init_hs, *init_x, train=False)
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=tx,
            )

        rng, rng_agent = jax.random.split(rng)
        train_state = create_agent(rng_agent)

        def _update_step(runner_state, unused):
            train_state, memory_transitions, expl_state, test_metrics, rng = runner_state

            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                obs = last_obs[np.newaxis]
                done = last_done[np.newaxis]
                la = last_action[np.newaxis]
                new_hs, q_vals, _ = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    hs,
                    obs,
                    done,
                    la,
                    train=False,
                    support=support,
                    method=RNNPhiTDQNetwork.q_values,
                )
                q_vals = q_vals.squeeze(axis=0)
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, action, env_params
                )
                transition = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    action=action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_vals=q_vals,
                )
                return (new_hs, new_obs, new_done, action, new_env_state, rng), (
                    transition,
                    info,
                )

            rng, rng_rollout = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env, (*expl_state, rng_rollout), None, config["NUM_STEPS"]
            )
            expl_state = tuple(expl_state)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            memory_transitions = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):
                    train_state, rng = carry
                    rng, rng_omega = jax.random.split(rng)
                    omegas = sample_frequencies(
                        rng_omega,
                        int(config["NUM_OMEGA_SAMPLES"]),
                        float(config["OMEGA_MAX"]),
                        distribution=config.get("OMEGA_SAMPLING_DISTRIBUTION", "pareto_1"),
                        omega_min=config.get("OMEGA_MIN"),
                    )

                    def _loss_fn(params):
                        variables = {"params": params, "batch_stats": train_state.batch_stats}
                        hs = jax.tree_util.tree_map(lambda x: x[0], minibatch.last_hs)
                        agent_in = (
                            minibatch.obs,
                            minibatch.last_done,
                            minibatch.last_action,
                        )
                        (_hs, out), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(variables, hs, *agent_in)
                        out_next_sg = jax.tree_util.tree_map(
                            lambda z: jax.lax.stop_gradient(z[1:]), out
                        )
                        out_cur = jax.tree_util.tree_map(lambda z: z[:-1], out)
                        t, b = minibatch.reward.shape
                        n_flat = (t - 1) * b
                        out_cur_f = jax.tree_util.tree_map(
                            lambda z: z.reshape(n_flat, *z.shape[2:]), out_cur
                        )
                        out_next_f = jax.tree_util.tree_map(
                            lambda z: z.reshape(n_flat, *z.shape[2:]), out_next_sg
                        )
                        rewards_f = minibatch.reward[:-1].reshape(n_flat)
                        dones_f = minibatch.done[:-1].reshape(n_flat)
                        actions_f = minibatch.action[:-1].reshape(n_flat)
                        online_phi_sel, td_target = phi_cf_flat_batch(
                            out_cur_f,
                            out_next_f,
                            rewards_f,
                            dones_f,
                            actions_f,
                            omegas,
                            family,
                            support,
                            config,
                        )
                        td_target = jax.lax.stop_gradient(td_target)
                        residual = online_phi_sel - td_target
                        err_squared = jnp.abs(residual) ** 2
                        imag_abs = jnp.abs(jnp.imag(residual))
                        if config.get("IS_DIVIDED_BY_OMEGA_SQUARED", False):
                            omega_weights = (1.0 / jnp.maximum(omegas, 1e-8) ** 2)[None, :]
                            err_squared = err_squared * omega_weights
                            imag_abs = imag_abs * omega_weights
                        cf_loss = jnp.mean(err_squared)
                        cf_im = jnp.mean(imag_abs)
                        extra_m = _metrics_from_out(out, family, support)
                        return cf_loss, (updates, cf_im, cf_loss, extra_m)

                    (loss, (updates, cf_im, cf_loss, extra_m)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    grad_norm = optax.global_norm(grads)
                    if config.get("LOG_LAYER_GRAD_NORMS", False):
                        layer_grad_norms = get_layer_grad_norms(grads)
                    else:
                        layer_grad_norms = {}
                    if per_layer_gradient_clipping:
                        grads = clip_grads_per_layer(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (
                        loss,
                        cf_im,
                        cf_loss,
                        grad_norm,
                        extra_m,
                        layer_grad_norms,
                    )

                def preprocess_transition(x, rng_perm):
                    x = jax.random.permutation(rng_perm, x, axis=1)
                    x = x.reshape(x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:])
                    return jnp.swapaxes(x, 0, 1)

                rng, rng_perm = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda z: preprocess_transition(z, rng_perm), memory_transitions
                )
                rng, rng_scan = jax.random.split(rng)
                (train_state, rng), scan_out = jax.lax.scan(
                    _learn_phase, (train_state, rng_scan), minibatches
                )
                return (train_state, rng), scan_out

            rng, rng_epochs = jax.random.split(rng)
            (train_state, rng), epoch_metrics = jax.lax.scan(
                _learn_epoch, (train_state, rng_epochs), None, config["NUM_EPOCHS"]
            )
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            loss, cf_im, cf_loss, grad_norm, extra_m, layer_grad_norms = epoch_metrics
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "total_loss": loss.mean(),
                "phi_cf_loss": cf_loss.mean(),
                "cf_loss_imag": cf_im.mean(),
                "grad_norm": grad_norm.mean(),
                "q_gap": extra_m["q_gap"].mean(),
                "dist_entropy": extra_m["dist_entropy"].mean(),
            }
            if config.get("LOG_LAYER_GRAD_NORMS", False):
                layer_grad_norms = jax.tree_util.tree_map(lambda x: x.mean(), layer_grad_norms)
                metrics.update({f"grad_norm/{k}": v for k, v in layer_grad_norms.items()})
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / jnp.maximum(infos["returned_episode"].sum(), 1.0),
                infos,
            )
            metrics.update(done_infos)

            if config.get("TEST_DURING_TRAINING", False):
                rng, rng_test = jax.random.split(rng)
                interval = max(int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]), 1)
                test_metrics = jax.lax.cond(
                    train_state.n_updates % interval == 0,
                    lambda _: get_test_metrics(train_state, rng_test),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {k: v for k, v in metrics.items() if "achievement" not in k.lower()}

            if config["WANDB_MODE"] != "disabled":

                def callback(m, seed_idx_):
                    interval = max(int(config.get("WANDB_LOG_INTERVAL", 100)), 1)
                    if int(m["update_steps"]) % interval != 0:
                        return

                    def to_wandb_scalar_or_none(value):
                        try:
                            arr = np.asarray(value)
                        except Exception:
                            return None
                        if arr.ndim == 0:
                            return arr.item() if np.isfinite(arr) else None
                        if arr.size == 1:
                            scalar = arr.reshape(()).item()
                            return scalar if np.isfinite(scalar) else None
                        return None

                    logged = dict(m)
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        logged = {
                            **{f"seed_{int(seed_idx_) + 1}/{k}": v for k, v in logged.items()},
                            **logged,
                        }
                    sanitized = {}
                    for k, v in logged.items():
                        scalar = to_wandb_scalar_or_none(v)
                        if scalar is not None:
                            sanitized[k] = scalar
                    if "update_steps" not in sanitized:
                        update_steps = to_wandb_scalar_or_none(m.get("update_steps"))
                        if update_steps is None:
                            return
                        sanitized["update_steps"] = update_steps
                    wandb.log(sanitized, step=sanitized["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            return (
                train_state,
                memory_transitions,
                tuple(expl_state),
                test_metrics,
                rng,
            ), metrics

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return {"returned_episode_returns": jnp.nan}
            rng, rng_reset = jax.random.split(rng)
            obs, env_state = test_env.reset(rng_reset, env_params)
            done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=jnp.int32)
            hs = network.initialize_carry(config["TEST_NUM_ENVS"])

            def _env_step(carry, _):
                hs, env_state, last_obs, last_done, last_action, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                new_hs, q_vals, _ = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    hs,
                    last_obs[np.newaxis],
                    last_done[np.newaxis],
                    last_action[np.newaxis],
                    train=False,
                    support=support,
                    method=RNNPhiTDQNetwork.q_values,
                )
                q_vals = q_vals.squeeze(axis=0)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    rng_s, env_state, new_action, env_params
                )
                return (new_hs, new_env_state, new_obs, new_done, new_action, rng), info

            _, infos = jax.lax.scan(
                _env_step,
                (hs, env_state, obs, done, action, rng),
                None,
                config["TEST_NUM_STEPS"],
            )
            return jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / jnp.maximum(infos["returned_episode"].sum(), 1.0),
                infos,
            )

        rng, rng_reset = jax.random.split(rng)
        obs, env_state = env.reset(rng_reset, env_params)
        done = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
        action = jnp.zeros((config["NUM_ENVS"]), dtype=jnp.int32)
        hs = network.initialize_carry(config["NUM_ENVS"])
        expl_state = (hs, obs, done, action, env_state)
        test_metrics = (
            get_test_metrics(train_state, rng)
            if config.get("TEST_DURING_TRAINING", False)
            else {"returned_episode_returns": jnp.nan}
        )

        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            new_hs, q_vals, _ = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                hs,
                last_obs[np.newaxis],
                last_done[np.newaxis],
                last_action[np.newaxis],
                train=False,
                support=support,
                method=RNNPhiTDQNetwork.q_values,
            )
            q_vals = q_vals.squeeze(axis=0)
            eps = jnp.ones(config["NUM_ENVS"])
            new_action = jax.vmap(eps_greedy_exploration)(
                jax.random.split(rng_a, config["NUM_ENVS"]), q_vals, eps
            )
            new_obs, new_env_state, reward, new_done, info = env.step(
                rng_s, env_state, new_action, env_params
            )
            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_vals=q_vals,
            )
            return (new_hs, new_obs, new_done, new_action, new_env_state, rng), transition

        rng, rng_warmup = jax.random.split(rng)
        (*expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (*expl_state, rng_warmup),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        runner_state = (
            train_state,
            memory_transitions,
            tuple(expl_state),
            test_metrics,
            rng,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):
    config = {**config, **config["alg"]}
    alg_name = config.get("ALG_NAME", f"phi_td_{config.get('FAMILY_DISTRIBUTION', 'cat')}_rnn")
    env_name = config["ENV_NAME"]
    tags = [alg_name.upper(), env_name.upper().replace(" ", "_"), f"jax_{jax.__version__}"]
    if config.get("EXPERIMENT_TAG"):
        tags.append(str(config["EXPERIMENT_TAG"]))
    if config.get("WANDB_EXTRA_TAGS"):
        tags.extend(str(x) for x in config["WANDB_EXTRA_TAGS"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    seed_idxs = jnp.arange(config["NUM_SEEDS"], dtype=jnp.int32)
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs, seed_idxs))
    elapsed = time.time() - t0
    print(f"Took {elapsed} seconds to complete.")
    print(f"SPS: {float(config['TOTAL_TIMESTEPS']) / max(elapsed, 1e-6)}")

    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        for i, _ in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors'
            )
            save_params(params, save_path)


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        seed_idxs = jnp.arange(config["NUM_SEEDS"], dtype=jnp.int32)
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        jax.block_until_ready(train_vjit(rngs, seed_idxs))

    sweep_config = {
        "name": f'{default_config.get("ALG_NAME", "phi_td")}_{default_config["ENV_NAME"]}',
        "method": "bayes",
        "metric": {"name": "returned_episode_returns", "goal": "maximize"},
        "parameters": {"LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]}},
    }
    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
