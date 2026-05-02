"""
Phi-TD on MinAtar: same CNN shell as ``mog_pqn_minatar.py``, but one distributional family
at a time with **only** the weighted characteristic-function Bellman loss (no TD(λ) / PQN
mean auxiliary).

Families (shared atom count ``NUM_ATOMS`` = m):

  - **dirac** (``F_m``): mixture of Diracs, φ(ω) = Σ_k π_k exp(i ω μ_k).
  - **categorical** (``F_{C,m}``): C51-style fixed support + softmax probs.
  - **quantile** (``F_{Q,m}``): fixed τ fractions + learned locations; φ is the empirical CF.
  - **mog**: mixture of Gaussians (same heads as ``mog_pqn_minatar``); φ via ``build_mog_cf`` (CF-only loss).
  - **cauchy**: mixture of Cauchy laws (MoC); optional / legacy.
  - **gamma**: mixture of Gammas (MoΓ); φ(ω) = Σ_k π_k (1 − i θ_k ω)^(−α_k). Q = Σ_k π_k α_k θ_k.
  - **laplace**: mixture of Laplaces; Q = Σ_k π_k μ_k (finite mean).
  - **logistic**: mixture of logistics; Q = Σ_k π_k μ_k.

See ``purejaxql.utils.mog_cf`` for CF primitives.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import chex
import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from flax import linen as nn
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnax.wrappers.purerl import LogWrapper
from omegaconf import OmegaConf

import gymnax

from purejaxql.distributional_pqn_minatar import project_categorical, select_action_values
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


def apply_norm(x: jnp.ndarray, norm_type: str, train: bool):
    if norm_type == "layer_norm":
        return nn.LayerNorm()(x)
    if norm_type == "batch_norm":
        return nn.BatchNorm(use_running_average=not train)(x)
    if norm_type == "justnorm":
        eps = jnp.finfo(x.dtype).eps
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
        return x / rms
    return x


class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        for i, channels in enumerate((32, 64, 128)):
            x = nn.Conv(
                channels,
                kernel_size=(3, 3),
                strides=1,
                padding="SAME",
                kernel_init=nn.initializers.he_normal(),
                name=f"conv_{i}",
            )(x)
            x = apply_norm(x, self.norm_type, train)
            x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        return x


class PhiTDMinAtarQNetwork(nn.Module):
    """CNN trunk + head chosen by ``family_distribution`` (incl. mog / gamma / laplace / logistic)."""

    family_distribution: str
    action_dim: int
    num_atoms: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    mlp_num_layers: int = 1
    mlp_hidden_dim: int = 128

    @nn.compact
    def encode(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, name="input_norm")(x)
        else:
            _ = nn.BatchNorm(use_running_average=not train, name="input_norm_dummy")(x)
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(
            self.mlp_hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            name="mlp_proj",
        )(x)
        x = apply_norm(x, self.norm_type, train)
        for i in range(self.mlp_num_layers):
            resid = x
            x = nn.Dense(
                4 * self.mlp_hidden_dim,
                kernel_init=nn.initializers.he_normal(),
                use_bias=False,
                name=f"mlp_up_{i}",
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                self.mlp_hidden_dim,
                kernel_init=nn.zeros,
                use_bias=False,
                name=f"mlp_down_{i}",
            )(x)
            x = apply_norm(x, self.norm_type, train)
            x = x + resid
        return x

    @nn.compact
    def decode(self, latent: jnp.ndarray):
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
            sigma = jax.nn.softplus(
                nn.Dense(a * m, name="head_sigma")(latent).reshape(-1, a, m)
            )
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
    def __call__(self, x: jnp.ndarray, train: bool):
        latent = self.encode(x, train)
        return self.decode(latent)

@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config: dict):
    config = dict(config)
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = int(
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    max_grad_norm = config["MAX_GRAD_NORM"]
    per_layer_gradient_clipping = config.get("PER_LAYER_GRADIENT_CLIPPING", False)

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config["NUM_MINIBATCHES"] == 0, (
        "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"
    )

    minibatch_size = (config["NUM_STEPS"] * config["NUM_ENVS"]) // config["NUM_MINIBATCHES"]

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    config["TEST_NUM_STEPS"] = env_params.max_steps_in_episode

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    action_dim = env.action_space(env_params).n
    family = str(config.get("FAMILY_DISTRIBUTION", "dirac")).lower()
    num_atoms = int(config["NUM_ATOMS"])
    support = jnp.linspace(
        float(config.get("V_MIN", 0.0)),
        float(config.get("V_MAX", 200.0)),
        num_atoms,
    )

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
            layer_key = "/".join(key[:-1]) if len(key) > 1 else key[0]
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
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        network = PhiTDMinAtarQNetwork(
            family_distribution=family,
            action_dim=action_dim,
            num_atoms=num_atoms,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
            mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
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
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_variables = network.init(rng_init, init_x, train=False)
            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, rng_agent = jax.random.split(rng)
        train_state = create_agent(rng_agent)

        def policy_q(variables, obs):
            out = network.apply(variables, obs, train=False)
            if family == "dirac":
                return mog_q_values(out["pi"], out["mu"])
            if family == "mog":
                return mog_q_values(out["pi"], out["mu"])
            if family == "categorical":
                probs = jax.nn.softmax(out["logits"], axis=-1)
                return jnp.sum(probs * support, axis=-1)
            if family == "cauchy":
                return mog_q_values(out["pi"], out["loc"])
            if family == "gamma":
                return gamma_mixture_q_values(out["pi"], out["shape"], out["scale"])
            if family in ("laplace", "logistic"):
                return mog_q_values(out["pi"], out["loc"])
            return jnp.mean(out["quantiles"], axis=-1)

        def phi_gradient_step(train_state, batch, rng_step):
            rng_step, omega_key = jax.random.split(rng_step)
            s_obs = batch.obs
            s_next_obs = batch.next_obs
            s_actions = batch.action
            s_rewards = batch.reward
            s_dones = batch.done
            bsz = s_obs.shape[0]
            bidx = jnp.arange(bsz)
            omegas = sample_frequencies(
                omega_key,
                int(config["NUM_OMEGA_SAMPLES"]),
                float(config["OMEGA_MAX"]),
                distribution=config.get("OMEGA_SAMPLING_DISTRIBUTION", "half_laplacian"),
            )
            variables_eval = {
                "params": train_state.params,
                "batch_stats": train_state.batch_stats,
            }

            out_next = network.apply(variables_eval, s_next_obs, train=False)

            if family == "dirac":
                q_next = mog_q_values(out_next["pi"], out_next["mu"])
                next_actions = jnp.argmax(q_next, axis=1)
                pi_sel = out_next["pi"][bidx, next_actions]
                mu_sel = out_next["mu"][bidx, next_actions]
                gammas = config["GAMMA"] * (1.0 - s_dones)
                bellman_mu = s_rewards[:, None] + gammas[:, None] * mu_sel
                bellman_pi = pi_sel
                td_target = build_dirac_mixture_cf(
                    bellman_pi[:, None, :],
                    bellman_mu[:, None, :],
                    omegas,
                )
                td_target = td_target[:, :, 0]
            elif family == "categorical":
                next_probs = jax.nn.softmax(out_next["logits"], axis=-1)
                next_q = jnp.sum(next_probs * support, axis=-1)
                next_actions = jnp.argmax(next_q, axis=-1)
                next_probs_sel = next_probs[bidx, next_actions]
                target_probs = project_categorical(
                    next_probs_sel,
                    s_rewards,
                    s_dones.astype(jnp.float32),
                    support,
                    float(config["GAMMA"]),
                    float(config.get("V_MIN", 0.0)),
                    float(config.get("V_MAX", 200.0)),
                )
                td_target = build_categorical_cf(
                    jax.lax.stop_gradient(target_probs),
                    support,
                    omegas,
                )
            elif family == "quantile":
                nq = out_next["quantiles"]
                next_q = jnp.mean(nq, axis=-1)
                next_actions = jnp.argmax(next_q, axis=-1)
                sel_nq = nq[bidx, next_actions]
                target_q = s_rewards[:, None] + config["GAMMA"] * (
                    1.0 - s_dones.astype(jnp.float32)[:, None]
                ) * sel_nq
                td_target = build_quantile_cf(jax.lax.stop_gradient(target_q), omegas)
            elif family == "mog":
                q_next = mog_q_values(out_next["pi"], out_next["mu"])
                next_actions = jnp.argmax(q_next, axis=1)
                pi_sel = out_next["pi"][bidx, next_actions]
                mu_sel = out_next["mu"][bidx, next_actions]
                sig_sel = out_next["sigma"][bidx, next_actions]
                gammas = config["GAMMA"] * (1.0 - s_dones)
                bellman_mu = s_rewards[:, None] + gammas[:, None] * mu_sel
                bellman_sigma = gammas[:, None] * sig_sel
                td_target = build_mog_cf(
                    pi_sel[:, None, :],
                    bellman_mu[:, None, :],
                    bellman_sigma[:, None, :],
                    omegas,
                )
                td_target = td_target[:, :, 0]
            elif family == "cauchy":
                q_next = mog_q_values(out_next["pi"], out_next["loc"])
                next_actions = jnp.argmax(q_next, axis=1)
                pi_sel = out_next["pi"][bidx, next_actions]
                loc_sel = out_next["loc"][bidx, next_actions]
                scale_sel = out_next["scale"][bidx, next_actions]
                gammas = config["GAMMA"] * (1.0 - s_dones)
                bellman_loc = s_rewards[:, None] + gammas[:, None] * loc_sel
                bellman_scale = gammas[:, None] * scale_sel
                td_target = build_cauchy_mixture_cf(
                    pi_sel[:, None, :],
                    bellman_loc[:, None, :],
                    bellman_scale[:, None, :],
                    omegas,
                )
                td_target = td_target[:, :, 0]
            elif family == "gamma":
                q_next = gamma_mixture_q_values(
                    out_next["pi"], out_next["shape"], out_next["scale"]
                )
                next_actions = jnp.argmax(q_next, axis=1)
                pi_sel = out_next["pi"][bidx, next_actions]
                shape_sel = out_next["shape"][bidx, next_actions]
                scale_sel = out_next["scale"][bidx, next_actions]
                gammas = config["GAMMA"] * (1.0 - s_dones)
                bellman_scale = gammas[:, None] * scale_sel
                phase_r = jnp.exp(1j * s_rewards[:, None] * omegas[None, :])
                mix_cf = build_gamma_mixture_cf(
                    pi_sel[:, None, :],
                    shape_sel[:, None, :],
                    bellman_scale[:, None, :],
                    omegas,
                )
                td_target = phase_r * mix_cf[:, :, 0]
            elif family == "laplace":
                q_next = mog_q_values(out_next["pi"], out_next["loc"])
                next_actions = jnp.argmax(q_next, axis=1)
                pi_sel = out_next["pi"][bidx, next_actions]
                loc_sel = out_next["loc"][bidx, next_actions]
                scale_sel = out_next["scale"][bidx, next_actions]
                gammas = config["GAMMA"] * (1.0 - s_dones)
                bellman_loc = s_rewards[:, None] + gammas[:, None] * loc_sel
                bellman_scale = gammas[:, None] * scale_sel
                td_target = build_laplace_mixture_cf(
                    pi_sel[:, None, :],
                    bellman_loc[:, None, :],
                    bellman_scale[:, None, :],
                    omegas,
                )
                td_target = td_target[:, :, 0]
            elif family == "logistic":
                q_next = mog_q_values(out_next["pi"], out_next["loc"])
                next_actions = jnp.argmax(q_next, axis=1)
                pi_sel = out_next["pi"][bidx, next_actions]
                loc_sel = out_next["loc"][bidx, next_actions]
                scale_sel = out_next["scale"][bidx, next_actions]
                gammas = config["GAMMA"] * (1.0 - s_dones)
                bellman_loc = s_rewards[:, None] + gammas[:, None] * loc_sel
                bellman_scale = gammas[:, None] * scale_sel
                td_target = build_logistic_mixture_cf(
                    pi_sel[:, None, :],
                    bellman_loc[:, None, :],
                    bellman_scale[:, None, :],
                    omegas,
                )
                td_target = td_target[:, :, 0]
            else:
                raise ValueError(f"Unknown FAMILY_DISTRIBUTION: {family}")

            td_target = jax.lax.stop_gradient(td_target)

            def _loss_fn(params):
                out, updates = network.apply(
                    {"params": params, "batch_stats": train_state.batch_stats},
                    s_obs,
                    train=True,
                    mutable=["batch_stats"],
                )
                if family == "dirac":
                    online_phi = build_dirac_mixture_cf(out["pi"], out["mu"], omegas)
                    online_phi_sel = online_phi[bidx, :, s_actions]
                elif family == "categorical":
                    probs = jax.nn.softmax(out["logits"], axis=-1)
                    sel_probs = select_action_values(probs, s_actions)
                    online_phi_sel = build_categorical_cf(sel_probs, support, omegas)
                elif family == "quantile":
                    sel_q = select_action_values(out["quantiles"], s_actions)
                    online_phi_sel = build_quantile_cf(sel_q, omegas)
                elif family == "mog":
                    online_phi = build_mog_cf(out["pi"], out["mu"], out["sigma"], omegas)
                    online_phi_sel = online_phi[bidx, :, s_actions]
                elif family == "cauchy":
                    online_phi = build_cauchy_mixture_cf(
                        out["pi"], out["loc"], out["scale"], omegas
                    )
                    online_phi_sel = online_phi[bidx, :, s_actions]
                elif family == "gamma":
                    online_phi = build_gamma_mixture_cf(
                        out["pi"], out["shape"], out["scale"], omegas
                    )
                    online_phi_sel = online_phi[bidx, :, s_actions]
                elif family == "laplace":
                    online_phi = build_laplace_mixture_cf(
                        out["pi"], out["loc"], out["scale"], omegas
                    )
                    online_phi_sel = online_phi[bidx, :, s_actions]
                elif family == "logistic":
                    online_phi = build_logistic_mixture_cf(
                        out["pi"], out["loc"], out["scale"], omegas
                    )
                    online_phi_sel = online_phi[bidx, :, s_actions]
                else:
                    raise ValueError(f"Unknown FAMILY_DISTRIBUTION: {family}")

                residual = online_phi_sel - td_target
                err_squared = jnp.abs(residual) ** 2
                imag_abs = jnp.abs(jnp.imag(residual))
                if config.get("IS_DIVIDED_BY_OMEGA_SQUARED", True):
                    omega_weights = (1.0 / jnp.maximum(omegas, 1e-8) ** 2)[None, :]
                    err_squared = err_squared * omega_weights
                    imag_abs = imag_abs * omega_weights
                cf_loss = jnp.mean(err_squared)
                cf_im = jnp.mean(imag_abs)

                if family == "dirac":
                    q_all = mog_q_values(out["pi"], out["mu"])
                elif family == "mog":
                    q_all = mog_q_values(out["pi"], out["mu"])
                elif family == "categorical":
                    probs = jax.nn.softmax(out["logits"], axis=-1)
                    q_all = jnp.sum(probs * support, axis=-1)
                elif family == "cauchy":
                    q_all = mog_q_values(out["pi"], out["loc"])
                elif family == "gamma":
                    q_all = gamma_mixture_q_values(out["pi"], out["shape"], out["scale"])
                elif family in ("laplace", "logistic"):
                    q_all = mog_q_values(out["pi"], out["loc"])
                else:
                    q_all = jnp.mean(out["quantiles"], axis=-1)
                chosen_action_qvals = jnp.take_along_axis(
                    q_all,
                    jnp.expand_dims(s_actions, axis=-1),
                    axis=-1,
                ).squeeze(axis=-1)
                q_mean = chosen_action_qvals.mean()
                q_gap = (q_all.max(axis=1) - q_all.min(axis=1)).mean()
                if family == "dirac":
                    ent = -jnp.sum(out["pi"] * jnp.log(out["pi"] + 1e-8), axis=-1).mean()
                elif family == "categorical":
                    ent = -jnp.sum(
                        jax.nn.softmax(out["logits"], -1)
                        * jax.nn.log_softmax(out["logits"], -1),
                        axis=-1,
                    ).mean()
                elif family in ("cauchy", "gamma", "mog", "laplace", "logistic"):
                    ent = -jnp.sum(out["pi"] * jnp.log(out["pi"] + 1e-8), axis=-1).mean()
                else:
                    ent = jnp.array(0.0, dtype=jnp.float32)
                return cf_loss, (updates, q_mean, q_gap, ent, cf_im, cf_loss)

            (loss, (updates, q_mean, q_gap, ent, cf_im, cf_loss)), grads = jax.value_and_grad(
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
            return train_state, loss, (q_mean, q_gap, ent, cf_im, cf_loss), grad_norm, layer_grad_norms

        def _update_step(runner_state, unused):
            train_state, expl_state, test_metrics, rng = runner_state

            eps = eps_scheduler(train_state.n_updates)

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_phi_mean = policy_q(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                )
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps_vec = jnp.full((config["NUM_ENVS"],), eps)
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_phi_mean, eps_vec)

                new_obs, new_env_state, reward, new_done, info = vmap_step(config["NUM_ENVS"])(
                    rng_s, env_state, new_action
                )
                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):
                    train_state, rng = carry
                    rng, rng_gs = jax.random.split(rng)
                    train_state, loss, aux, grad_norm, layer_grad_norms = phi_gradient_step(
                        train_state, minibatch, rng_gs
                    )
                    q_mean, q_gap, ent, cf_im, cf_l = aux
                    return (train_state, rng), (loss, q_mean, q_gap, ent, cf_im, cf_l, grad_norm, layer_grad_norms)

                def preprocess_transition(x, rng_perm):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng_perm, x)
                    return x.reshape(
                        config["NUM_MINIBATCHES"], minibatch_size, *x.shape[1:]
                    )

                rng, rng_perm = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, rng_perm), transitions
                )

                rng, rng_epoch = jax.random.split(rng)
                (train_state, rng), (
                    losses,
                    q_means,
                    q_gaps,
                    ents,
                    cf_ims,
                    cf_losses,
                    grad_norms,
                    layer_grad_norms,
                ) = jax.lax.scan(
                    _learn_phase,
                    (train_state, rng_epoch),
                    minibatches,
                )
                return (train_state, rng), (
                    losses,
                    q_means,
                    q_gaps,
                    ents,
                    cf_ims,
                    cf_losses,
                    grad_norms,
                    layer_grad_norms,
                )

            rng, _rng = jax.random.split(rng)
            (
                train_state,
                rng,
            ), (
                loss,
                q_mean_e,
                q_gap_e,
                ent_e,
                cf_im_e,
                cf_loss_e,
                grad_norm,
                layer_grad_norms,
            ) = jax.lax.scan(_learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"])

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            metrics = {
                "env_step": train_state.timesteps,
                "global_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps
                * env.observation_space(env_params).shape[-1],
                "grad_steps": train_state.grad_steps,
                "total_loss": loss.mean(),
                "phi_cf_loss": cf_loss_e.mean(),
                "q_values": q_mean_e.mean(),
                "q_gap": q_gap_e.mean(),
                "dist_entropy": ent_e.mean(),
                "cf_loss_imag": cf_im_e.mean(),
                "grad_norm": grad_norm.mean(),
                "epsilon": eps,
            }
            if config.get("LOG_LAYER_GRAD_NORMS", False):
                layer_grad_norms = jax.tree_util.tree_map(lambda x: x.mean(), layer_grad_norms)
                metrics.update({f"grad_norm/{k}": v for k, v in layer_grad_norms.items()})
            metrics.update({k: v.mean() for k, v in infos.items()})
            if "returned_episode_returns" in metrics:
                metrics["charts/episodic_return"] = metrics["returned_episode_returns"]

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(m, seed_idx_):
                    _iv = max(int(config.get("WANDB_LOG_INTERVAL", 100)), 1)
                    if int(m["update_steps"]) % _iv != 0:
                        return
                    logged = dict(m)
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        logged = {
                            **{f"seed_{int(seed_idx_) + 1}/{k}": v for k, v in logged.items()},
                            **logged,
                        }
                    wandb.log(logged, step=logged["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)
            return runner_state, metrics

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return {}

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = policy_q(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                )
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = vmap_step(config["TEST_NUM_ENVS"])(
                    _rng, env_state, new_action
                )
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)
            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, rng), None, config["TEST_NUM_STEPS"]
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )

        rng, rng_test = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, rng_test)

        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def _wandb_tags(config: dict) -> list[str]:
    alg_name = config.get("ALG_NAME", "phi_td_pqn")
    env_name = config["ENV_NAME"]
    tags = [
        alg_name.upper(),
        env_name.upper().replace(" ", "_"),
        f"jax_{jax.__version__}",
    ]
    et = config.get("EXPERIMENT_TAG")
    if et:
        tags.append(str(et))
    extra = config.get("WANDB_EXTRA_TAGS")
    if extra:
        tags.extend(str(x) for x in extra)
    return tags


def single_run(config):
    config = {**config, **config["alg"]}

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=_wandb_tags(config),
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
    _elapsed = time.time() - t0
    print(f"Took {_elapsed} seconds to complete.")
    print(f"SPS: {float(config['TOTAL_TIMESTEPS']) / max(_elapsed, 1e-6)}")

    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], config["ENV_NAME"])
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i in range(config["NUM_SEEDS"]):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_seed{config["SEED"]}_vmap{i}.safetensors',
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
        "name": f"{default_config['ALG_NAME']}_{default_config['ENV_NAME']}",
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
