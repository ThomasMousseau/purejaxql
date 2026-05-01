"""
MoG-PQN on MinAtar (gymnax): CNN trunk + mixture-of-Gaussians CF heads.

**Fair PQN comparison:** Same training shell as ``pqn_minatar.py`` — single network (no target
net, no Polyak), RAdam + global grad clip, ε schedule over **update index**, and
``CustomTrainState`` with batch norm stats.

Algorithm vs PQN baseline:

  - **Actions / values:** Q(s,a) = Σ_k π_k μ_k (``mog_q_values``) instead of scalar logits.
  - **Rollout:** stores ``q_mog_mean`` = full vector Q(s,·) at collection time — same role as
    ``q_val`` in ``pqn_minatar.py``.
  - **Loss (fused):** (1) MoG characteristic-function Bellman MSE on φ; (2) TD(λ) half-MSE on
    chosen-action ``q_mog_mean`` vs λ-return targets built **like** ``pqn_minatar`` (reverse scan,
    rollout ``q_mog_mean`` in the bootstrap). ``total = cf_loss + AUX_MEAN_LOSS_WEIGHT * td_lambda``.
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

from purejaxql.utils.mog_cf import build_mog_cf, mog_q_values, sample_frequencies


def normal_cdf(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def mog_centered_cramer_distance(
    pi_old: jnp.ndarray,
    mu_old: jnp.ndarray,
    sigma_old: jnp.ndarray,
    pi_new: jnp.ndarray,
    mu_new: jnp.ndarray,
    sigma_new: jnp.ndarray,
    *,
    grid_size: int = 129,
) -> jnp.ndarray:
    """Cramer distance between centered old/new action distributions."""
    mean_old = jnp.sum(pi_old * mu_old, axis=-1, keepdims=True)
    mean_new = jnp.sum(pi_new * mu_new, axis=-1, keepdims=True)
    mu_old_c = mu_old - mean_old
    mu_new_c = mu_new - mean_new

    scale_old = jnp.sum(pi_old * sigma_old, axis=-1)
    scale_new = jnp.sum(pi_new * sigma_new, axis=-1)
    max_scale = jnp.maximum(scale_old, scale_new)
    max_mu = jnp.maximum(
        jnp.max(jnp.abs(mu_old_c), axis=-1),
        jnp.max(jnp.abs(mu_new_c), axis=-1),
    )
    radius = jnp.maximum(1.0, max_mu + 4.0 * max_scale)
    x = jnp.linspace(-1.0, 1.0, grid_size, dtype=pi_old.dtype)
    x = radius[..., None] * x
    dx = jnp.maximum(1e-6, (2.0 * radius) / jnp.maximum(grid_size - 1, 1))

    cdf_old = jnp.sum(
        pi_old[..., None, :] * normal_cdf((x[..., None] - mu_old_c[..., None, :]) / (sigma_old[..., None, :] + 1e-6)),
        axis=-1,
    )
    cdf_new = jnp.sum(
        pi_new[..., None, :] * normal_cdf((x[..., None] - mu_new_c[..., None, :]) / (sigma_new[..., None, :] + 1e-6)),
        axis=-1,
    )
    sq = jnp.square(cdf_old - cdf_new)
    per_dist = jnp.sum(sq, axis=-1) * dx
    return jnp.mean(per_dist)
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


class MoGMinAtarQNetwork(nn.Module):
    """CNN trunk + MoG heads (π, μ, σ) per action × component."""

    action_dim: int
    num_components: int
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
        a, m = self.action_dim, self.num_components
        pi_logits = nn.Dense(a * m, name="head_pi_logits")(latent)
        pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
        mu = nn.Dense(a * m, name="head_mu")(latent).reshape(-1, a, m)
        sigma = jax.nn.softplus(nn.Dense(a * m, name="head_sigma")(latent).reshape(-1, a, m))
        return pi, mu, sigma

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
    q_mog_mean: chex.Array  # (NUM_ENVS, action_dim), same contract as ``q_val`` in pqn_minatar


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

        network = MoGMinAtarQNetwork(
            action_dim=action_dim,
            num_components=int(config["NUM_COMPONENTS"]),
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

        aux_w = float(config.get("AUX_MEAN_LOSS_WEIGHT", 1.0))

        def mog_gradient_step(train_state, batch, lambda_target, rng_step):
            """MoG–CF Bellman + TD(λ) on mean Q (same bootstrap role as PQN)."""
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
                distribution=config.get("OMEGA_SAMPLING_DISTRIBUTION", "half_laplacian")
            )

            pi_n, mu_n, sig_n = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                s_next_obs,
                train=False,
            )
            q_next = mog_q_values(pi_n, mu_n)
            next_actions = jnp.argmax(q_next, axis=1)

            pi_sel = pi_n[bidx, next_actions]
            mu_sel = mu_n[bidx, next_actions]
            sig_sel = sig_n[bidx, next_actions]

            gammas = config["GAMMA"] * (1.0 - s_dones)
            bellman_mu = s_rewards[:, None] + gammas[:, None] * mu_sel
            bellman_sigma = gammas[:, None] * sig_sel
            bellman_pi = pi_sel

            td_target = build_mog_cf(
                bellman_pi[:, None, :],
                bellman_mu[:, None, :],
                bellman_sigma[:, None, :],
                omegas,
            )
            td_target = td_target[:, :, 0]
            td_target = jax.lax.stop_gradient(td_target)

            def _loss_fn(params):
                (online_pi, online_mu, online_sigma), updates = network.apply(
                    {"params": params, "batch_stats": train_state.batch_stats},
                    s_obs,
                    train=True,
                    mutable=["batch_stats"],
                )
                online_phi = build_mog_cf(online_pi, online_mu, online_sigma, omegas)
                online_phi_sel = online_phi[bidx, :, s_actions]
                residual = online_phi_sel - td_target
                err_squared = jnp.abs(residual) ** 2
                imag_abs = jnp.abs(jnp.imag(residual))
                if config.get("IS_DIVIDED_BY_OMEGA_SQUARED", True):
                    # CF Loss weighted by 1/ω², thus ∫|φ(ω)−φ̂(ω)|²/ω² dω.
                    omega_weights = (1.0 / jnp.maximum(omegas, 1e-8) ** 2)[None, :]
                    err_squared = err_squared * omega_weights
                    imag_abs = imag_abs * omega_weights
                cf_loss = jnp.mean(err_squared)
                cf_im = jnp.mean(imag_abs)
                q_all = mog_q_values(online_pi, online_mu)
                chosen_action_qvals = jnp.take_along_axis(
                    q_all,
                    jnp.expand_dims(s_actions, axis=-1),
                    axis=-1,
                ).squeeze(axis=-1)
                td_lambda_loss = 0.5 * jnp.mean(
                    jnp.square(chosen_action_qvals - lambda_target)
                )
                total_loss = cf_loss + aux_w * td_lambda_loss
                q_mean = chosen_action_qvals.mean()
                q_gap = (q_all.max(axis=1) - q_all.min(axis=1)).mean()
                mog_ent = -jnp.sum(online_pi * jnp.log(online_pi + 1e-8), axis=-1).mean()
                sig_m = online_sigma.mean()
                return total_loss, (
                    updates,
                    q_mean,
                    q_gap,
                    mog_ent,
                    sig_m,
                    cf_im,
                    cf_loss,
                    td_lambda_loss,
                )

            def _separate_loss_grads(params):
                variables = {"params": params, "batch_stats": train_state.batch_stats}
                latent = network.apply(variables, s_obs, train=False, method=MoGMinAtarQNetwork.encode)
                online_pi, online_mu, online_sigma = network.apply(
                    variables, latent, method=MoGMinAtarQNetwork.decode
                )
                online_phi = build_mog_cf(online_pi, online_mu, online_sigma, omegas)
                online_phi_sel = online_phi[bidx, :, s_actions]
                residual = online_phi_sel - td_target
                err_squared = jnp.abs(residual) ** 2
                if config.get("IS_DIVIDED_BY_OMEGA_SQUARED", True):
                    omega_weights = (1.0 / jnp.maximum(omegas, 1e-8) ** 2)[None, :]
                    err_squared = err_squared * omega_weights
                dist_loss = jnp.mean(err_squared)

                q_all = mog_q_values(online_pi, online_mu)
                chosen_action_qvals = jnp.take_along_axis(
                    q_all,
                    jnp.expand_dims(s_actions, axis=-1),
                    axis=-1,
                ).squeeze(axis=-1)
                td_loss = 0.5 * jnp.mean(jnp.square(chosen_action_qvals - lambda_target))

                td_rep_grad = jax.grad(
                    lambda h: 0.5
                    * jnp.mean(
                        jnp.square(
                            jnp.take_along_axis(
                                mog_q_values(
                                    network.apply(variables, h, method=MoGMinAtarQNetwork.decode)[0],
                                    network.apply(variables, h, method=MoGMinAtarQNetwork.decode)[1],
                                ),
                                jnp.expand_dims(s_actions, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            - lambda_target
                        )
                    )
                )(latent)
                dist_rep_grad = jax.grad(
                    lambda h: jnp.mean(
                        (
                            jnp.abs(
                                build_mog_cf(*network.apply(variables, h, method=MoGMinAtarQNetwork.decode), omegas)[
                                    bidx, :, s_actions
                                ]
                                - td_target
                            )
                            ** 2
                        )
                        * (
                            (1.0 / jnp.maximum(omegas, 1e-8) ** 2)[None, :]
                            if config.get("IS_DIVIDED_BY_OMEGA_SQUARED", True)
                            else 1.0
                        )
                    )
                )(latent)
                td_rep_norm = jnp.sqrt(jnp.sum(jnp.square(td_rep_grad)) + 1e-8)
                dist_rep_norm = jnp.sqrt(jnp.sum(jnp.square(dist_rep_grad)) + 1e-8)
                grad_cos = jnp.sum(td_rep_grad * dist_rep_grad) / (td_rep_norm * dist_rep_norm + 1e-8)

                aux_param_grads = jax.grad(
                    lambda p: 0.5
                    * jnp.mean(
                        jnp.square(
                            jnp.take_along_axis(
                                mog_q_values(
                                    network.apply(
                                        {"params": p, "batch_stats": train_state.batch_stats},
                                        s_obs,
                                        train=False,
                                    )[0],
                                    network.apply(
                                        {"params": p, "batch_stats": train_state.batch_stats},
                                        s_obs,
                                        train=False,
                                    )[1],
                                ),
                                jnp.expand_dims(s_actions, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            - lambda_target
                        )
                    )
                )(params)
                step_size = float(config.get("CENTERED_SHAPE_STEP_SIZE", config["LR"]))
                updated_params = jax.tree_util.tree_map(lambda p, g: p - step_size * g, params, aux_param_grads)
                new_pi, new_mu, new_sigma = network.apply(
                    {"params": updated_params, "batch_stats": train_state.batch_stats},
                    s_obs,
                    train=False,
                )
                centered_shape_volatility = mog_centered_cramer_distance(
                    online_pi[bidx, s_actions],
                    online_mu[bidx, s_actions],
                    online_sigma[bidx, s_actions],
                    new_pi[bidx, s_actions],
                    new_mu[bidx, s_actions],
                    new_sigma[bidx, s_actions],
                    grid_size=int(config.get("CENTERED_SHAPE_GRID_SIZE", 129)),
                )
                return dist_loss, td_loss, td_rep_norm, dist_rep_norm, grad_cos, centered_shape_volatility

            (loss, (updates, q_mean, q_gap, mog_ent, sig_m, cf_im, cf_loss, td_lambda_loss)), grads = jax.value_and_grad(
                _loss_fn, has_aux=True
            )(train_state.params)
            (
                dist_only_loss,
                td_only_loss,
                td_rep_grad_norm,
                dist_rep_grad_norm,
                grad_cosine_similarity,
                centered_shape_volatility,
            ) = jax.lax.cond(
                config.get("LOG_AUX_DIST_GRAD_ALIGNMENT", True),
                lambda p: _separate_loss_grads(p),
                lambda _: (
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                    jnp.array(0.0, dtype=jnp.float32),
                ),
                operand=train_state.params,
            )
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
            return (
                train_state,
                loss,
                (q_mean, q_gap, mog_ent, sig_m, cf_im, cf_loss, td_lambda_loss),
                grad_norm,
                layer_grad_norms,
                (
                    dist_only_loss,
                    td_only_loss,
                    td_rep_grad_norm,
                    dist_rep_grad_norm,
                    grad_cosine_similarity,
                    centered_shape_volatility,
                ),
            )

        def _update_step(runner_state, unused):
            train_state, expl_state, test_metrics, rng = runner_state

            eps = eps_scheduler(train_state.n_updates)

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                pi, mu, sig = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                q_mog_mean = mog_q_values(pi, mu)
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps_vec = jnp.full((config["NUM_ENVS"],), eps)
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_mog_mean, eps_vec)

                new_obs, new_env_state, reward, new_done, info = vmap_step(config["NUM_ENVS"])(
                    rng_s, env_state, new_action
                )
                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_mog_mean=q_mog_mean,
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

            pi_last, mu_last, _ = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(mog_q_values(pi_last, mu_last), axis=-1)
            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward
                    + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (1 - transition.done) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_mog_mean, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, mb_and_td):
                    train_state, rng = carry
                    minibatch, lambda_td = mb_and_td
                    rng, rng_gs = jax.random.split(rng)
                    (
                        train_state,
                        loss,
                        aux,
                        grad_norm,
                        layer_grad_norms,
                        grad_alignment_aux,
                    ) = mog_gradient_step(train_state, minibatch, lambda_td, rng_gs)
                    q_mean, q_gap, mog_ent, sig_m, cf_im, cf_l, td_l = aux
                    (
                        dist_only_loss,
                        td_only_loss,
                        td_rep_grad_norm,
                        dist_rep_grad_norm,
                        grad_cosine_similarity,
                        centered_shape_volatility,
                    ) = grad_alignment_aux
                    return (
                        train_state,
                        rng,
                    ), (
                        loss,
                        q_mean,
                        q_gap,
                        mog_ent,
                        sig_m,
                        cf_im,
                        cf_l,
                        td_l,
                        grad_norm,
                        layer_grad_norms,
                        dist_only_loss,
                        td_only_loss,
                        td_rep_grad_norm,
                        dist_rep_grad_norm,
                        grad_cosine_similarity,
                        centered_shape_volatility,
                    )

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
                lambda_mb = preprocess_transition(lambda_targets, rng_perm)

                rng, rng_epoch = jax.random.split(rng)
                (train_state, rng), (
                    losses,
                    q_means,
                    q_gaps,
                    mog_ents,
                    sig_ms,
                    cf_ims,
                    cf_losses,
                    td_lambda_losses,
                    grad_norms,
                    layer_grad_norms,
                    dist_only_losses,
                    td_only_losses,
                    td_rep_grad_norms,
                    dist_rep_grad_norms,
                    grad_cosines,
                    centered_shape_vols,
                ) = jax.lax.scan(
                    _learn_phase,
                    (train_state, rng_epoch),
                    (minibatches, lambda_mb),
                )
                return (train_state, rng), (
                    losses,
                    q_means,
                    q_gaps,
                    mog_ents,
                    sig_ms,
                    cf_ims,
                    cf_losses,
                    td_lambda_losses,
                    grad_norms,
                    layer_grad_norms,
                    dist_only_losses,
                    td_only_losses,
                    td_rep_grad_norms,
                    dist_rep_grad_norms,
                    grad_cosines,
                    centered_shape_vols,
                )

            rng, _rng = jax.random.split(rng)
            (
                train_state,
                rng,
            ), (
                loss,
                q_mean_e,
                q_gap_e,
                mog_ent_e,
                sig_m_e,
                cf_im_e,
                cf_loss_e,
                td_lambda_e,
                grad_norm,
                layer_grad_norms,
                dist_only_loss_e,
                td_only_loss_e,
                td_rep_grad_norm_e,
                dist_rep_grad_norm_e,
                grad_cosine_e,
                centered_shape_vol_e,
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
                "mog_cf_loss": cf_loss_e.mean(),
                "td_lambda_loss": td_lambda_e.mean(),
                "q_values": q_mean_e.mean(),
                "q_gap": q_gap_e.mean(),
                "mog_entropy": mog_ent_e.mean(),
                "sigma_mean": sig_m_e.mean(),
                "cf_loss_imag": cf_im_e.mean(),
                "grad_norm": grad_norm.mean(),
                "aux_td_lambda_loss_for_grad_probe": td_only_loss_e.mean(),
                "dist_loss_for_grad_probe": dist_only_loss_e.mean(),
                "trunk_td_lambda_grad_norm": td_rep_grad_norm_e.mean(),
                "trunk_dist_grad_norm": dist_rep_grad_norm_e.mean(),
                "trunk_grad_cosine_similarity": grad_cosine_e.mean(),
                "centered_shape_volatility": centered_shape_vol_e.mean(),
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
                pi, mu, sig = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                q_vals = mog_q_values(pi, mu)
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
    alg_name = config.get("ALG_NAME", "mog_pqn")
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

    alg_name = config.get("ALG_NAME", "mog_pqn")
    env_name = config["ENV_NAME"]

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
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i in range(config["NUM_SEEDS"]):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "mog_pqn")
    env_name = default_config["ENV_NAME"]

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
        "name": f"{alg_name}_{env_name}",
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
