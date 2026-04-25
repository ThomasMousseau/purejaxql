"""Distributional PQN baselines on MinAtar.

This module keeps the same CNN/PQN rollout shell as ``pqn_minatar.py`` and
swaps only the value head and distributional Bellman loss:

* CTD/C51: fixed atoms with learned categorical probabilities.
* QTD/QR-DQN: fixed quantile fractions with learned quantile locations.
* IQN: sampled quantile fractions with cosine conditioning.

All variants also optimize the PQN TD(lambda) mean loss, so greedy policy
learning uses the same long-horizon scalar target as the PQN and MoG-PQN
comparisons.
"""

from __future__ import annotations

import copy
import os
import random
import time
from typing import Any

import chex
import gymnax
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
        return jnp.mean(x, axis=(1, 2))


class CosineEmbedding(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, taus: jnp.ndarray) -> jnp.ndarray:
        basis = jnp.arange(1, self.hidden_dim + 1, dtype=taus.dtype)
        x = jnp.cos(jnp.pi * taus[..., None] * basis)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        return nn.relu(x)


class DistributionalMinAtarNetwork(nn.Module):
    action_dim: int
    variant: str
    num_atoms: int = 51
    num_quantiles: int = 51
    policy_num_quantiles: int = 32
    norm_type: str = "layer_norm"
    norm_input: bool = False
    mlp_num_layers: int = 1
    mlp_hidden_dim: int = 128
    cosine_embed_dim: int = 64

    def _encode(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
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
                kernel_init=nn.initializers.zeros,
                use_bias=False,
                name=f"mlp_down_{i}",
            )(x)
            x = apply_norm(x, self.norm_type, train)
            x = x + resid
        return x

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool, taus: jnp.ndarray | None = None):
        latent = self._encode(x, train)
        if self.variant == "ctd":
            logits = nn.Dense(self.action_dim * self.num_atoms, kernel_init=nn.initializers.zeros)(
                latent
            )
            return {"logits": logits.reshape(latent.shape[0], self.action_dim, self.num_atoms)}

        if self.variant == "qtd":
            quantiles = nn.Dense(
                self.action_dim * self.num_quantiles,
                kernel_init=nn.initializers.zeros,
            )(latent)
            return {
                "quantiles": quantiles.reshape(
                    latent.shape[0], self.action_dim, self.num_quantiles
                )
            }

        if taus is None:
            taus = (
                jnp.arange(self.policy_num_quantiles, dtype=latent.dtype) + 0.5
            ) / self.policy_num_quantiles
            taus = jnp.broadcast_to(taus[None, :], (latent.shape[0], taus.shape[0]))
        tau_embed = CosineEmbedding(self.cosine_embed_dim)(taus)
        tau_embed = nn.Dense(self.mlp_hidden_dim)(tau_embed)
        hidden = latent[:, None, :] * tau_embed
        quantiles = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(hidden)
        return {"quantiles": jnp.swapaxes(quantiles, -1, -2), "taus": taus}

    def q_values(
        self,
        x: jnp.ndarray,
        train: bool,
        support: jnp.ndarray | None = None,
        taus: jnp.ndarray | None = None,
    ):
        out = self(x, train=train, taus=taus)
        if self.variant == "ctd":
            probs = jax.nn.softmax(out["logits"], axis=-1)
            return jnp.sum(probs * support, axis=-1), out
        return out["quantiles"].mean(axis=-1), out


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def select_action_values(values: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    if values.ndim == actions.ndim + 1:
        action_idx = jnp.expand_dims(actions, axis=-1)
        return jnp.take_along_axis(values, action_idx, axis=-1).squeeze(axis=-1)
    if values.ndim == actions.ndim + 2:
        action_idx = jnp.expand_dims(jnp.expand_dims(actions, axis=-1), axis=-1)
        action_idx = jnp.broadcast_to(action_idx, actions.shape + (1, values.shape[-1]))
        return jnp.take_along_axis(values, action_idx, axis=-2).squeeze(axis=-2)
    raise ValueError(
        f"Cannot select actions with values shape {values.shape} and actions shape {actions.shape}"
    )


def project_categorical(
    next_probs: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    support: jnp.ndarray,
    gamma: float,
    v_min: float,
    v_max: float,
) -> jnp.ndarray:
    num_atoms = support.shape[0]
    flat_probs = next_probs.reshape(-1, num_atoms)
    flat_rewards = rewards.reshape(-1)
    flat_dones = dones.reshape(-1)
    delta_z = (v_max - v_min) / (num_atoms - 1)
    tz = flat_rewards[:, None] + gamma * (1.0 - flat_dones[:, None]) * support[None, :]
    tz = jnp.clip(tz, v_min, v_max)
    b = (tz - v_min) / delta_z
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)
    same = (upper == lower).astype(flat_probs.dtype)
    lower_w = flat_probs * (upper.astype(flat_probs.dtype) - b + same)
    upper_w = flat_probs * (b - lower.astype(flat_probs.dtype))
    batch_idx = jnp.arange(flat_probs.shape[0])[:, None]
    projected = jnp.zeros_like(flat_probs)
    projected = projected.at[batch_idx, lower].add(lower_w)
    projected = projected.at[batch_idx, upper].add(upper_w)
    return projected.reshape(next_probs.shape)


def quantile_huber_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    taus: jnp.ndarray,
    kappa: float,
) -> jnp.ndarray:
    td_error = target[:, None, :] - pred[:, :, None]
    abs_error = jnp.abs(td_error)
    huber = jnp.where(
        abs_error <= kappa,
        0.5 * jnp.square(td_error),
        kappa * (abs_error - 0.5 * kappa),
    )
    tau = taus[:, :, None]
    weight = jnp.abs(tau - (td_error < 0).astype(pred.dtype))
    return (weight * huber / kappa).sum(axis=1).mean()


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
    action_dim = env.action_space(env_params).n
    support = jnp.linspace(
        float(config.get("V_MIN", -10.0)),
        float(config.get("V_MAX", 10.0)),
        int(config.get("NUM_ATOMS", 51)),
    )

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

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
            layer_sq_norms[layer_key] = layer_sq_norms.get(layer_key, 0.0) + sq_norm
        return {k: jnp.sqrt(v) for k, v in layer_sq_norms.items()}

    def clip_grads_per_layer(grads):
        flat_grads = flatten_dict(grads)
        layer_sq_norms = {}
        for key, value in flat_grads.items():
            layer_key = key[:-1] if len(key) > 1 else key
            sq_norm = jnp.sum(jnp.square(value))
            layer_sq_norms[layer_key] = layer_sq_norms.get(layer_key, 0.0) + sq_norm
        max_norm = jnp.asarray(max_grad_norm, dtype=jnp.float32)
        layer_scales = {
            layer_key: jnp.minimum(1.0, max_norm / (jnp.sqrt(sq_norm) + 1e-6))
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
        network = DistributionalMinAtarNetwork(
            action_dim=action_dim,
            variant=config["ALG_VARIANT"],
            num_atoms=int(config.get("NUM_ATOMS", 51)),
            num_quantiles=int(config.get("NUM_QUANTILES", 51)),
            policy_num_quantiles=int(config.get("POLICY_NUM_QUANTILES", 32)),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
            mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
            cosine_embed_dim=config.get("COS_EMBED_DIM", 64),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            init_taus = None
            if config["ALG_VARIANT"] == "iqn":
                init_taus = jnp.zeros((1, int(config.get("NUM_QUANTILES", 32))))
            variables = network.init(rng, init_x, train=False, taus=init_taus)
            clip_transform = (
                optax.identity()
                if per_layer_gradient_clipping
                else optax.clip_by_global_norm(max_grad_norm)
            )
            tx = optax.chain(clip_transform, optax.radam(learning_rate=lr))
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=tx,
            )

        rng, rng_agent = jax.random.split(rng)
        train_state = create_agent(rng_agent)

        def compute_lambda_targets(transitions):
            if config["ALG_VARIANT"] == "iqn":
                q_last, _ = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    transitions.next_obs[-1],
                    train=False,
                    support=support,
                    method=DistributionalMinAtarNetwork.q_values,
                )
            else:
                q_last, _ = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    transitions.next_obs[-1],
                    train=False,
                    support=support,
                    method=DistributionalMinAtarNetwork.q_values,
                )
            last_q = jnp.max(q_last, axis=-1) * (1.0 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q

            def _get_target(carry, transition):
                lambda_returns, next_q = carry
                target_bootstrap = transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                delta = lambda_returns - next_q
                lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                lambda_returns = (1 - transition.done) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            return jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

        def _update_step(runner_state, unused):
            train_state, expl_state, test_metrics, rng = runner_state

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals, _ = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    last_obs,
                    train=False,
                    support=support,
                    method=DistributionalMinAtarNetwork.q_values,
                )
                eps = jnp.full((config["NUM_ENVS"],), eps_scheduler(train_state.n_updates))
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, done, info = vmap_step(config["NUM_ENVS"])(
                    rng_s, env_state, action
                )
                transition = Transition(
                    obs=last_obs,
                    action=action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            rng, rng_rollout = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env, (*expl_state, rng_rollout), None, config["NUM_STEPS"]
            )
            expl_state = tuple(expl_state)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            lambda_targets = compute_lambda_targets(transitions)

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, mb_and_target):
                    train_state, rng = carry
                    minibatch, lambda_target = mb_and_target
                    rng, rng_tau, rng_target_tau = jax.random.split(rng, 3)

                    def _loss_fn(params):
                        variables = {"params": params, "batch_stats": train_state.batch_stats}
                        variant = config["ALG_VARIANT"]
                        if variant == "iqn":
                            bsz = minibatch.obs.shape[0]
                            taus = jax.random.uniform(
                                rng_tau, (bsz, int(config.get("NUM_QUANTILES", 32)))
                            )
                            out, updates = network.apply(
                                variables,
                                minibatch.obs,
                                train=True,
                                taus=taus,
                                mutable=["batch_stats"],
                            )
                            q_vals = out["quantiles"].mean(axis=-1)
                        else:
                            out, updates = network.apply(
                                variables,
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            if variant == "ctd":
                                probs = jax.nn.softmax(out["logits"], axis=-1)
                                q_vals = jnp.sum(probs * support, axis=-1)
                            else:
                                q_vals = out["quantiles"].mean(axis=-1)

                        chosen_q = select_action_values(q_vals, minibatch.action)
                        mean_loss = 0.5 * jnp.square(chosen_q - lambda_target).mean()

                        if variant == "ctd":
                            next_out = network.apply(
                                variables, minibatch.next_obs, train=False
                            )
                            next_probs_all = jax.nn.softmax(next_out["logits"], axis=-1)
                            next_q = jnp.sum(next_probs_all * support, axis=-1)
                            next_actions = jnp.argmax(next_q, axis=-1)
                            next_probs = select_action_values(next_probs_all, next_actions)
                            target_probs = jax.lax.stop_gradient(
                                project_categorical(
                                    next_probs,
                                    minibatch.reward,
                                    minibatch.done.astype(jnp.float32),
                                    support,
                                    float(config["GAMMA"]),
                                    float(config.get("V_MIN", -10.0)),
                                    float(config.get("V_MAX", 10.0)),
                                )
                            )
                            chosen_logits = select_action_values(out["logits"], minibatch.action)
                            dist_loss = -jnp.sum(
                                target_probs * jax.nn.log_softmax(chosen_logits, axis=-1),
                                axis=-1,
                            ).mean()
                        else:
                            if variant == "iqn":
                                bsz = minibatch.next_obs.shape[0]
                                target_taus = jax.random.uniform(
                                    rng_target_tau,
                                    (bsz, int(config.get("NUM_TARGET_QUANTILES", 32))),
                                )
                                next_out = network.apply(
                                    variables,
                                    minibatch.next_obs,
                                    train=False,
                                    taus=target_taus,
                                )
                                tau_values = taus
                            else:
                                next_out = network.apply(
                                    variables, minibatch.next_obs, train=False
                                )
                                tau_values = (
                                    jnp.arange(out["quantiles"].shape[-1], dtype=jnp.float32)
                                    + 0.5
                                ) / out["quantiles"].shape[-1]
                                tau_values = jnp.broadcast_to(
                                    tau_values[None, :],
                                    (minibatch.obs.shape[0], tau_values.shape[0]),
                                )
                            next_quantiles_all = next_out["quantiles"]
                            next_q = next_quantiles_all.mean(axis=-1)
                            next_actions = jnp.argmax(next_q, axis=-1)
                            next_quantiles = select_action_values(
                                next_quantiles_all, next_actions
                            )
                            target_quantiles = jax.lax.stop_gradient(
                                minibatch.reward[:, None]
                                + config["GAMMA"]
                                * (1.0 - minibatch.done.astype(jnp.float32)[:, None])
                                * next_quantiles
                            )
                            pred_quantiles = select_action_values(
                                out["quantiles"], minibatch.action
                            )
                            dist_loss = quantile_huber_loss(
                                pred_quantiles,
                                target_quantiles,
                                tau_values,
                                float(config.get("HUBER_KAPPA", 1.0)),
                            )

                        loss = dist_loss + config.get("AUX_MEAN_LOSS_WEIGHT", 1.0) * mean_loss
                        return loss, (updates, dist_loss, mean_loss, chosen_q)

                    (loss, (updates, dist_loss, mean_loss, chosen_q)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    grad_norm = optax.global_norm(grads)
                    layer_grad_norms = (
                        get_layer_grad_norms(grads)
                        if config.get("LOG_LAYER_GRAD_NORMS", False)
                        else {}
                    )
                    if per_layer_gradient_clipping:
                        grads = clip_grads_per_layer(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (
                        loss,
                        dist_loss,
                        mean_loss,
                        chosen_q.mean(),
                        grad_norm,
                        layer_grad_norms,
                    )

                def preprocess_transition(x, rng_perm):
                    x = x.reshape(-1, *x.shape[2:])
                    x = jax.random.permutation(rng_perm, x)
                    return x.reshape(config["NUM_MINIBATCHES"], minibatch_size, *x.shape[1:])

                rng, rng_perm = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, rng_perm), transitions
                )
                target_mb = preprocess_transition(lambda_targets, rng_perm)
                rng, rng_scan = jax.random.split(rng)
                (train_state, rng), metrics = jax.lax.scan(
                    _learn_phase, (train_state, rng_scan), (minibatches, target_mb)
                )
                return (train_state, rng), metrics

            rng, rng_epochs = jax.random.split(rng)
            (train_state, rng), epoch_metrics = jax.lax.scan(
                _learn_epoch, (train_state, rng_epochs), None, config["NUM_EPOCHS"]
            )
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            loss, dist_loss, mean_loss, qvals, grad_norm, layer_grad_norms = epoch_metrics
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "total_loss": loss.mean(),
                "dist_loss": dist_loss.mean(),
                "td_lambda_loss": mean_loss.mean(),
                "qvals": qvals.mean(),
                "grad_norm": grad_norm.mean(),
            }
            if config.get("LOG_LAYER_GRAD_NORMS", False):
                layer_grad_norms = jax.tree_util.tree_map(lambda x: x.mean(), layer_grad_norms)
                metrics.update({f"grad_norm/{k}": v for k, v in layer_grad_norms.items()})
            done_infos = jax.tree_util.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(infos["returned_episode"], x, jnp.nan), axis=0
                ),
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

            if config["WANDB_MODE"] != "disabled":
                def callback(m, seed_idx_):
                    logged = dict(m)
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        logged = {
                            **{f"seed_{int(seed_idx_) + 1}/{k}": v for k, v in logged.items()},
                            **logged,
                        }
                    wandb.log(logged, step=logged["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            return (train_state, tuple(expl_state), test_metrics, rng), metrics

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return {"returned_episode_returns": jnp.nan}

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals, _ = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    last_obs,
                    train=False,
                    support=support,
                    method=DistributionalMinAtarNetwork.q_values,
                )
                eps = jnp.full((config["TEST_NUM_ENVS"],), config["EPS_TEST"])
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, done, info = vmap_step(config["TEST_NUM_ENVS"])(
                    rng_s, env_state, action
                )
                return (new_env_state, new_obs, rng), info

            rng, rng_reset = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(rng_reset)
            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, rng), None, config["TEST_NUM_STEPS"]
            )
            return jax.tree_util.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(infos["returned_episode"], x, jnp.nan), axis=0
                ),
                infos,
            )

        rng, rng_reset = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
        test_metrics = get_test_metrics(train_state, rng) if config.get("TEST_DURING_TRAINING", False) else {"returned_episode_returns": jnp.nan}
        runner_state = (train_state, (init_obs, env_state), test_metrics, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def maybe_print_network_summary(config):
    if not config.get("PRINT_NETWORK_SUMMARY", False):
        return
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    network = DistributionalMinAtarNetwork(
        action_dim=env.action_space(env_params).n,
        variant=config["ALG_VARIANT"],
        num_atoms=int(config.get("NUM_ATOMS", 51)),
        num_quantiles=int(config.get("NUM_QUANTILES", 51)),
        policy_num_quantiles=int(config.get("POLICY_NUM_QUANTILES", 32)),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
        mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
        cosine_embed_dim=config.get("COS_EMBED_DIM", 64),
    )
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    init_taus = None
    if config["ALG_VARIANT"] == "iqn":
        init_taus = jnp.zeros((1, int(config.get("NUM_QUANTILES", 32))))
    print(network.tabulate(jax.random.PRNGKey(config["SEED"]), init_x, False, init_taus))


def single_run(config):
    config = {**config, **config["alg"]}
    print(config)
    maybe_print_network_summary(config)
    alg_name = config.get("ALG_NAME", f'{config["ALG_VARIANT"]}_minatar')
    env_name = config["ENV_NAME"]
    tags = [alg_name.upper(), env_name.upper(), f"jax_{jax.__version__}"]
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
    seed = config["SEED"] if config.get("FIXED_SEED", True) else random.randint(0, 100_000)
    rng = jax.random.PRNGKey(seed)
    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    seed_idxs = jnp.arange(config["NUM_SEEDS"], dtype=jnp.int32)
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs, seed_idxs))
    print(f"Took {time.time() - t0} seconds to complete.")

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
        "name": f'{default_config.get("ALG_NAME", "distributional")}_{default_config["ENV_NAME"]}',
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
