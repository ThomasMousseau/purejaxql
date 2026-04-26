"""Distributional PQN-RNN baselines on Craftax.

The rollout, LSTM trunk, BatchRenorm usage, optimizer, memory window, and
TD(lambda) mean objective mirror ``pqn_rnn_craftax.py``. CTD/C51, QTD/QR-DQN,
and IQN differ only in their distributional heads and Bellman losses.
"""

from __future__ import annotations

import copy
import os
import time
from functools import partial
from typing import Any

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.utils.batch_renorm import BatchRenorm
from purejaxql.utils.craftax_wrappers import (
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]
        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree_util.tree_map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
            init_rnn_state,
            rnn_state,
        )
        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


def apply_norm(x: jnp.ndarray, norm_type: str, train: bool):
    if norm_type == "layer_norm":
        return nn.LayerNorm()(x)
    if norm_type == "batch_norm":
        return BatchRenorm(use_running_average=not train)(x)
    return x


class CosineEmbedding(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, taus: jnp.ndarray) -> jnp.ndarray:
        basis = jnp.arange(1, self.hidden_dim + 1, dtype=taus.dtype)
        x = jnp.cos(jnp.pi * taus[..., None] * basis)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        return nn.relu(x)


class RNNDistributionalQNetwork(nn.Module):
    action_dim: int
    variant: str
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    add_last_action: bool = False
    num_atoms: int = 51
    num_quantiles: int = 51
    policy_num_quantiles: int = 32
    cosine_embed_dim: int = 64

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
    def __call__(self, hidden, obs, done, last_action, train: bool = False, taus=None):
        new_hidden, latent = self.encode_latent(hidden, obs, done, last_action, train=train)
        if self.variant == "ctd":
            logits = nn.Dense(
                self.action_dim * self.num_atoms,
                kernel_init=nn.initializers.zeros,
            )(latent)
            return new_hidden, {
                "logits": logits.reshape(*latent.shape[:-1], self.action_dim, self.num_atoms)
            }
        if self.variant == "qtd":
            quantiles = nn.Dense(
                self.action_dim * self.num_quantiles,
                kernel_init=nn.initializers.zeros,
            )(latent)
            return new_hidden, {
                "quantiles": quantiles.reshape(
                    *latent.shape[:-1], self.action_dim, self.num_quantiles
                )
            }

        t, b, d = latent.shape
        flat = latent.reshape(t * b, d)
        if taus is None:
            tau_grid = (
                jnp.arange(self.policy_num_quantiles, dtype=flat.dtype) + 0.5
            ) / self.policy_num_quantiles
            taus = jnp.broadcast_to(tau_grid[None, None, :], (t, b, tau_grid.shape[0]))
        flat_taus = taus.reshape(t * b, taus.shape[-1])
        tau_embed = CosineEmbedding(self.cosine_embed_dim)(flat_taus)
        tau_embed = nn.Dense(self.hidden_size)(tau_embed)
        hidden_tau = flat[:, None, :] * tau_embed
        quantiles = nn.Dense(self.action_dim, kernel_init=nn.initializers.zeros)(hidden_tau)
        quantiles = jnp.swapaxes(quantiles, -1, -2).reshape(
            t, b, self.action_dim, taus.shape[-1]
        )
        return new_hidden, {"quantiles": quantiles, "taus": taus}

    def q_values(self, hidden, obs, done, last_action, train: bool = False, support=None):
        new_hidden, out = self(hidden, obs, done, last_action, train=train)
        if self.variant == "ctd":
            probs = jax.nn.softmax(out["logits"], axis=-1)
            q_vals = jnp.sum(probs * support, axis=-1)
        else:
            q_vals = out["quantiles"].mean(axis=-1)
        return new_hidden, q_vals, out

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_vals: chex.Array


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


def project_categorical(next_probs, rewards, dones, support, gamma, v_min, v_max):
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


def quantile_huber_loss(pred, target, taus, kappa):
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    taus = taus.reshape(-1, taus.shape[-1])
    td_error = target[:, None, :] - pred[:, :, None]
    abs_error = jnp.abs(td_error)
    huber = jnp.where(
        abs_error <= kappa,
        0.5 * jnp.square(td_error),
        kappa * (abs_error - 0.5 * kappa),
    )
    weight = jnp.abs(taus[:, :, None] - (td_error < 0).astype(pred.dtype))
    return (weight * huber / kappa).sum(axis=1).mean()


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
    support = jnp.linspace(
        float(config.get("V_MIN", -100.0)),
        float(config.get("V_MAX", 100.0)),
        int(config.get("NUM_ATOMS", 51)),
    )

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        return jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(rng_a, greedy_actions.shape, 0, q_vals.shape[-1]),
            greedy_actions,
        )

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
        network = RNNDistributionalQNetwork(
            action_dim=action_dim,
            variant=config["ALG_VARIANT"],
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
            num_atoms=int(config.get("NUM_ATOMS", 51)),
            num_quantiles=int(config.get("NUM_QUANTILES", 51)),
            policy_num_quantiles=int(config.get("POLICY_NUM_QUANTILES", 32)),
            cosine_embed_dim=int(config.get("COS_EMBED_DIM", 64)),
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, *obs_shape)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            )
            init_hs = network.initialize_carry(1)
            init_taus = None
            if config["ALG_VARIANT"] == "iqn":
                init_taus = jnp.zeros((1, 1, int(config.get("NUM_QUANTILES", 32))))
            variables = network.init(rng, init_hs, *init_x, train=False, taus=init_taus)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            return CustomTrainState.create(
                apply_fn=network.apply,
                params=variables["params"],
                batch_stats=variables["batch_stats"],
                tx=tx,
            )

        rng, rng_agent = jax.random.split(rng)
        train_state = create_agent(rng_agent)

        def compute_lambda_targets(last_q, q_vals, reward, done):
            def _get_target(carry, rew_q_done):
                reward, q, done = rew_q_done
                lambda_returns, next_q = carry
                target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                delta = lambda_returns - next_q
                lambda_returns = target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                lambda_returns = (1 - done) * lambda_returns + done * reward
                next_q = jnp.max(q, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
            last_q = jnp.max(q_vals[-1], axis=-1)
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                reverse=True,
            )
            return jnp.concatenate([targets, lambda_returns[jnp.newaxis]])

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
                    method=RNNDistributionalQNetwork.q_values,
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
                    rng, rng_tau = jax.random.split(rng)
                    hs = jax.tree_util.tree_map(lambda x: x[0], minibatch.last_hs)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )

                    def _loss_fn(params):
                        variables = {"params": params, "batch_stats": train_state.batch_stats}
                        variant = config["ALG_VARIANT"]
                        taus = None
                        if variant == "iqn":
                            taus = jax.random.uniform(
                                rng_tau,
                                (
                                    minibatch.obs.shape[0],
                                    minibatch.obs.shape[1],
                                    int(config.get("NUM_QUANTILES", 32)),
                                ),
                            )
                        (_hs, out), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(variables, hs, *agent_in, taus=taus)
                        if variant == "ctd":
                            probs = jax.nn.softmax(out["logits"], axis=-1)
                            q_vals = jnp.sum(probs * support, axis=-1)
                        else:
                            q_vals = out["quantiles"].mean(axis=-1)

                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        last_q = target_q_vals[-1].max(axis=-1)
                        lambda_targets = compute_lambda_targets(
                            last_q,
                            target_q_vals[:-1],
                            minibatch.reward[:-1],
                            minibatch.done[:-1],
                        ).reshape(-1)
                        chosen_q_all = select_action_values(q_vals, minibatch.action)
                        chosen_q = chosen_q_all[:-1].reshape(-1)
                        mean_loss = 0.5 * jnp.square(chosen_q - lambda_targets).mean()

                        rewards = minibatch.reward[:-1]
                        dones = minibatch.done[:-1].astype(jnp.float32)
                        if variant == "ctd":
                            next_probs_all = jax.lax.stop_gradient(probs[1:])
                            next_q = jnp.sum(next_probs_all * support, axis=-1)
                            next_actions = jnp.argmax(next_q, axis=-1)
                            next_probs = select_action_values(next_probs_all, next_actions)
                            target_probs = project_categorical(
                                next_probs,
                                rewards,
                                dones,
                                support,
                                float(config["GAMMA"]),
                                float(config.get("V_MIN", -100.0)),
                                float(config.get("V_MAX", 100.0)),
                            )
                            chosen_logits = select_action_values(
                                out["logits"][:-1], minibatch.action[:-1]
                            )
                            dist_loss = -jnp.sum(
                                jax.lax.stop_gradient(target_probs)
                                * jax.nn.log_softmax(chosen_logits, axis=-1),
                                axis=-1,
                            ).mean()
                        else:
                            next_quantiles_all = jax.lax.stop_gradient(out["quantiles"][1:])
                            next_q = next_quantiles_all.mean(axis=-1)
                            next_actions = jnp.argmax(next_q, axis=-1)
                            next_quantiles = select_action_values(
                                next_quantiles_all, next_actions
                            )
                            target_quantiles = jax.lax.stop_gradient(
                                rewards[..., None]
                                + config["GAMMA"] * (1.0 - dones[..., None]) * next_quantiles
                            )
                            pred_quantiles = select_action_values(
                                out["quantiles"][:-1], minibatch.action[:-1]
                            )
                            if variant == "iqn":
                                tau_values = taus[:-1]
                            else:
                                tau_grid = (
                                    jnp.arange(pred_quantiles.shape[-1], dtype=jnp.float32)
                                    + 0.5
                                ) / pred_quantiles.shape[-1]
                                tau_values = jnp.broadcast_to(
                                    tau_grid,
                                    pred_quantiles.shape[:2] + (tau_grid.shape[0],),
                                )
                            dist_loss = quantile_huber_loss(
                                pred_quantiles,
                                target_quantiles,
                                tau_values,
                                float(config.get("HUBER_KAPPA", 1.0)),
                            )

                        loss = dist_loss + config.get("AUX_MEAN_LOSS_WEIGHT", 1.0) * mean_loss
                        return loss, (updates, dist_loss, mean_loss, chosen_q_all)

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params
                    )
                    updates, dist_loss, mean_loss, qvals = aux
                    grad_norm = optax.global_norm(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (
                        loss,
                        dist_loss,
                        mean_loss,
                        qvals.mean(),
                        grad_norm,
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
            loss, dist_loss, mean_loss, qvals, grad_norm = epoch_metrics
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
                        # Keep W&B logging scalar-only to avoid histogram
                        # auto-conversion on vector metrics.
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
                    logged = sanitized
                    wandb.log(logged, step=logged["update_steps"])

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
                    method=RNNDistributionalQNetwork.q_values,
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
        test_metrics = get_test_metrics(train_state, rng) if config.get("TEST_DURING_TRAINING", False) else {"returned_episode_returns": jnp.nan}

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
                method=RNNDistributionalQNetwork.q_values,
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
    alg_name = config.get("ALG_NAME", f'{config["ALG_VARIANT"]}_rnn')
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
