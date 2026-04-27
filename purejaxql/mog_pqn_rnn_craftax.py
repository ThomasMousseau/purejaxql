"""
MoG-PQN-RNN on Craftax — **aligned with ``pqn_rnn_craftax``** (single network, BatchRenorm,
Q(λ)-style *exploration* via Q=Σπμ, RAdam, memory window, same rollout/minibatch layout).

**What differs from PQN-RNN** (the MoG novelty):
  - Trunk ends in **MoG heads** (π, μ, σ) instead of scalar Q logits.
  - **Fused dual loss** (every update applies both):
      1. **TD(λ) loss** — scalar λ-returns vs **Σ_k π_k μ_k** over the full 128-step sequence,
         anchoring component means and mixture weights for stable greedy policy.
      2. **MoG CF loss** — one-step characteristic-function Bellman MSE (stop-gradient bootstrap
         from the same weights; no target net), shaping σ to capture local return variance.
      ``total_loss = cf_loss + AUX_MEAN_LOSS_WEIGHT * lambda_loss``

  See ``AUX_MEAN_LOSS_WEIGHT`` in the alg YAML (default 1.0).

This RNN file deliberately matches PQN-RNN training mechanics for controlled comparison.

Run::

  uv run python -m purejaxql.mog_pqn_rnn_craftax --config-path purejaxql/config \\
    --config-name config +alg=mog_pqn_rnn_craftax
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
from purejaxql.utils.save_load import save_train_metrics_npz
from purejaxql.utils.craftax_wrappers import (
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from purejaxql.utils.mog_cf import build_mog_cf, mog_q_values, sample_frequencies


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


class RNNMoGNetwork(nn.Module):
    """Same trunk contract as ``RNNQNetwork``; MoG heads instead of a single Q vector."""

    action_dim: int
    num_components: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    add_last_action: bool = False

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda z: nn.LayerNorm()(z)
        elif self.norm_type == "batch_norm":
            normalize = lambda z: BatchRenorm(use_running_average=not train)(z)
        else:
            normalize = lambda z: z

        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            _ = BatchRenorm(use_running_average=not train)(x)

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        if self.add_last_action:
            la = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, la], axis=-1)

        new_hidden = []
        for _ in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[_], rnn_in)
            new_hidden.append(hidden_aux)

        a, m = self.action_dim, self.num_components
        pi_logits = nn.Dense(a * m)(x)
        pi = jax.nn.softmax(pi_logits.reshape(*pi_logits.shape[:-1], a, m), axis=-1)
        mu = nn.Dense(a * m)(x).reshape(*x.shape[:-1], a, m)
        sigma = jax.nn.softplus(nn.Dense(a * m)(x).reshape(*x.shape[:-1], a, m))
        return new_hidden, (pi, mu, sigma)

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


def mog_cf_loss_single_network(
    pi: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    omegas: jnp.ndarray,
    gamma: float,
    is_divided_by_sigma_squared: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-step MoG CF loss; bootstrap from stop-gradient MoG at t+1 (same weights as PQN-RNN: no target net).

    If ``is_divided_by_sigma_squared`` is true, the per-ω MSE is scaled by 1/ω² (with a small floor on
    |ω|) to match a sample-based surrogate for ∫|φ(ω)−φ̂(ω)|²/ω² dω. The config key is
    ``IS_DIVIDED_BY_SIGMA_SQUARED``; the weighting uses the CF frequency ω, not MoG component stds.
    """
    sg = jax.lax.stop_gradient
    pi_tp1 = pi[1:]
    mu_tp1 = mu[1:]
    sig_tp1 = sigma[1:]

    q_next = jnp.sum(sg(pi_tp1) * sg(mu_tp1), axis=-1)
    next_actions = jnp.argmax(q_next, axis=-1)

    tb, b, _a, _m = pi_tp1.shape
    pi_sel = pi_tp1[jnp.arange(tb)[:, None], jnp.arange(b), next_actions, :]
    mu_sel = mu_tp1[jnp.arange(tb)[:, None], jnp.arange(b), next_actions, :]
    sig_sel = sig_tp1[jnp.arange(tb)[:, None], jnp.arange(b), next_actions, :]
    pi_sel = sg(pi_sel)
    mu_sel = sg(mu_sel)
    sig_sel = sg(sig_sel)

    r = rewards[:-1]
    d = dones[:-1].astype(jnp.float32)
    gammas = gamma * (1.0 - d)
    bellman_mu = r[..., None] + gammas[..., None] * mu_sel
    bellman_sigma = gammas[..., None] * sig_sel
    bellman_pi = pi_sel

    td_target = build_mog_cf(
        bellman_pi[:, :, None, :],
        bellman_mu[:, :, None, :],
        bellman_sigma[:, :, None, :],
        omegas,
    )
    td_target = td_target[:, :, :, 0]
    td_target = sg(td_target)

    pi_a = pi[:-1]
    mu_a = mu[:-1]
    sig_a = sigma[:-1]
    act = actions[:-1]
    online_phi = build_mog_cf(pi_a, mu_a, sig_a, omegas)
    online_phi_sel = jnp.take_along_axis(
        online_phi, act[..., None, None], axis=-1
    ).squeeze(-1)

    err = online_phi_sel - td_target
    err_squared = jnp.abs(err) ** 2
    if is_divided_by_sigma_squared:
        denominator_sigma_squared = 1.0 / jnp.maximum(omegas, 1e-8) ** 2
        denominator_sigma_squared = denominator_sigma_squared[None, None, :]
        err_squared = err_squared * denominator_sigma_squared
    loss = jnp.mean(err_squared)
    im = jnp.abs(jnp.imag(err))
    if is_divided_by_sigma_squared:
        im = im * denominator_sigma_squared
    cf_im = jnp.mean(im)
    return loss, cf_im


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
            num_envs=int(config.get("TEST_NUM_ENVS", config["NUM_ENVS"])),
            reset_ratio=min(
                config["OPTIMISTIC_RESET_RATIO"],
                int(config.get("TEST_NUM_ENVS", config["NUM_ENVS"])),
            ),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(
            log_env, num_envs=int(config.get("TEST_NUM_ENVS", config["NUM_ENVS"]))
        )

    action_dim = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

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

        network = RNNMoGNetwork(
            action_dim=action_dim,
            num_components=int(config["NUM_COMPONENTS"]),
            hidden_size=int(config.get("HIDDEN_SIZE", 512)),
            num_layers=int(config.get("NUM_LAYERS", 4)),
            num_rnn_layers=int(config.get("NUM_RNN_LAYERS", 1)),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
        )

        def create_agent(rng_init):
            init_x = (
                jnp.zeros((1, 1, *obs_shape)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1), dtype=jnp.int32),
            )
            init_hs = network.initialize_carry(1)
            network_variables = network.init(rng_init, init_hs, *init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        def _update_step(runner_state, unused):
            train_state, memory_transitions, expl_state, test_metrics, rng = runner_state

            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]
                _done = last_done[np.newaxis]
                _last_action = last_action[np.newaxis]

                new_hs, (pi, mu, sigma) = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )
                q_vals = mog_q_values(pi, mu).squeeze(0)

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

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
                return (new_hs, new_obs, new_done, new_action, new_env_state, rng), (
                    transition,
                    info,
                )

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
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
                    hs = jax.tree_util.tree_map(lambda x: x[0], minibatch.last_hs)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )

                    def _compute_targets(last_q, q_vals, reward, done):
                        def _get_target(lambda_returns_and_next_q, rew_q_done):
                            reward, q, done = rew_q_done
                            lambda_returns, next_q = lambda_returns_and_next_q
                            target_bootstrap = (
                                reward + config["GAMMA"] * (1 - done) * next_q
                            )
                            delta = lambda_returns - next_q
                            lambda_returns = (
                                target_bootstrap
                                + config["GAMMA"] * config["LAMBDA"] * delta
                            )
                            lambda_returns = (
                                1 - done
                            ) * lambda_returns + done * reward
                            next_q = jnp.max(q, axis=-1)
                            return (lambda_returns, next_q), lambda_returns

                        lambda_returns = (
                            reward[-1]
                            + config["GAMMA"] * (1 - done[-1]) * last_q
                        )
                        last_q = jnp.max(q_vals[-1], axis=-1)
                        _, targets = jax.lax.scan(
                            _get_target,
                            (lambda_returns, last_q),
                            jax.tree_util.tree_map(
                                lambda x: x[:-1],
                                (reward, q_vals, done),
                            ),
                            reverse=True,
                        )
                        targets = jnp.concatenate(
                            [targets, lambda_returns[np.newaxis]]
                        )
                        return targets

                    def _loss_fn(params):
                        omegas = sample_frequencies(
                            rng_omega,
                            int(config["NUM_OMEGA_SAMPLES"]),
                            float(config["OMEGA_MAX"]),
                        )
                        (_, (pi, mu, sigma)), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(
                            {
                                "params": params,
                                "batch_stats": train_state.batch_stats,
                            },
                            hs,
                            *agent_in,
                        )

                        # TD(λ) loss — anchors component means over the full 128-step horizon
                        q_vals = mog_q_values(pi, mu)
                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        last_q = target_q_vals[-1].max(axis=-1)
                        target = _compute_targets(
                            last_q,
                            target_q_vals[:-1],
                            minibatch.reward[:-1],
                            minibatch.done[:-1],
                        ).reshape(-1)
                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        chosen_action_qvals = chosen_action_qvals[:-1].reshape(-1)
                        lambda_loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        # MoG CF loss — shapes σ via 1-step characteristic-function Bellman
                        cf_loss, cf_im = mog_cf_loss_single_network(
                            pi,
                            mu,
                            sigma,
                            minibatch.action,
                            minibatch.reward,
                            minibatch.done.astype(jnp.float32),
                            omegas,
                            float(config["GAMMA"]),
                            bool(
                                config.get("IS_DIVIDED_BY_SIGMA_SQUARED")
                            ),
                        )

                        total_loss = cf_loss + config.get("AUX_MEAN_LOSS_WEIGHT", 1.0) * lambda_loss
                        return total_loss, (updates, cf_loss, lambda_loss, cf_im, chosen_action_qvals)

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params
                    )
                    grad_norm = optax.global_norm(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    updates, cf_loss, lambda_loss, cf_im, qvals = aux
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, cf_loss, lambda_loss, cf_im, qvals, grad_norm)

                def preprocess_transition(x, rng_perm):
                    x = jax.random.permutation(rng_perm, x, axis=1)
                    x = x.reshape(
                        x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
                    )
                    return jnp.swapaxes(x, 0, 1)

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda z: preprocess_transition(z, _rng),
                    memory_transitions,
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), scan_out = jax.lax.scan(
                    _learn_phase, (train_state, _rng), minibatches
                )
                loss, cf_loss, lambda_loss, cf_im, qvals, grad_norm = scan_out
                return (train_state, rng), (loss, cf_loss, lambda_loss, cf_im, qvals, grad_norm)

            rng, _rng = jax.random.split(rng)
            scan_epochs = jax.lax.scan(_learn_epoch, (train_state, _rng), None, config["NUM_EPOCHS"])
            (train_state, rng), epoch_metrics = scan_epochs

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)

            loss, cf_loss, lambda_loss, cf_im, qvals, grad_norm = epoch_metrics
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "total_loss": loss.mean(),
                "mog_cf_loss": cf_loss.mean(),
                "td_lambda_loss": lambda_loss.mean(),
                "cf_loss_imag": cf_im.mean(),
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

            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            if config["WANDB_MODE"] != "disabled":
                _sps_state = {"t": time.time(), "s": 0}

                def callback(m, seed_idx_):
                    _iv = max(int(config.get("WANDB_LOG_INTERVAL", 100)), 1)
                    if int(m["update_steps"]) % _iv != 0:
                        return
                    now = time.time()
                    steps = int(m["env_step"])
                    dt = now - _sps_state["t"]
                    sps = (steps - _sps_state["s"]) / max(dt, 1e-9)
                    _sps_state["t"], _sps_state["s"] = now, steps
                    logged = dict(m)
                    logged["sps"] = sps
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        logged = {
                            **{f"seed_{int(seed_idx_) + 1}/{k}": v for k, v in logged.items()},
                            **logged,
                        }
                    wandb.log(logged, step=logged["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            runner_state = (
                train_state,
                memory_transitions,
                tuple(expl_state),
                test_metrics,
                rng,
            )
            return runner_state, metrics

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(step_state, _):
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]
                _done = last_done[np.newaxis]
                _last_action = last_action[np.newaxis]
                new_hs, (pi, mu, sigma) = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )
                q_vals = mog_q_values(pi, mu).squeeze(0)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    rng_s, env_state, new_action, env_params
                )
                return (new_hs, new_obs, new_done, new_action, new_env_state, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)
            init_done = jnp.zeros((config["TEST_NUM_ENVS"],), dtype=bool)
            init_action = jnp.zeros((config["TEST_NUM_ENVS"],), dtype=jnp.int32)
            init_hs = network.initialize_carry(config["TEST_NUM_ENVS"])
            step_state = (
                init_hs,
                init_obs,
                init_done,
                init_action,
                env_state,
                _rng,
            )
            _, infos = jax.lax.scan(_greedy_env_step, step_state, None, config["TEST_NUM_STEPS"])
            return jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / jnp.maximum(infos["returned_episode"].sum(), 1.0),
                infos,
            )

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        init_dones = jnp.zeros((config["NUM_ENVS"],), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)
        init_hs = network.initialize_carry(config["NUM_ENVS"])
        expl_state = (init_hs, obs, init_dones, init_action, env_state)

        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]
            _done = last_done[np.newaxis]
            _last_action = last_action[np.newaxis]
            new_hs, (pi, mu, sigma) = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _done,
                _last_action,
                train=False,
            )
            q_vals = mog_q_values(pi, mu).squeeze(0)
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.0)
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
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
            return (
                new_hs,
                new_obs,
                new_done,
                new_action,
                new_env_state,
                rng,
            ), transition

        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (*expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        expl_state = tuple(expl_state)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "mog_pqn_rnn")
    env_name = config["ENV_NAME"]

    _tags = [
        alg_name.upper(),
        env_name.upper().replace(" ", "_"),
        f"jax_{jax.__version__}",
    ]
    _et = config.get("EXPERIMENT_TAG")
    if _et:
        _tags.append(str(_et))
    _extra = config.get("WANDB_EXTRA_TAGS")
    if _extra:
        _tags.extend(str(x) for x in _extra)

    _wk = {}
    _g = os.environ.get("WANDB_RUN_GROUP") or config.get("WANDB_RUN_GROUP")
    if _g:
        _wk["group"] = str(_g)
    _jt = os.environ.get("WANDB_JOB_TYPE") or config.get("WANDB_JOB_TYPE")
    if _jt:
        _wk["job_type"] = str(_jt)
    _nt = os.environ.get("WANDB_NOTES") or config.get("WANDB_NOTES")
    if _nt:
        _wk["notes"] = str(_nt)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=_tags,
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
        **_wk,
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

    train_metrics = outs.get("metrics")
    if train_metrics is not None:
        metrics_path = config.get("METRICS_SAVE_PATH")
        if metrics_path is None and config.get("WANDB_MODE") == "disabled":
            _base = config.get("SAVE_PATH") or "."
            metrics_path = os.path.join(
                _base,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_train_metrics.npz',
            )
        if metrics_path:
            save_train_metrics_npz(train_metrics, metrics_path)
            print(f"Saved training metrics to {metrics_path}")

    _save_root = config.get("SAVE_PATH")
    if _save_root is not None and str(_save_root).strip() != "":
        from purejaxql.utils.save_load import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(os.path.abspath(str(_save_root)), env_name)
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
            print(f"Saved final training checkpoint: {save_path}")


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "mog_pqn_rnn")
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
