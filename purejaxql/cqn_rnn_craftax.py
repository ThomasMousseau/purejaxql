"""
CQN-RNN on Craftax — **aligned with ``pqn_rnn_craftax``** (same env, rollout, LSTM trunk,
BatchRenorm, RAdam, memory window, minibatch layout), but **training loss** is the
characteristic-function n-step objective from ``cqn_minatar`` (with RNN-correct bootstrap
latents). Exploration uses the mean head at ω=0 (same as feedforward CQN).

Run::

  uv run python -m purejaxql.cqn_rnn_craftax --config-path purejaxql/config \\
    --config-name config +alg=cqn_rnn_craftax
"""

from __future__ import annotations

import copy
import os
import time
from functools import partial
from typing import Any, Union

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
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
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

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
        new_rnn_state, y = nn.OptimizedLSTMCell(
            hidden_size,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class FrequencyEmbedding(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, omega: jnp.ndarray) -> jnp.ndarray:
        omega = jnp.asarray(omega)
        basis = jnp.arange(1, self.hidden_dim + 1, dtype=omega.dtype)
        return jnp.cos(jnp.pi * omega[..., None] * basis)


def apply_norm(x: jnp.ndarray, norm_type: str, train: bool):
    if norm_type == "layer_norm":
        return nn.LayerNorm(dtype=x.dtype, param_dtype=jnp.float32)(x)
    if norm_type == "batch_norm":
        return BatchRenorm(use_running_average=not train)(x)
    return x


class RNNCharacteristicQNetwork(nn.Module):
    """LSTM trunk (same contract as ``RNNQNetwork``) + characteristic-function heads."""

    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    add_last_action: bool = False
    mlp_num_layers: int = 1
    mlp_hidden_dim: int = 128
    sigma_min: float = 1e-6
    compute_dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    def _prepare_omegas(
        self, omega: Union[jnp.ndarray, float], batch_size: int, dtype: jnp.dtype
    ) -> jnp.ndarray:
        omega = jnp.asarray(omega, dtype=dtype)
        if omega.ndim == 0:
            omega = jnp.broadcast_to(omega[None, None], (batch_size, 1))
        elif omega.ndim == 1:
            omega = jnp.broadcast_to(omega[None, :], (batch_size, omega.shape[0]))
        elif omega.ndim == 2:
            if omega.shape[0] == 1 and batch_size != 1:
                omega = jnp.broadcast_to(omega, (batch_size, omega.shape[1]))
            elif omega.shape[0] != batch_size:
                raise ValueError(
                    f"omega batch {omega.shape[0]} vs observation batch {batch_size}."
                )
        else:
            raise ValueError(f"Invalid omega rank: {omega.ndim}")
        return omega

    def _shared_trunk(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = x.reshape(-1, x.shape[-1])
        for i in range(self.mlp_num_layers):
            resid = x
            x = nn.Dense(
                4 * self.mlp_hidden_dim,
                kernel_init=nn.initializers.he_normal(),
                use_bias=False,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name=f"mlp_up_{i}",
            )(x)
            x = nn.relu(x)
            x = nn.Dense(
                self.mlp_hidden_dim,
                kernel_init=nn.initializers.zeros,
                use_bias=False,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
                name=f"mlp_down_{i}",
            )(x)
            x = apply_norm(x, self.norm_type, train)
            x = x + resid
        return x

    @nn.compact
    def encode_latent(
        self,
        hidden,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        last_action: jnp.ndarray,
        train: bool = False,
    ):
        """Returns (new_hidden, state_latent) with state_latent (T, B, mlp_hidden_dim)."""
        x = obs
        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            _ = BatchRenorm(use_running_average=not train)(x)
        x = x.astype(self.compute_dtype)

        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )(x)
            x = apply_norm(x, self.norm_type, train)
            x = nn.relu(x)

        if self.add_last_action:
            la = jax.nn.one_hot(last_action, self.action_dim, dtype=self.compute_dtype)
            x = jnp.concatenate([x, la], axis=-1)

        new_hidden = []
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN(
                compute_dtype=self.compute_dtype,
                param_dtype=self.param_dtype,
            )(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        state_latent = nn.Dense(
            self.mlp_hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="state_proj",
        )(x)
        return new_hidden, state_latent

    @nn.compact
    def apply_cf_heads(
        self,
        state_latent: jnp.ndarray,
        omega: jnp.ndarray,
        train: bool,
        compute_cf: bool = True,
    ):
        """state_latent (N, D), omega (N, L). Returns dict mean, cf, scale."""
        batch_size = state_latent.shape[0]
        dtype = state_latent.dtype
        if compute_cf:
            omega = self._prepare_omegas(omega, batch_size, dtype)
            freq_latent = FrequencyEmbedding(hidden_dim=self.mlp_hidden_dim)(omega)
            joint_latent = state_latent[:, None, :] * freq_latent
            hidden = self._shared_trunk(joint_latent, train)
            hidden = hidden.reshape(state_latent.shape[0], omega.shape[1], -1)
        else:
            hidden = self._shared_trunk(state_latent[:, None, :], train)
            hidden = hidden.reshape(state_latent.shape[0], 1, -1)

        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="mean_head",
        )(hidden)
        if not compute_cf:
            return {"mean": mean}

        scale = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
            name="scale_head",
        )(hidden)
        scale = nn.softplus(scale) + self.sigma_min

        kappa = jnp.square(omega) / (1.0 + jnp.square(omega))
        log_magnitude = -0.5 * kappa[..., None] * scale
        phase = omega[..., None] * mean
        cf = jnp.exp(log_magnitude + 1j * phase)

        return {"mean": mean, "scale": scale, "cf": cf}

    @nn.compact
    def __call__(
        self,
        hidden,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        last_action: jnp.ndarray,
        omega: Union[jnp.ndarray, None],
        train: bool = False,
        compute_cf: bool = True,
    ):
        new_hidden, state_latent = self.encode_latent(
            hidden, obs, done, last_action, train=train
        )
        t, b, _ = state_latent.shape
        flat = state_latent.reshape(t * b, -1)
        if omega is None:
            out = self.apply_cf_heads(
                flat,
                jnp.zeros((t * b, 1), dtype=flat.dtype),
                train,
                compute_cf=False,
            )
            out = jax.tree_util.tree_map(
                lambda z: z.reshape(t, b, *z.shape[1:]),
                out,
            )
            return new_hidden, out

        om = omega.reshape(t * b, -1)
        out = self.apply_cf_heads(flat, om, train, compute_cf=compute_cf)
        out = jax.tree_util.tree_map(
            lambda z: z.reshape(t, b, *z.shape[1:]),
            out,
        )
        return new_hidden, out

    def q_values(self, hidden, obs, done, last_action, train: bool = False):
        new_hidden, out = self(
            hidden,
            obs,
            done,
            last_action,
            omega=None,
            train=train,
            compute_cf=False,
        )
        return new_hidden, out["mean"][..., 0, :]

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    next_obs: chex.Array
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


def sample_omegas(rng, batch_size: int, config) -> jnp.ndarray:
    num_omegas = config["NUM_OMEGAS"]
    shape = (batch_size, num_omegas)
    sampler = config.get("OMEGA_SAMPLER", "low_freq_mix")
    omega_scale = jnp.asarray(config.get("OMEGA_SCALE", 1.0), dtype=jnp.float32)
    omega_clip = jnp.asarray(config.get("OMEGA_CLIP", jnp.inf), dtype=jnp.float32)

    if sampler == "uniform":
        base_omegas = jax.random.uniform(rng, shape, minval=0.0, maxval=1.0)
    elif sampler == "beta":
        alpha = jnp.asarray(config.get("OMEGA_BETA_ALPHA", 1.0), dtype=jnp.float32)
        beta = jnp.asarray(config.get("OMEGA_BETA_BETA", 3.0), dtype=jnp.float32)
        base_omegas = jax.random.beta(rng, alpha, beta, shape=shape)
    elif sampler == "low_freq_mix":
        rng_mix, rng_beta, rng_uniform = jax.random.split(rng, 3)
        alpha = jnp.asarray(config.get("OMEGA_BETA_ALPHA", 1.0), dtype=jnp.float32)
        beta = jnp.asarray(config.get("OMEGA_BETA_BETA", 3.0), dtype=jnp.float32)
        uniform_mix_prob = jnp.asarray(
            config.get("OMEGA_UNIFORM_MIX_PROB", 0.25), dtype=jnp.float32
        )
        beta_omegas = jax.random.beta(rng_beta, alpha, beta, shape=shape)
        uniform_omegas = jax.random.uniform(rng_uniform, shape, minval=0.0, maxval=1.0)
        use_uniform = jax.random.uniform(rng_mix, shape) < uniform_mix_prob
        base_omegas = jnp.where(use_uniform, uniform_omegas, beta_omegas)
    else:
        raise ValueError(f"Unsupported omega sampler: {sampler!r}")

    omegas = omega_scale * base_omegas
    return jnp.clip(omegas, 0.0, omega_clip)


def select_action_values(values: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    if values.ndim < 2:
        raise ValueError(f"Unsupported tensor rank: {values.ndim}")
    action_idx = actions
    for _ in range(values.ndim - actions.ndim - 1):
        action_idx = jnp.expand_dims(action_idx, axis=-1)
    action_idx = jnp.expand_dims(action_idx, axis=-1)
    action_idx = jnp.broadcast_to(action_idx, values.shape[:-1] + (1,))
    return jnp.take_along_axis(values, action_idx, axis=-1).squeeze(axis=-1)


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

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

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        return jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),
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
        compute_dtype = jnp.bfloat16 if config.get("USE_BF16", False) else jnp.float32

        network = RNNCharacteristicQNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 512),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
            mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
            mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
            sigma_min=config.get("SIGMA_MIN", 1e-6),
            compute_dtype=compute_dtype,
            param_dtype=jnp.float32,
        )

        def create_agent(rng_key):
            init_x = (
                jnp.zeros((1, 1, *env.observation_space(env_params).shape)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1), dtype=jnp.int32),
            )
            init_hs = network.initialize_carry(1)
            init_omega = jnp.zeros((1, 1, 1 + int(config["NUM_OMEGAS"])))
            network_variables = network.init(
                rng_key, init_hs, *init_x, init_omega, train=False
            )
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

        def compute_n_step_targets_rnn(train_state, transitions, omegas, hs_start):
            num_steps, num_envs, num_omegas = omegas.shape
            reward_dtype = transitions.reward.dtype
            gamma = jnp.asarray(config["GAMMA"], dtype=reward_dtype)
            n_step = int(config.get("N_STEP", 1))
            flat_batch_size = num_steps * num_envs

            time_indices = jnp.arange(num_steps)
            bootstrap_mask = jnp.ones((num_steps, num_envs), dtype=reward_dtype)
            bootstrap_gamma = jnp.ones((num_steps, num_envs), dtype=reward_dtype)
            reward_sum = jnp.zeros((num_steps, num_envs), dtype=reward_dtype)

            for k in range(n_step):
                valid = ((time_indices + k) < num_steps).astype(reward_dtype)[:, None]
                step_indices = jnp.minimum(time_indices + k, num_steps - 1)
                step_reward = transitions.reward[step_indices]
                step_done = transitions.done[step_indices].astype(reward_dtype)

                reward_sum = (
                    reward_sum
                    + valid * bootstrap_mask * bootstrap_gamma * step_reward
                )
                bootstrap_mask = bootstrap_mask * (1.0 - valid * step_done)
                bootstrap_gamma = bootstrap_gamma * jnp.where(valid > 0, gamma, 1.0)

            effective_horizon = jnp.minimum(n_step, num_steps - time_indices)
            bootstrap_indices = time_indices + effective_horizon - 1

            # Build one extra latent step so bootstrap indices can safely use t+h
            # without clipping at sequence end (uses true final next_obs context).
            latent_obs = jnp.concatenate(
                [transitions.obs, transitions.next_obs[-1:]], axis=0
            )
            latent_done = jnp.concatenate(
                [transitions.last_done, transitions.done[-1:]], axis=0
            )
            latent_last_action = jnp.concatenate(
                [transitions.last_action, transitions.action[-1:]], axis=0
            )

            _, state_latent = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                hs_start,
                latent_obs,
                latent_done,
                latent_last_action,
                train=False,
                method=RNNCharacteristicQNetwork.encode_latent,
            )
            state_latent = jax.lax.stop_gradient(state_latent)
            d = state_latent.shape[-1]
            # Bootstrap from latent at t+h. With the extra appended step above,
            # index num_steps is valid for tail transitions.
            latent_idx = bootstrap_indices + 1
            z_boot = state_latent[latent_idx[:, None], jnp.arange(num_envs)[None, :], :]
            z_flat = z_boot.reshape(flat_batch_size, d)

            scaled_omegas = (bootstrap_gamma[..., None] * omegas).reshape(
                flat_batch_size, num_omegas
            )
            bootstrap_omegas = jnp.concatenate(
                (
                    jnp.zeros((flat_batch_size, 1), dtype=omegas.dtype),
                    scaled_omegas,
                ),
                axis=-1,
            )
            bootstrap_outputs = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                z_flat,
                bootstrap_omegas,
                False,
                method=RNNCharacteristicQNetwork.apply_cf_heads,
            )
            next_mean_values = bootstrap_outputs["mean"][:, 0, :].reshape(
                num_steps, num_envs, -1
            )
            next_actions = jnp.argmax(next_mean_values, axis=-1)
            next_mean_bootstrap = select_action_values(next_mean_values, next_actions)

            next_cf = bootstrap_outputs["cf"][:, 1:, :].reshape(
                num_steps, num_envs, num_omegas, -1
            )
            next_cf = select_action_values(next_cf, next_actions)

            reward_phase = jnp.exp(1j * omegas * reward_sum[..., None])
            cf_targets = reward_phase * (
                (1.0 - bootstrap_mask[..., None])
                + bootstrap_mask[..., None] * next_cf
            )

            mean_targets = (
                reward_sum + bootstrap_mask * bootstrap_gamma * next_mean_bootstrap
            )
            return jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                (cf_targets, mean_targets),
            )

        def _update_step(runner_state, unused):
            train_state, memory_transitions, expl_state, test_metrics, rng = runner_state

            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]
                _done = last_done[np.newaxis]
                _last_action = last_action[np.newaxis]

                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                    method=RNNCharacteristicQNetwork.q_values,
                )
                q_vals = q_vals.squeeze(axis=0)

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                transition = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    next_obs=new_obs,
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

            rng, rng_omega_targets = jax.random.split(rng)
            omegas = sample_omegas(
                rng_omega_targets,
                config["NUM_STEPS"] * config["NUM_ENVS"],
                config,
            ).reshape(config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_OMEGAS"])

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):
                    train_state, rng = carry
                    hs = jax.tree_util.tree_map(lambda x: x[0], minibatch.last_hs)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )
                    rng, rng_omega = jax.random.split(rng)
                    om_mb = sample_omegas(
                        rng_omega,
                        config["NUM_STEPS"] * (config["NUM_ENVS"] // config["NUM_MINIBATCHES"]),
                        config,
                    ).reshape(
                        config["NUM_STEPS"],
                        config["NUM_ENVS"] // config["NUM_MINIBATCHES"],
                        config["NUM_OMEGAS"],
                    )

                    cf_targets, mean_targets = compute_n_step_targets_rnn(
                        train_state, minibatch, om_mb, hs
                    )

                    omega_input = jnp.concatenate(
                        (
                            jnp.zeros((*om_mb.shape[:2], 1), dtype=om_mb.dtype),
                            om_mb,
                        ),
                        axis=-1,
                    )

                    def _loss_fn(params):
                        (_hs, outputs), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hs,
                            *agent_in,
                            omega_input,
                        )
                        pred_qvals = outputs["mean"][..., 0, :].astype(jnp.float32)
                        chosen_action_qvals = select_action_values(
                            pred_qvals, minibatch.action
                        )
                        pred_cf = select_action_values(
                            outputs["cf"][..., 1:, :], minibatch.action
                        ).astype(jnp.complex64)
                        pred_mean = chosen_action_qvals
                        diff = pred_cf - cf_targets.astype(jnp.complex64)
                        cf_loss = 0.5 * (
                            jnp.square(jnp.real(diff)) + jnp.square(jnp.imag(diff))
                        ).mean()
                        mean_loss = 0.5 * jnp.square(
                            pred_mean - mean_targets.astype(jnp.float32)
                        ).mean()
                        loss = cf_loss + config.get("AUX_MEAN_LOSS_WEIGHT", 0.1) * mean_loss
                        return loss, (
                            updates,
                            chosen_action_qvals,
                            pred_cf,
                            cf_loss,
                            mean_loss,
                        )

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params
                    )
                    updates, chosen_action_qvals, pred_cf, cf_loss, mean_loss = aux
                    grad_norm = optax.global_norm(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (
                        loss,
                        cf_loss,
                        mean_loss,
                        chosen_action_qvals.mean(),
                        jnp.abs(pred_cf).mean(),
                        grad_norm,
                    )

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
                return (train_state, rng), scan_out

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (
                loss,
                cf_loss,
                mean_loss,
                qvals,
                cf_mag,
                grad_norm,
            ) = jax.lax.scan(_learn_epoch, (train_state, _rng), None, config["NUM_EPOCHS"])

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "cf_loss": cf_loss.mean(),
                "aux_mean_loss": mean_loss.mean(),
                "qvals": qvals.mean(),
                "cf_magnitude": cf_mag.mean(),
                "grad_norm": grad_norm.mean(),
                "omega_min": omegas.min(),
                "omega_mean": omegas.mean(),
                "omega_std": omegas.std(),
                "omega_max": omegas.max(),
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

                def callback(metrics_cb, seed_idx_cb):
                    _iv = max(int(config.get("WANDB_LOG_INTERVAL", 100)), 1)
                    if int(metrics_cb["update_steps"]) % _iv != 0:
                        return
                    logged_metrics = dict(metrics_cb)
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        logged_metrics.update(
                            {
                                f"seed_{int(seed_idx_cb) + 1}/{k}": v
                                for k, v in logged_metrics.items()
                            }
                        )
                    wandb.log(logged_metrics, step=logged_metrics["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            runner_state = (
                train_state,
                memory_transitions,
                tuple(expl_state),
                test_metrics,
                rng,
            )
            return runner_state, None

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(step_state, _):
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]
                _done = last_done[np.newaxis]
                _last_action = last_action[np.newaxis]
                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                    method=RNNCharacteristicQNetwork.q_values,
                )
                q_vals = q_vals.squeeze(axis=0)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    rng_s, env_state, new_action, env_params
                )
                step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
                return step_state, info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)
            init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
            init_hs = network.initialize_carry(config["TEST_NUM_ENVS"])
            step_state = (
                init_hs,
                init_obs,
                init_done,
                init_action,
                env_state,
                _rng,
            )
            step_state, infos = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / jnp.maximum(infos["returned_episode"].sum(), 1.0),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
        init_hs = network.initialize_carry(config["NUM_ENVS"])
        expl_state = (init_hs, obs, init_dones, init_action, env_state)

        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]
            _done = last_done[np.newaxis]
            _last_action = last_action[np.newaxis]
            new_hs, q_vals = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _done,
                _last_action,
                train=False,
                method=RNNCharacteristicQNetwork.q_values,
            )
            q_vals = q_vals.squeeze(axis=0)
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.0)
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            new_obs, new_env_state, reward, new_done, info = env.step(
                rng_s, env_state, new_action, env_params
            )
            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                next_obs=new_obs,
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

        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": None}

    return train


def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "cqn_rnn")
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

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=_tags,
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
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

        for i, rng_i in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "cqn_rnn")
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
        "parameters": {
            "LR": {"values": [0.001, 0.0005, 0.0001, 0.00005]},
            "OMEGA_SCALE": {"values": [0.5, 1.0, 2.0]},
        },
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
