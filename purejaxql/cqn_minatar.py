"""
This script is like pqn_minatar but uses a characteristic-function network
based on the architecture described in cqn.pdf.
"""
import copy
import os
import random
import time
from typing import Any, Union

import chex
import flax.linen as nn
import gymnax
import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
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
        x = jnp.mean(x, axis=(1, 2))
        return x


class FrequencyEmbedding(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, omega: jnp.ndarray) -> jnp.ndarray:
        omega = jnp.asarray(omega, dtype=jnp.float32)
        basis = jnp.arange(1, self.hidden_dim + 1, dtype=omega.dtype)
        cos_features = jnp.cos(jnp.pi * omega[..., None] * basis)
        projected = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            name="proj",
        )(cos_features)
        return nn.relu(projected)


class CharacteristicQNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    mlp_num_layers: int = 1
    mlp_hidden_dim: int = 128
    sigma_min: float = 1e-6

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
                    "Frequency batch dimension must match observation batch "
                    f"dimension. Got omega batch {omega.shape[0]} and "
                    f"observation batch {batch_size}."
                )
        else:
            raise ValueError(
                "Frequencies must be scalar, rank-1, or rank-2. "
                f"Got shape {omega.shape}."
            )
        return omega

    def _encode_state(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, name="input_norm")(x)
        else:
            _ = nn.BatchNorm(use_running_average=not train, name="input_norm_dummy")(x)
        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(
            self.mlp_hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            name="state_proj",
        )(x)
        x = apply_norm(x, self.norm_type, train)
        return x

    def _shared_trunk(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = x.reshape(-1, x.shape[-1])
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
    def __call__(
        self,
        x: jnp.ndarray,
        omega: Union[jnp.ndarray, float, None],
        train: bool,
        compute_cf: bool = True,
    ):
        state_latent = self._encode_state(x, train)
        if compute_cf:
            omega = self._prepare_omegas(omega, state_latent.shape[0], state_latent.dtype)
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
            name="mean_head",
        )(hidden)

        if not compute_cf:
            return {"mean": mean}

        scale = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            name="scale_head",
        )(hidden)
        scale = nn.softplus(scale) + self.sigma_min

        kappa = jnp.square(omega) / (1.0 + jnp.square(omega))
        log_magnitude = -0.5 * kappa[..., None] * scale
        phase = omega[..., None] * mean
        cf = jnp.exp(log_magnitude + 1j * phase)

        return {"mean": mean, "scale": scale, "cf": cf}

    def q_values(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        outputs = self(x, None, train, compute_cf=False)
        return outputs["mean"][:, 0, :]


def sample_omegas(rng, batch_size: int, config) -> jnp.ndarray:
    num_omegas = config["NUM_OMEGAS"]
    del config
    return jax.random.uniform(
        rng,
        (batch_size, num_omegas),
        minval=0.0,
        maxval=1.0,
    )


def select_action_values(values: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    if values.ndim < 2:
        raise ValueError(f"Unsupported tensor rank for action selection: {values.ndim}")
    action_idx = actions
    for _ in range(values.ndim - actions.ndim - 1):
        action_idx = jnp.expand_dims(action_idx, axis=-1)
    action_idx = jnp.expand_dims(action_idx, axis=-1)
    action_idx = jnp.broadcast_to(action_idx, values.shape[:-1] + (1,))
    return jnp.take_along_axis(values, action_idx, axis=-1).squeeze(axis=-1)


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


def make_train(config):
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = int(
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    max_grad_norm = config["MAX_GRAD_NORM"]
    per_layer_gradient_clipping = config.get("PER_LAYER_GRADIENT_CLIPPING", False)

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    config["TEST_NUM_STEPS"] = env_params.max_steps_in_episode

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),
            greedy_actions,
        )
        return chosen_actions

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
            config["EPS_DECAY"] * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=config["NUM_UPDATES_DECAY"]
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        network = CharacteristicQNetwork(
            action_dim=env.action_space(env_params).n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
            mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
            sigma_min=config.get("SIGMA_MIN", 1e-6),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            init_omega = jnp.zeros((1, config["NUM_OMEGAS"]))
            network_variables = network.init(rng, init_x, init_omega, train=False)
            clip_transform = (
                optax.identity()
                if per_layer_gradient_clipping
                else optax.clip_by_global_norm(max_grad_norm)
            )
            tx = optax.chain(
                clip_transform,
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        train_state = create_agent(rng)

        def q_values_from_state(train_state, obs, train: bool):
            return network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                obs,
                train=train,
                method=CharacteristicQNetwork.q_values,
            )

        def compute_n_step_targets(train_state, transitions, omegas):
            num_steps, num_envs, num_omegas = omegas.shape
            obs_shape = transitions.next_obs.shape[2:]
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
            flat_next_obs = transitions.next_obs[bootstrap_indices].reshape(
                flat_batch_size, *obs_shape
            )
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
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                flat_next_obs,
                bootstrap_omegas,
                train=False,
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
            train_state, expl_state, test_metrics, rng = runner_state

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = q_values_from_state(train_state, last_obs, train=False)

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
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
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            rng, rng_omega_targets = jax.random.split(rng)
            omegas = sample_omegas(
                rng_omega_targets,
                config["NUM_STEPS"] * config["NUM_ENVS"],
                config,
            ).reshape(config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_OMEGAS"])
            cf_targets, mean_targets = compute_n_step_targets(
                train_state, transitions, omegas
            )

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):
                    train_state, rng = carry
                    minibatch, target_cf, omega_input, target_mean = minibatch_and_target

                    def _loss_fn(params):
                        outputs, updates = network.apply(
                            {
                                "params": params,
                                "batch_stats": train_state.batch_stats,
                            },
                            minibatch.obs,
                            omega_input,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        pred_qvals = outputs["mean"][:, 0, :]
                        chosen_action_qvals = select_action_values(
                            pred_qvals, minibatch.action
                        )
                        pred_cf = select_action_values(
                            outputs["cf"][:, 1:, :], minibatch.action
                        )
                        pred_scale = select_action_values(
                            outputs["scale"][:, 1:, :], minibatch.action
                        )
                        pred_mean = chosen_action_qvals
                        diff = pred_cf - target_cf
                        cf_loss = 0.5 * (
                            jnp.square(jnp.real(diff)) + jnp.square(jnp.imag(diff))
                        ).mean()
                        mean_loss = 0.5 * jnp.square(pred_mean - target_mean).mean()
                        loss = (
                            cf_loss
                            + config.get("AUX_MEAN_LOSS_WEIGHT", 0.1) * mean_loss
                        )
                        return loss, (
                            updates,
                            chosen_action_qvals,
                            pred_cf,
                            pred_scale,
                            cf_loss,
                            mean_loss,
                        )

                    (
                        loss,
                        (
                            updates,
                            chosen_action_qvals,
                            pred_cf,
                            pred_scale,
                            cf_loss,
                            mean_loss,
                        ),
                    ), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        train_state.params
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
                        rng,
                    ), (
                        loss,
                        cf_loss,
                        mean_loss,
                        chosen_action_qvals.mean(),
                        jnp.abs(pred_cf).mean(),
                        pred_scale.mean(),
                        grad_norm,
                        layer_grad_norms,
                    )

                def preprocess_transition(x, permutation):
                    x = x.reshape(-1, *x.shape[2:])
                    x = x[permutation]
                    x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    return x

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(
                    _rng, config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, permutation), transitions
                )
                cf_target_minibatches = preprocess_transition(cf_targets, permutation)
                mean_target_minibatches = preprocess_transition(mean_targets, permutation)
                omega_input_minibatches = preprocess_transition(
                    jnp.concatenate(
                        (
                            jnp.zeros((*omegas.shape[:2], 1), dtype=omegas.dtype),
                            omegas,
                        ),
                        axis=-1,
                    ),
                    permutation,
                )

                (train_state, rng), (
                    loss,
                    cf_loss,
                    mean_loss,
                    qvals,
                    cf_magnitude,
                    scale_mean,
                    grad_norm,
                    layer_grad_norms,
                ) = jax.lax.scan(
                    _learn_phase,
                    (train_state, rng),
                    (
                        minibatches,
                        cf_target_minibatches,
                        omega_input_minibatches,
                        mean_target_minibatches,
                    ),
                )

                return (train_state, rng), (
                    loss,
                    cf_loss,
                    mean_loss,
                    qvals,
                    cf_magnitude,
                    scale_mean,
                    grad_norm,
                    layer_grad_norms,
                )

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (
                loss,
                cf_loss,
                mean_loss,
                qvals,
                cf_magnitude,
                scale_mean,
                grad_norm,
                layer_grad_norms,
            ) = jax.lax.scan(_learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"])

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps
                * env.observation_space(env_params).shape[-1],
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "cf_loss": cf_loss.mean(),
                "aux_mean_loss": mean_loss.mean(),
                "qvals": qvals.mean(),
                "cf_magnitude": cf_magnitude.mean(),
                "scale_mean": scale_mean.mean(),
                "grad_norm": grad_norm.mean(),
            }
            if config.get("LOG_LAYER_GRAD_NORMS", False):
                layer_grad_norms = jax.tree_util.tree_map(
                    lambda x: x.mean(), layer_grad_norms
                )
                metrics.update(
                    {f"grad_norm/{k}": v for k, v in layer_grad_norms.items()}
                )
            metrics.update({k: v.mean() for k, v in infos.items()})

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

                def callback(metrics, seed_idx):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"seed_{int(seed_idx) + 1}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)
            return runner_state, metrics

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = q_values_from_state(train_state, last_obs, train=False)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(_rng, env_state, action)
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)

            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            done_infos = jax.tree_util.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_metrics, _rng)
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
    network = CharacteristicQNetwork(
        action_dim=env.action_space(env_params).n,
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
        mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
        sigma_min=config.get("SIGMA_MIN", 1e-6),
    )
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    init_omega = jnp.zeros((1, config["NUM_OMEGAS"]))
    summary = network.tabulate(
        jax.random.PRNGKey(config["SEED"]),
        init_x,
        init_omega,
        train=False,
    )
    print(summary)


def single_run(config):
    config = {**config, **config["alg"]}
    print(config)

    alg_name = config.get("ALG_NAME", "cqn")
    env_name = config["ENV_NAME"]

    maybe_print_network_summary(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    seed = random.randint(0, 100_000)
    rng = jax.random.PRNGKey(seed)

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    seed_idxs = jnp.arange(config["NUM_SEEDS"], dtype=jnp.int32)
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs, seed_idxs))
    print(f"Took {time.time()-t0} seconds to complete.")

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

        for i, _ in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    print(default_config)
    alg_name = default_config.get("ALG_NAME", "cqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        maybe_print_network_summary(config)

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        seed_idxs = jnp.arange(config["NUM_SEEDS"], dtype=jnp.int32)
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        jax.block_until_ready(train_vjit(rngs, seed_idxs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
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
