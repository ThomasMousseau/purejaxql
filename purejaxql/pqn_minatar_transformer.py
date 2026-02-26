"""
This script is like pqn_gymnax but the network uses a CNN + transformer block.
"""
import os
import copy 
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any, Optional
import random

import chex
import optax
import flax.linen as nn
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import hydra
from omegaconf import OmegaConf
import gymnax
import wandb


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


def add_sinusoidal_positional_encoding(x: jnp.ndarray) -> jnp.ndarray:
    """Adds deterministic position encodings for sequence-aware attention."""
    seq_len = x.shape[1]
    hidden_dim = x.shape[-1]
    half_dim = hidden_dim // 2
    if half_dim == 0:
        return x

    position = jnp.arange(seq_len, dtype=x.dtype)[:, None]
    freq_scale = jnp.maximum(half_dim - 1, 1)
    div_term = jnp.exp(
        -jnp.log(jnp.array(10000.0, dtype=x.dtype))
        * jnp.arange(half_dim, dtype=x.dtype)
        / freq_scale
    )
    angles = position * div_term[None, :]
    pos_emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if hidden_dim % 2 != 0:
        pos_emb = jnp.pad(pos_emb, ((0, 0), (0, 1)))
    return x + pos_emb[None, :, :]


class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        for i, channels in enumerate((32, 64)):
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


class ManualCausalSelfAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    causal: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_padding: jnp.ndarray,
        train: bool,
    ) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        qkv = nn.Dense(
            3 * self.hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            name="c_attn",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t: jnp.ndarray) -> jnp.ndarray:
            return t.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(
                0, 2, 1, 3
            )

        q = split_heads(q)  # (B, H, T, D)
        k = split_heads(k)  # (B, H, T, D)
        v = split_heads(v)  # (B, H, T, D)

        scale = jnp.asarray(head_dim, dtype=x.dtype) ** -0.5
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, H, T, T)

        mask = attention_padding[:, None, None, :]  # key padding mask
        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            mask = mask & causal_mask[None, None, :, :]

        neg_large = jnp.asarray(-1e9, dtype=attn_logits.dtype)
        attn_logits = jnp.where(mask, attn_logits, neg_large)

        attn = nn.softmax(attn_logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout_rate, name="attn_dropout")(
            attn, deterministic=not train
        )
        y = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, H, T, D)
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)

        y = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            name="c_proj",
        )(y)
        y = nn.Dropout(rate=self.dropout_rate, name="resid_dropout")(
            y, deterministic=not train
        )
        y = y * attention_padding[:, :, None].astype(y.dtype)
        return y


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    mlp_num_layers: int = 1
    mlp_hidden_dim: int = 128
    num_attention_heads: int = 4
    attention_dropout: float = 0.0
    causal_attention: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool,
        attention_padding: Optional[jnp.ndarray] = None,
    ):
        if self.mlp_hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                "MLP_HIDDEN_DIM must be divisible by NUM_ATTENTION_HEADS. "
                f"Got {self.mlp_hidden_dim} and {self.num_attention_heads}."
            )

        is_sequence_input = x.ndim == 5  # [batch, T, H, W, C]
        if is_sequence_input:
            batch_size, seq_len = x.shape[:2]
            x = x.reshape(batch_size * seq_len, *x.shape[2:])

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, name="input_norm")(x)
        else:
            # dummy normalize input for global compatibility
            _ = nn.BatchNorm(use_running_average=not train, name="input_norm_dummy")(x)
            # x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)

        if is_sequence_input:
            x = x.reshape(batch_size, seq_len, -1)
        else:
            x = x[:, None, :]

        if attention_padding is None:
            attention_padding = jnp.ones(x.shape[:2], dtype=jnp.bool_)
        elif not is_sequence_input:
            attention_padding = attention_padding[:, None]

        if x.shape[1] > 1:
            x = add_sinusoidal_positional_encoding(x)

        # Project to hidden_dim
        x = nn.Dense(
            self.mlp_hidden_dim,
            kernel_init=nn.initializers.he_normal(),
            name=f"mlp_proj",
        )(x)
        x = apply_norm(x, self.norm_type, train)

        for i in range(self.mlp_num_layers):
            resid = x
            x = nn.Dense(
                2*self.mlp_hidden_dim,
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

            attn_resid = x
            x = ManualCausalSelfAttention(
                hidden_dim=self.mlp_hidden_dim,
                num_heads=self.num_attention_heads,
                dropout_rate=self.attention_dropout,
                causal=self.causal_attention,
                name=f"self_attention_{i}",
            )(x, attention_padding=attention_padding, train=train)
            x = apply_norm(x, self.norm_type, train)
            x = x + attn_resid

        x = nn.Dense(self.action_dim, kernel_init=nn.zeros)(x)
        if not is_sequence_input:
            x = x.squeeze(axis=1)
        return x


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

    sequence_length = int(config.get("SEQUENCE_LENGTH", 1))
    assert 1 <= sequence_length <= config["NUM_STEPS"], (
        "SEQUENCE_LENGTH must be in [1, NUM_STEPS]"
    )
    action_history_length = int(
        config.get("ACTION_HISTORY_LENGTH", sequence_length)
    )
    assert 1 <= action_history_length <= config["NUM_STEPS"], (
        "ACTION_HISTORY_LENGTH must be in [1, NUM_STEPS]"
    )
    num_sequence_chunks = (
        config["NUM_STEPS"] + sequence_length - 1
    ) // sequence_length
    time_padding = num_sequence_chunks * sequence_length - config["NUM_STEPS"]
    total_sequences = config["NUM_ENVS"] * num_sequence_chunks
    sequence_batch_padding = (
        config["NUM_MINIBATCHES"] - (total_sequences % config["NUM_MINIBATCHES"])
    ) % config["NUM_MINIBATCHES"]
    total_sequences_padded = total_sequences + sequence_batch_padding

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    max_grad_norm = config["MAX_GRAD_NORM"]
    per_layer_gradient_clipping = config.get("PER_LAYER_GRADIENT_CLIPPING", False)

    assert config["NUM_MINIBATCHES"] > 0, "NUM_MINIBATCHES must be positive"

    def update_obs_history(
        history_obs: jnp.ndarray,
        history_valid: jnp.ndarray,
        obs: jnp.ndarray,
        reset_mask: jnp.ndarray,
    ):
        obs_with_time = obs[:, None, ...]
        shifted_obs = jnp.concatenate([history_obs[:, 1:], obs_with_time], axis=1)
        shifted_valid = jnp.concatenate(
            [
                history_valid[:, 1:],
                jnp.ones((history_valid.shape[0], 1), dtype=jnp.bool_),
            ],
            axis=1,
        )

        reset_obs = jnp.zeros_like(history_obs)
        reset_obs = reset_obs.at[:, -1].set(obs)
        reset_valid = jnp.zeros_like(history_valid)
        reset_valid = reset_valid.at[:, -1].set(True)

        obs_mask = reset_mask.reshape(
            (reset_mask.shape[0],) + (1,) * (history_obs.ndim - 1)
        )
        updated_obs = jnp.where(obs_mask, reset_obs, shifted_obs)
        updated_valid = jnp.where(reset_mask[:, None], reset_valid, shifted_valid)
        return updated_obs, updated_valid

    def preprocess_transition(x, rng, pad_value=0):
        # x: (num_steps, num_envs, ...)
        x = jnp.swapaxes(x, 0, 1)  # (num_envs, num_steps, ...)
        if time_padding > 0:
            x = jnp.pad(
                x,
                ((0, 0), (0, time_padding)) + ((0, 0),) * (x.ndim - 2),
                constant_values=pad_value,
            )
        x = x.reshape(
            config["NUM_ENVS"],
            num_sequence_chunks,
            sequence_length,
            *x.shape[2:],
        )  # (num_envs, num_chunks, T, ...)
        x = x.reshape(-1, sequence_length, *x.shape[3:])  # (batch, T, ...)
        if sequence_batch_padding > 0:
            x = jnp.pad(
                x,
                ((0, sequence_batch_padding), (0, 0)) + ((0, 0),) * (x.ndim - 2),
                constant_values=pad_value,
            )
        x = jax.random.permutation(rng, x, axis=0)  # shuffle full sequences
        x = x.reshape(
            config["NUM_MINIBATCHES"],
            total_sequences_padded // config["NUM_MINIBATCHES"],
            sequence_length,
            *x.shape[2:],
        )  # (num_minibatches, batch_per_minibatch, T, ...)
        return x

    def preprocess_valid_mask(rng):
        valid = jnp.ones((config["NUM_ENVS"], config["NUM_STEPS"]), dtype=jnp.bool_)
        if time_padding > 0:
            valid = jnp.pad(
                valid,
                ((0, 0), (0, time_padding)),
                constant_values=False,
            )
        valid = valid.reshape(config["NUM_ENVS"], num_sequence_chunks, sequence_length)
        valid = valid.reshape(-1, sequence_length)
        if sequence_batch_padding > 0:
            valid = jnp.pad(
                valid,
                ((0, sequence_batch_padding), (0, 0)),
                constant_values=False,
            )
        valid = jax.random.permutation(rng, valid, axis=0)
        valid = valid.reshape(
            config["NUM_MINIBATCHES"],
            total_sequences_padded // config["NUM_MINIBATCHES"],
            sequence_length,
        )
        return valid

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = LogWrapper(env)
    config["TEST_NUM_STEPS"] = env_params.max_steps_in_episode

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

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

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
            mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
            num_attention_heads=config.get("NUM_ATTENTION_HEADS", 4),
            attention_dropout=config.get("ATTENTION_DROPOUT", 0.0),
            causal_attention=config.get("CAUSAL_ATTENTION", True),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_variables = network.init(rng, init_x, train=False)
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

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, history_obs, history_valid, last_done, rng = carry
                history_obs, history_valid = update_obs_history(
                    history_obs,
                    history_valid,
                    last_obs,
                    last_done,
                )
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_seq = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    history_obs,
                    train=False,
                    attention_padding=history_valid,
                )
                q_vals = q_seq[:, -1]

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1)*reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (
                    new_obs,
                    new_env_state,
                    history_obs,
                    history_valid,
                    new_done,
                    rng,
                ), (transition, info)

            # step the env
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
            )  # update timesteps count

            # jax.debug.breakpoint()

            last_history_obs, last_history_valid = update_obs_history(
                expl_state[2],
                expl_state[3],
                transitions.next_obs[-1],
                transitions.done[-1],
            )
            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                last_history_obs,
                train=False,
                attention_padding=last_history_valid,
            )
            last_q = jnp.max(last_q[:, -1], axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target, valid_mask = minibatch_and_target

                    def _loss_fn(params):
                        safe_attention_padding = valid_mask.at[:, -1].set(True)
                        safe_attention_padding = jnp.where(
                            jnp.any(valid_mask, axis=-1, keepdims=True),
                            valid_mask,
                            safe_attention_padding,
                        )
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            minibatch.obs,
                            train=True,
                            attention_padding=safe_attention_padding,
                            mutable=["batch_stats"],
                        )  # (batch_size, sequence_length, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        td_loss = 0.5 * jnp.square(chosen_action_qvals - target)
                        valid = valid_mask.astype(td_loss.dtype)
                        loss = jnp.sum(td_loss * valid) / jnp.maximum(
                            jnp.sum(valid), 1.0
                        )

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
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
                    return (
                        train_state,
                        rng,
                    ), (
                        loss,
                        qvals,
                        grad_norm,
                        layer_grad_norms,
                    )

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = preprocess_transition(lambda_targets, _rng)
                valid_masks = preprocess_valid_mask(_rng)

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals, grad_norm, layer_grad_norms) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets, valid_masks)
                )

                return (train_state, rng), (loss, qvals, grad_norm, layer_grad_norms)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals, grad_norm, layer_grad_norms) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "env_frame": train_state.timesteps * env.observation_space(env_params).shape[-1],
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
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

            # report on wandb if required
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
                env_state, last_obs, history_obs, history_valid, last_done, rng = carry
                history_obs, history_valid = update_obs_history(
                    history_obs,
                    history_valid,
                    last_obs,
                    last_done,
                )
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_seq = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    history_obs,
                    train=False,
                    attention_padding=history_valid,
                )
                q_vals = q_seq[:, -1]
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(rng_s, env_state, action)
                return (
                    new_env_state,
                    new_obs,
                    history_obs,
                    history_valid,
                    done,
                    rng,
                ), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)
            test_history_obs = jnp.zeros(
                (
                    config["TEST_NUM_ENVS"],
                    action_history_length,
                    *env.observation_space(env_params).shape,
                ),
                dtype=init_obs.dtype,
            )
            test_history_valid = jnp.zeros(
                (config["TEST_NUM_ENVS"], action_history_length), dtype=jnp.bool_
            )
            test_last_done = jnp.ones((config["TEST_NUM_ENVS"],), dtype=jnp.bool_)

            _, infos = jax.lax.scan(
                _env_step,
                (
                    env_state,
                    init_obs,
                    test_history_obs,
                    test_history_valid,
                    test_last_done,
                    _rng,
                ),
                None,
                config["TEST_NUM_STEPS"],
            )
            # return mean of done infos
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
        init_obs, init_env_state = vmap_reset(config["NUM_ENVS"])(_rng)
        history_obs = jnp.zeros(
            (
                config["NUM_ENVS"],
                action_history_length,
                *env.observation_space(env_params).shape,
            ),
            dtype=init_obs.dtype,
        )
        history_valid = jnp.zeros(
            (config["NUM_ENVS"], action_history_length), dtype=jnp.bool_
        )
        last_done = jnp.ones((config["NUM_ENVS"],), dtype=jnp.bool_)
        expl_state = (init_obs, init_env_state, history_obs, history_valid, last_done)

        # train
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
    network = QNetwork(
        action_dim=env.action_space(env_params).n,
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
        mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
        num_attention_heads=config.get("NUM_ATTENTION_HEADS", 4),
        attention_dropout=config.get("ATTENTION_DROPOUT", 0.0),
        causal_attention=config.get("CAUSAL_ATTENTION", True),
    )
    if config.get("SEQUENCE_LENGTH", 1) > 1:
        init_x = jnp.zeros(
            (1, config["SEQUENCE_LENGTH"], *env.observation_space(env_params).shape)
        )
    else:
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    summary = network.tabulate(
        jax.random.PRNGKey(config["SEED"]),
        init_x,
        train=False,
    )
    print(summary)


def single_run(config):

    config = {**config, **config["alg"]}
    print(config)

    alg_name = config.get("ALG_NAME", "pqn")
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

        for i, rng in enumerate(rngs):
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
    alg_name = default_config.get("ALG_NAME", "pqn")
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
        outs = jax.block_until_ready(train_vjit(rngs, seed_idxs))

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
