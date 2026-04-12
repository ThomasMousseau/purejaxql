"""
MoG-PQN on MinAtar (gymnax): CNN trunk + mixture-of-Gaussians CF heads.

**Stabilized (this file):** Double DQN-style next-action selection from **online** Q and
bootstrap from **target** MoG parameters, with **Polyak** soft target updates (``TAU``).

See ``mog_pqn_minatar.py`` for the non-stabilized variant (greedy next action on the
target network + periodic hard target sync, no Polyak).

Run from repo root::

  uv run python -m purejaxql.mog_pqn_minatar_stabilized \\
    --config-path purejaxql/config --config-name config +alg=mog_pqn_minatar_stabilized
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
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
from omegaconf import OmegaConf

import gymnax

from purejaxql.utils.mog_cf import build_mog_cf, mog_q_values, sample_frequencies
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
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, name="input_norm")(x)
        # No dummy BatchNorm: params-only TrainState.apply cannot supply batch_stats for unused BN.
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

        a, m = self.action_dim, self.num_components
        pi_logits = nn.Dense(a * m)(x)
        pi = jax.nn.softmax(pi_logits.reshape(-1, a, m), axis=-1)
        mu = nn.Dense(a * m)(x).reshape(-1, a, m)
        sigma = jax.nn.softplus(nn.Dense(a * m)(x).reshape(-1, a, m))
        return pi, mu, sigma


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array


def polyak_update(online: Any, target: Any, tau: float) -> Any:
    return jax.tree.map(lambda o, t: tau * o + (1.0 - tau) * t, online, target)


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

    exploration_fraction = float(config.get("EXPLORATION_FRACTION", 0.3))
    total_ts = float(config["TOTAL_TIMESTEPS"])
    learning_starts = int(config.get("LEARNING_STARTS", 10_000))

    def linear_epsilon(env_steps: jnp.ndarray) -> jnp.ndarray:
        slope = (config["EPS_FINISH"] - config["EPS_START"]) / (
            exploration_fraction * jnp.maximum(total_ts, 1.0)
        )
        return jnp.maximum(slope * env_steps + config["EPS_START"], config["EPS_FINISH"])

    adam_eps_div = float(
        config["ADAM_EPS_BATCH_REF"] if config.get("ADAM_EPS_BATCH_REF") is not None else minibatch_size
    )

    lr_scheduler = optax.linear_schedule(
        init_value=config["LR"],
        end_value=1e-20,
        transition_steps=int(config["NUM_UPDATES_DECAY"])
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

    def train(rng, seed_idx):
        network = MoGMinAtarQNetwork(
            action_dim=action_dim,
            num_components=int(config["NUM_COMPONENTS"]),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            mlp_num_layers=config.get("MLP_NUM_LAYERS", 1),
            mlp_hidden_dim=config.get("MLP_HIDDEN_DIM", 128),
        )

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=0.01 / adam_eps_div),
        )

        def create_agent(rng_init):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            params = network.init(rng_init, init_x, train=False)["params"]
            st = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
            target_params = jax.tree.map(lambda x: x, st.params)
            return st, target_params

        rng, rng_agent = jax.random.split(rng)
        train_state, target_params = create_agent(rng_agent)

        def mog_gradient_step(train_state, target_params, batch, rng_step):
            rng_step, omega_key = jax.random.split(rng_step)
            s_obs = batch.obs
            s_next_obs = batch.next_obs
            s_actions = batch.action
            s_rewards = batch.reward
            s_dones = batch.done
            bsz = s_obs.shape[0]
            bidx = jnp.arange(bsz)
            omegas = sample_frequencies(
                omega_key, int(config["NUM_OMEGA_SAMPLES"]), float(config["OMEGA_MAX"])
            )

            # Double DQN: greedy action from online Q, bootstrap from target MoG.
            pi_n, mu_n, _ = network.apply({"params": train_state.params}, s_next_obs, train=False)
            online_q_next = mog_q_values(pi_n, mu_n)
            next_actions = jnp.argmax(online_q_next, axis=1)

            target_pi, target_mu, target_sigma = network.apply(
                {"params": target_params}, s_next_obs, train=False
            )

            target_pi_sel = target_pi[bidx, next_actions]
            target_mu_sel = target_mu[bidx, next_actions]
            target_sigma_sel = target_sigma[bidx, next_actions]

            gammas = config["GAMMA"] * (1.0 - s_dones)
            bellman_mu = s_rewards[:, None] + gammas[:, None] * target_mu_sel
            bellman_sigma = gammas[:, None] * target_sigma_sel
            bellman_pi = target_pi_sel

            td_target = build_mog_cf(
                bellman_pi[:, None, :],
                bellman_mu[:, None, :],
                bellman_sigma[:, None, :],
                omegas,
            )
            td_target = td_target[:, :, 0]
            td_target = jax.lax.stop_gradient(td_target)

            def loss_fn(p):
                online_pi, online_mu, online_sigma = network.apply({"params": p}, s_obs, train=True)
                online_phi = build_mog_cf(online_pi, online_mu, online_sigma, omegas)
                online_phi_sel = online_phi[bidx, :, s_actions]
                loss = jnp.mean(jnp.abs(online_phi_sel - td_target) ** 2)
                residual = online_phi_sel - td_target
                cf_im = jnp.mean(jnp.abs(jnp.imag(residual)))
                q_all = mog_q_values(online_pi, online_mu)
                q_mean = q_all[bidx, s_actions].mean()
                q_gap = (q_all.max(axis=1) - q_all.min(axis=1)).mean()
                mog_ent = -jnp.sum(online_pi * jnp.log(online_pi + 1e-8), axis=-1).mean()
                sig_m = online_sigma.mean()
                return loss, (q_mean, q_gap, mog_ent, sig_m, cf_im)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss, aux

        def _update_step(runner_state, unused):
            train_state, target_params, timesteps, n_updates, expl_state, test_metrics, rng = (
                runner_state
            )

            env_steps_before = timesteps
            eps = linear_epsilon(jnp.asarray(env_steps_before, dtype=jnp.float32))
            is_warmup = env_steps_before < learning_starts

            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                pi, mu, sig = network.apply({"params": train_state.params}, last_obs, train=False)
                q_vals = mog_q_values(pi, mu)
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps_vec = jnp.full((config["NUM_ENVS"],), eps)
                greedy = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps_vec)
                rand_a = jax.random.randint(rng_a, (config["NUM_ENVS"],), 0, action_dim)
                new_action = jnp.where(is_warmup, rand_a, greedy)

                new_obs, new_env_state, reward, new_done, info = vmap_step(config["NUM_ENVS"])(
                    rng_s, env_state, new_action
                )
                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1.0) * reward,
                    done=new_done,
                    next_obs=new_obs,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            rng, rng_scan = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env, (*expl_state, rng_scan), None, config["NUM_STEPS"]
            )
            expl_state = tuple(expl_state)

            timesteps = timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]

            def _learn_epoch(carry, _):
                train_state, target_params, rng = carry

                def _learn_phase(carry, minibatch):
                    train_state, target_params, rng = carry
                    rng, rng_gs = jax.random.split(rng)
                    train_state, loss, aux = mog_gradient_step(
                        train_state, target_params, minibatch, rng_gs
                    )
                    return (train_state, target_params, rng), (loss, aux)

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
                (train_state, target_params, rng), (losses, aux_stack) = jax.lax.scan(
                    _learn_phase,
                    (train_state, target_params, rng_epoch),
                    minibatches,
                )
                return (train_state, target_params, rng), (losses, aux_stack)

            def do_train(op):
                train_state, target_params, rng = op
                rng, _ = jax.random.split(rng)
                (train_state, target_params, rng), (epoch_losses, aux) = jax.lax.scan(
                    _learn_epoch, (train_state, target_params, rng), None, config["NUM_EPOCHS"]
                )
                target_params = polyak_update(
                    train_state.params, target_params, float(config["TAU"])
                )
                loss = epoch_losses.mean()
                q_mean, q_gap, mog_ent, sig_m, cf_im = (a.mean() for a in aux)
                return train_state, target_params, rng, loss, q_mean, q_gap, mog_ent, sig_m, cf_im

            def skip_train(op):
                train_state, target_params, rng = op
                z = jnp.float32(0.0)
                return train_state, target_params, rng, z, z, z, z, z, z

            train_state, target_params, rng, mean_loss, mean_q, mean_qg, mean_ent, mean_sig, mean_cf = (
                jax.lax.cond(is_warmup, skip_train, do_train, (train_state, target_params, rng))
            )

            n_updates = n_updates + 1

            metrics = {
                "env_step": timesteps,
                "global_step": timesteps,
                "update_steps": n_updates,
                "mog_loss": mean_loss,
                "q_values": mean_q,
                "q_gap": mean_qg,
                "mog_entropy": mean_ent,
                "sigma_mean": mean_sig,
                "cf_loss_imag": mean_cf,
                "epsilon": eps,
            }
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / jnp.maximum(infos["returned_episode"].sum(), 1.0),
                infos,
            )
            metrics.update(done_infos)
            if "returned_episode_returns" in metrics:
                metrics["charts/episodic_return"] = metrics["returned_episode_returns"]

            if config.get("TEST_DURING_TRAINING", False):
                rng, rng_t = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_test_metrics(train_state, rng_t),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(m, seed_idx_):
                    _iv = max(int(config.get("WANDB_LOG_INTERVAL", 100)), 1)
                    if int(m["update_steps"]) % _iv != 0:
                        return
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        m = {
                            **{f"seed_{int(seed_idx_) + 1}/{k}": v for k, v in m.items()},
                            **m,
                        }
                    wandb.log(m, step=m["update_steps"])

                jax.debug.callback(callback, metrics, seed_idx)

            runner_state = (
                train_state,
                target_params,
                timesteps,
                n_updates,
                tuple(expl_state),
                test_metrics,
                rng,
            )
            return runner_state, metrics

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return {}

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                pi, mu, sig = network.apply({"params": train_state.params}, last_obs, train=False)
                q_vals = mog_q_values(pi, mu)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = vmap_step(config["TEST_NUM_ENVS"])(
                    _rng, env_state, new_action
                )
                return (new_env_state, new_obs, rng), info

            rng, rng_reset = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(rng_reset)
            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, rng), None, config["TEST_NUM_STEPS"]
            )
            return jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / jnp.maximum(infos["returned_episode"].sum(), 1.0),
                infos,
            )

        rng, rng_test = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, rng_test)

        rng, rng_reset = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(rng_reset)
        timesteps = jnp.int32(0)
        n_updates = jnp.int32(0)
        runner_state = (
            train_state,
            target_params,
            timesteps,
            n_updates,
            expl_state,
            test_metrics,
            rng,
        )

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def _wandb_tags(config: dict) -> list[str]:
    alg_name = config.get("ALG_NAME", "mog_pqn_minatar_stabilized")
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

    alg_name = config.get("ALG_NAME", "mog_pqn_minatar_stabilized")
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
    alg_name = default_config.get("ALG_NAME", "mog_pqn_minatar_stabilized")
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
