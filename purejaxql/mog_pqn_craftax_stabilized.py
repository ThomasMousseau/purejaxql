"""
MoG-PQN on Craftax (purejaxql layout).

**Algorithm (same as ``old_cvi/mog-pqn.py`` / ``cvi_dqn_nocollapse_jax.py``):**
  - Network outputs a Gaussian mixture per action: (π, μ, σ).
  - Double DQN: greedy next action from online Q = Σ_k π_k μ_k; target MoG on s'
    for that action gives a **closed** Bellman target in CF space at sampled ω.
  - Loss: mean squared |φ_online(a) − φ_target|^2 in the complex plane.

**What changes vs standard ``pqn_craftax.py`` in this repo:**
  - Scalar Q-head is replaced by MoG heads; exploration still uses closed-form Q.
  - A **target network** + Polyak ``TAU`` (MoG-CF bootstrap mirrors the CVI DQN script).
  - Adam ``eps`` defaults to ``0.01 / minibatch_size`` like the CleanRL CVI scripts.

**What changes vs the old CleanRL tyro scripts:**
  - Config comes from Hydra (``config.yaml`` + ``alg/mog_pqn_craftax.yaml``).
  - Env stepping uses this repo’s Craftax wrappers (``LogWrapper``, optimistic reset, etc.).
  - Optional multi-seed ``vmap`` matches other ``*_craftax.py`` trainers here.

Run (from repo root, with ``craftax`` extra installed):

  uv run python purejaxql/mog_pqn_craftax.py --config-path purejaxql/config \\
    --config-name config alg=mog_pqn_craftax
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
from omegaconf import OmegaConf

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.utils.craftax_wrappers import (
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from purejaxql.utils.mog_cf import build_mog_cf, mog_q_values, sample_frequencies

class MoGQNetwork(nn.Module):
    """Feedforward trunk + MoG heads (π, μ, σ) per action × component (cf. ``CF_QNetwork`` in CVI)."""

    action_dim: int
    num_components: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        del train
        if self.norm_type == "layer_norm":
            normalize = lambda z: nn.LayerNorm()(z)
        else:
            normalize = lambda z: z

        if self.norm_input:
            x = nn.LayerNorm()(x)

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

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

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        return jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(rng_a, greedy_actions.shape, 0, q_vals.shape[-1]),
            greedy_actions,
        )

    exploration_fraction = float(config.get("EXPLORATION_FRACTION", 0.5))
    total_ts = float(config["TOTAL_TIMESTEPS"])
    learning_starts = int(config.get("LEARNING_STARTS", 10_000))

    def linear_epsilon(env_steps: jnp.ndarray) -> jnp.ndarray:
        """Match ``cvi_dqn_nocollapse_jax.linear_schedule`` (env-step based)."""
        slope = (config["EPS_FINISH"] - config["EPS_START"]) / (
            exploration_fraction * jnp.maximum(total_ts, 1.0)
        )
        return jnp.maximum(slope * env_steps + config["EPS_START"], config["EPS_FINISH"])

    adam_eps_div = float(
        config["ADAM_EPS_BATCH_REF"] if config.get("ADAM_EPS_BATCH_REF") is not None else minibatch_size
    )

    def train(rng, seed_idx):
        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=int(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        network = MoGQNetwork(
            action_dim=action_dim,
            num_components=int(config["NUM_COMPONENTS"]),
            hidden_size=int(config.get("HIDDEN_SIZE", 512)),
            num_layers=int(config.get("NUM_LAYERS", 4)),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=0.01 / adam_eps_div),
        )

        def create_agent(rng_init):
            init_x = jnp.zeros((1, *obs_shape))
            params = network.init(rng_init, init_x, train=False)["params"]
            st = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
            target_params = jax.tree.map(lambda x: x, st.params)
            return st, target_params

        rng, rng_agent = jax.random.split(rng)
        train_state, target_params = create_agent(rng_agent)

        def mog_gradient_step(train_state, target_params, batch, rng_step):
            """One SGD step — same tensor contract as ``old_cvi/mog-pqn.py:gradient_step``."""
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

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
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
                # epoch_losses: (NUM_EPOCHS, NUM_MINIBATCHES); aux: 5-tuple of same shape
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

            if config.get("TEST_DURING_TRAINING", False):
                rng, rng_t = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    n_updates % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_test_metrics(train_state, rng_t),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {k: v for k, v in metrics.items() if "achievement" not in k.lower()}

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
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                return (new_env_state, new_obs, rng), info

            rng, rng_reset = jax.random.split(rng)
            init_obs, env_state = test_env.reset(rng_reset, env_params)
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
        expl_state = env.reset(rng_reset, env_params)
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


def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "mog_pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
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

        print("running experiment with params:", config)
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
