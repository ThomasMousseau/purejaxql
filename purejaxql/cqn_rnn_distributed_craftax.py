"""
CQN-RNN (Craftax) with pmap-based distributed data parallel training.

#! Distributed-vs-non-distributed:
#! - Standard DDP behavior: full params + optimizer state are replicated per GPU.
#! - Env batch is sharded across GPUs, so activation-heavy tensors are split.
#! - Gradients and batch_stats are synchronized with lax.pmean every update.
"""

from __future__ import annotations

import os
import time
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from omegaconf import OmegaConf

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.cqn_rnn_craftax import (
    CustomTrainState,
    RNNCharacteristicQNetwork,
    Transition,
    sample_omegas,
    select_action_values,
)
from purejaxql.utils.craftax_wrappers import (
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


def make_train(config):
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = int(
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    available = jax.local_device_count()
    requested = int(config.get("DISTRIBUTED_NUM_DEVICES", 0))
    n_devices = requested if requested > 0 else available
    if n_devices > available:
        raise ValueError(f"Requested {n_devices} devices, only {available} available.")
    if config["NUM_ENVS"] % n_devices != 0:
        raise ValueError("NUM_ENVS must be divisible by number of devices.")
    if config["TEST_NUM_ENVS"] % n_devices != 0:
        raise ValueError("TEST_NUM_ENVS must be divisible by number of devices.")

    num_envs_local = config["NUM_ENVS"] // n_devices
    test_num_envs_local = config["TEST_NUM_ENVS"] // n_devices
    if (config["NUM_STEPS"] * num_envs_local) % config["NUM_MINIBATCHES"] != 0:
        raise ValueError(
            "NUM_MINIBATCHES must divide NUM_STEPS*(NUM_ENVS/devices)."
        )

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=num_envs_local,
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], num_envs_local),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=test_num_envs_local,
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], test_num_envs_local),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=num_envs_local)
        test_env = BatchEnvWrapper(log_env, num_envs=test_num_envs_local)

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
            # Fix #4: Force float32 — bfloat16 destroys complex trig precision.
            compute_dtype=jnp.float32,
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
            variables = network.init(rng_key, init_hs, *init_x, init_omega, train=False)
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

        def train_device(init_rng, rng_local, seed_idx_local):
            train_state = create_agent(init_rng)

            def compute_targets(train_state, transitions, omegas, hs_start):
                num_steps, num_envs, num_omegas = omegas.shape
                reward_dtype = transitions.reward.dtype
                gamma = jnp.asarray(config["GAMMA"], dtype=reward_dtype)
                n_step = int(config.get("N_STEP", 1))
                flat_batch = num_steps * num_envs

                t_idx = jnp.arange(num_steps)
                bootstrap_mask = jnp.ones((num_steps, num_envs), dtype=reward_dtype)
                bootstrap_gamma = jnp.ones((num_steps, num_envs), dtype=reward_dtype)
                reward_sum = jnp.zeros((num_steps, num_envs), dtype=reward_dtype)

                for k in range(n_step):
                    valid = ((t_idx + k) < num_steps).astype(reward_dtype)[:, None]
                    s_idx = jnp.minimum(t_idx + k, num_steps - 1)
                    step_reward = transitions.reward[s_idx]
                    step_done = transitions.done[s_idx].astype(reward_dtype)
                    reward_sum = reward_sum + valid * bootstrap_mask * bootstrap_gamma * step_reward
                    bootstrap_mask = bootstrap_mask * (1.0 - valid * step_done)
                    bootstrap_gamma = bootstrap_gamma * jnp.where(valid > 0, gamma, 1.0)

                horizon = jnp.minimum(n_step, num_steps - t_idx)
                bootstrap_idx = t_idx + horizon - 1

                latent_obs = jnp.concatenate([transitions.obs, transitions.next_obs[-1:]], axis=0)
                latent_done = jnp.concatenate([transitions.last_done, transitions.done[-1:]], axis=0)
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
                latent_idx = bootstrap_idx + 1
                z_boot = state_latent[latent_idx[:, None], jnp.arange(num_envs)[None, :], :]
                z_flat = z_boot.reshape(flat_batch, d)

                scaled_omegas = (bootstrap_gamma[..., None] * omegas).reshape(flat_batch, num_omegas)
                bootstrap_omegas = jnp.concatenate(
                    [jnp.zeros((flat_batch, 1), dtype=omegas.dtype), scaled_omegas], axis=-1
                )
                boot = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    z_flat,
                    bootstrap_omegas,
                    False,
                    method=RNNCharacteristicQNetwork.apply_cf_heads,
                )
                next_mean = boot["mean"][:, 0, :].reshape(num_steps, num_envs, -1)
                next_actions = jnp.argmax(next_mean, axis=-1)
                next_mean_boot = select_action_values(next_mean, next_actions)
                next_cf = boot["cf"][:, 1:, :].reshape(num_steps, num_envs, num_omegas, -1)
                next_cf = select_action_values(next_cf, next_actions)

                reward_phase = jnp.exp(1j * omegas * reward_sum[..., None])
                cf_targets = reward_phase * (
                    (1.0 - bootstrap_mask[..., None]) + bootstrap_mask[..., None] * next_cf
                )
                return jax.lax.stop_gradient(cf_targets)

            def update_step(runner_state, _):
                train_state, memory_transitions, expl_state, rng_local = runner_state

                def step_env(carry, _):
                    hs, last_obs, last_done, last_action, env_state, rng_i = carry
                    rng_i, rng_a, rng_s = jax.random.split(rng_i, 3)
                    new_hs, q_vals = network.apply(
                        {"params": train_state.params, "batch_stats": train_state.batch_stats},
                        hs,
                        last_obs[jnp.newaxis],
                        last_done[jnp.newaxis],
                        last_action[jnp.newaxis],
                        train=False,
                        method=RNNCharacteristicQNetwork.q_values,
                    )
                    q_vals = q_vals.squeeze(axis=0)
                    eps = jnp.full(num_envs_local, eps_scheduler(train_state.n_updates))
                    actions = jax.vmap(eps_greedy_exploration)(
                        jax.random.split(rng_a, num_envs_local), q_vals, eps
                    )
                    new_obs, new_env_state, reward, new_done, info = env.step(
                        rng_s, env_state, actions, env_params
                    )
                    transition = Transition(
                        last_hs=hs,
                        obs=last_obs,
                        next_obs=new_obs,
                        action=actions,
                        reward=config.get("REW_SCALE", 1) * reward,
                        done=new_done,
                        last_done=last_done,
                        last_action=last_action,
                        q_vals=q_vals,
                    )
                    return (new_hs, new_obs, new_done, actions, new_env_state, rng_i), (transition, info)

                rng_local, rs = jax.random.split(rng_local)
                (*expl_state, rng_local), (transitions, infos) = jax.lax.scan(
                    step_env, (*expl_state, rs), None, config["NUM_STEPS"]
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

                def learn_epoch(carry, _):
                    train_state, rng_local = carry

                    def learn_phase(carry, minibatch):
                        train_state, rng_local = carry
                        hs = jax.tree_util.tree_map(lambda x: x[0], minibatch.last_hs)
                        rng_local, ro = jax.random.split(rng_local)
                        om_mb = sample_omegas(
                            ro,
                            config["NUM_STEPS"] * (num_envs_local // config["NUM_MINIBATCHES"]),
                            config,
                        ).reshape(
                            config["NUM_STEPS"],
                            num_envs_local // config["NUM_MINIBATCHES"],
                            config["NUM_OMEGAS"],
                        )
                        cf_targets = compute_targets(train_state, minibatch, om_mb, hs)
                        omega_input = jnp.concatenate(
                            [jnp.zeros((*om_mb.shape[:2], 1), dtype=om_mb.dtype), om_mb], axis=-1
                        )

                        # Fix #1: PQN-style recursive TD(λ) for the mean (policy) head.
                        def _compute_td_lambda_targets(last_q, q_vals, reward, done):
                            """Backward scan computing TD(λ) exactly as PQN does."""
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
                                lambda_returns = (1 - done) * lambda_returns + done * reward
                                next_q = jnp.max(q, axis=-1)
                                return (lambda_returns, next_q), lambda_returns

                            lambda_returns = (
                                reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                            )
                            last_q = jnp.max(q_vals[-1], axis=-1)
                            _, targets = jax.lax.scan(
                                _get_target,
                                (lambda_returns, last_q),
                                jax.tree_util.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                                reverse=True,
                            )
                            targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                            return targets

                        def loss_fn(params):
                            (_, outputs), updates = partial(
                                network.apply, train=True, mutable=["batch_stats"]
                            )(
                                {"params": params, "batch_stats": train_state.batch_stats},
                                hs,
                                minibatch.obs,
                                minibatch.last_done,
                                minibatch.last_action,
                                omega_input,
                            )
                            pred_q = outputs["mean"][..., 0, :].astype(jnp.float32)

                            # --- CF loss (n-step targets, all timesteps) ---
                            chosen_q_all = select_action_values(pred_q, minibatch.action)
                            pred_cf = select_action_values(
                                outputs["cf"][..., 1:, :], minibatch.action
                            ).astype(jnp.complex64)
                            diff = pred_cf - cf_targets.astype(jnp.complex64)
                            cf_loss = 0.5 * (jnp.square(jnp.real(diff)) + jnp.square(jnp.imag(diff))).mean()

                            # --- Mean / policy loss (TD-λ targets, steps [:-1]) ---
                            target_q_vals = jax.lax.stop_gradient(pred_q)
                            last_q = target_q_vals[-1].max(axis=-1)
                            mean_targets = _compute_td_lambda_targets(
                                last_q,
                                target_q_vals[:-1],
                                minibatch.reward[:-1],
                                minibatch.done[:-1],
                            ).reshape(-1)
                            chosen_q = chosen_q_all[:-1].reshape(-1)
                            mean_loss = 0.5 * jnp.square(chosen_q - mean_targets).mean()

                            loss = cf_loss + config.get("AUX_MEAN_LOSS_WEIGHT", 1.0) * mean_loss
                            return loss, (updates, chosen_q_all, pred_cf, cf_loss, mean_loss)

                        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
                        #! Core DDP sync: average grads across devices before optimizer step.
                        grads = jax.lax.pmean(grads, axis_name="devices")
                        updates, chosen_q, pred_cf, cf_loss, mean_loss = aux
                        train_state = train_state.apply_gradients(grads=grads)
                        #! Keep batch_stats replicas aligned across devices.
                        synced_bs = jax.lax.pmean(updates["batch_stats"], axis_name="devices")
                        train_state = train_state.replace(
                            grad_steps=train_state.grad_steps + 1,
                            batch_stats=synced_bs,
                        )
                        grad_norm = optax.global_norm(grads)
                        stats = (
                            loss,
                            cf_loss,
                            mean_loss,
                            chosen_q.mean(),
                            jnp.abs(pred_cf).mean(),
                            grad_norm,
                        )
                        stats = jax.tree_util.tree_map(
                            lambda x: jax.lax.pmean(x, axis_name="devices"), stats
                        )
                        return (train_state, rng_local), stats

                    def preprocess(x, rng_perm):
                        x = jax.random.permutation(rng_perm, x, axis=1)
                        x = x.reshape(x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:])
                        return jnp.swapaxes(x, 0, 1)

                    rng_local, rp = jax.random.split(rng_local)
                    minibatches = jax.tree_util.tree_map(lambda z: preprocess(z, rp), memory_transitions)
                    rng_local, rs = jax.random.split(rng_local)
                    (train_state, rng_local), scan_out = jax.lax.scan(
                        learn_phase, (train_state, rs), minibatches
                    )
                    return (train_state, rng_local), scan_out

                rng_local, re = jax.random.split(rng_local)
                (train_state, rng_local), (loss, cf_loss, mean_loss, qvals, cf_mag, grad_norm) = jax.lax.scan(
                    learn_epoch, (train_state, re), None, config["NUM_EPOCHS"]
                )
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
                }
                done_num = jax.tree_util.tree_map(lambda x: (x * infos["returned_episode"]).sum(), infos)
                done_den = infos["returned_episode"].sum()
                done_num = jax.tree_util.tree_map(lambda x: jax.lax.psum(x, "devices"), done_num)
                done_den = jax.lax.psum(done_den, "devices")
                done_infos = jax.tree_util.tree_map(lambda x: x / jnp.maximum(done_den, 1.0), done_num)
                metrics.update(done_infos)

                if config["WANDB_MODE"] != "disabled":
                    leader = jax.lax.axis_index("devices") == 0
                    _sps_state = {"t": time.time(), "s": 0}

                    def _log(_):
                        def callback(metrics_cb, seed_idx_cb):
                            _iv = max(int(config.get("WANDB_LOG_INTERVAL", 100)), 1)
                            if int(metrics_cb["update_steps"]) % _iv != 0:
                                return
                            now = time.time()
                            steps = int(metrics_cb["env_step"])
                            dt = now - _sps_state["t"]
                            sps = (steps - _sps_state["s"]) / max(dt, 1e-9)
                            _sps_state["t"], _sps_state["s"] = now, steps
                            logged = dict(metrics_cb)
                            logged["sps"] = sps
                            if config.get("WANDB_LOG_ALL_SEEDS", False):
                                logged.update(
                                    {f"seed_{int(seed_idx_cb) + 1}/{k}": v for k, v in logged.items()}
                                )
                            wandb.log(logged, step=logged["update_steps"])

                        jax.debug.callback(callback, metrics, seed_idx_local)
                        return None

                    jax.lax.cond(leader, _log, lambda _: None, operand=None)

                return (train_state, memory_transitions, expl_state, rng_local), None

            rng_local, rr = jax.random.split(rng_local)
            obs, env_state = env.reset(rr, env_params)
            init_done = jnp.zeros((num_envs_local,), dtype=bool)
            init_action = jnp.zeros((num_envs_local,), dtype=int)
            init_hs = network.initialize_carry(num_envs_local)
            expl_state = (init_hs, obs, init_done, init_action, env_state)

            def random_step(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng_i = carry
                rng_i, rng_a, rng_s = jax.random.split(rng_i, 3)
                new_hs, q_vals = network.apply(
                    {"params": train_state.params, "batch_stats": train_state.batch_stats},
                    hs,
                    last_obs[jnp.newaxis],
                    last_done[jnp.newaxis],
                    last_action[jnp.newaxis],
                    train=False,
                    method=RNNCharacteristicQNetwork.q_values,
                )
                q_vals = q_vals.squeeze(axis=0)
                actions = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, num_envs_local),
                    q_vals,
                    jnp.full((num_envs_local,), 1.0),
                )
                new_obs, new_env_state, reward, new_done, _ = env.step(rng_s, env_state, actions, env_params)
                tr = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    next_obs=new_obs,
                    action=actions,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_vals=q_vals,
                )
                return (new_hs, new_obs, new_done, actions, new_env_state, rng_i), tr

            rng_local, rw = jax.random.split(rng_local)
            (*expl_state, rng_local), memory_transitions = jax.lax.scan(
                random_step, (*expl_state, rw), None, config["MEMORY_WINDOW"] + config["NUM_STEPS"]
            )
            runner_state = (train_state, memory_transitions, tuple(expl_state), rng_local)
            runner_state, _ = jax.lax.scan(update_step, runner_state, None, config["NUM_UPDATES"])
            return {"runner_state": runner_state, "metrics": None}

        rng, rng_init = jax.random.split(rng)
        init_rngs = jnp.repeat(rng_init[None, :], n_devices, axis=0)
        device_rngs = jax.random.split(rng, n_devices)
        seed_idxs = jnp.full((n_devices,), seed_idx, dtype=jnp.int32)
        #! pmap axis "devices" executes one replica per GPU.
        train_pmapped = jax.pmap(
            train_device, axis_name="devices", devices=jax.local_devices()[:n_devices]
        )
        return train_pmapped(init_rngs, device_rngs, seed_idxs)

    return train


def single_run(config):
    config = {**config, **config["alg"]}
    if int(config.get("NUM_SEEDS", 1)) != 1:
        raise ValueError(
            "cqn_rnn_distributed_craftax currently expects NUM_SEEDS=1; use SLURM arrays for multi-seed."
        )

    alg_name = config.get("ALG_NAME", "cqn_rnn_distributed")
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
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    t0 = time.time()
    train_fn = make_train(config)
    outs = jax.block_until_ready(train_fn(rng, jnp.asarray(0, dtype=jnp.int32)))
    elapsed = time.time() - t0
    print(f"Took {elapsed} seconds to complete.")
    print(f"SPS: {float(config['TOTAL_TIMESTEPS']) / max(elapsed, 1e-6)}")

    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params

        #! Outputs are replicated over device axis; save leader replica only.
        runner_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"])
        model_state = runner_state[0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'),
        )
        save_params(
            model_state.params,
            os.path.join(save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_ddp.safetensors'),
        )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        raise NotImplementedError(
            "HYP_TUNE is not wired for distributed trainer yet; launch sweeps via SLURM arrays."
        )
    single_run(config)


if __name__ == "__main__":
    main()
