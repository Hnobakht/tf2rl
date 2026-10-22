"""PPO training with selectable vector backends and environment-derived dimensions."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.envs.vector_env import make_vector_env
from tf2rl.experiments.vector_on_policy_trainer import VectorOnPolicyTrainer


def evaluate(policy, env_id, episodes, seed, env_kwargs):
    """Evaluate deterministic actions in independent native Gymnasium episodes."""
    if episodes == 0:
        return None
    returns = []
    env = gym.make(env_id, **env_kwargs)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            while True:
                with tf.device(policy.device):
                    action, _, _ = policy.get_action_and_val_tensor(
                        tf.convert_to_tensor(obs[None], tf.float32), test=True
                    )
                action = action.numpy()[0]
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    return {"returns": returns, "mean_return": float(np.mean(returns))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--units", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float, default=0.075)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument(
        "--env-gpu",
        type=int,
        default=None,
        help="MJWarp GPU index; defaults to --gpu (or 0 for a CPU policy)",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a checkpoint every N measured iterations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/ppo"))
    args = parser.parse_args(argv)
    if min(args.num_envs, args.rollout_steps, args.iterations, args.batch_size, args.epochs) < 1:
        parser.error("Counts must be positive")
    if args.eval_episodes < 0 or args.warmup_iterations < 0:
        parser.error("Evaluation and warmup counts must be nonnegative")
    if args.cpu_threads < 1 or args.checkpoint_interval < 1:
        parser.error("cpu-threads and checkpoint-interval must be positive")
    if len(args.units) != 2 or min(args.units) < 1:
        parser.error("--units requires two positive hidden widths")
    count = args.num_envs * args.rollout_steps
    if count % args.batch_size:
        parser.error("num-envs * rollout-steps must be divisible by batch-size")
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpu >= len(gpus):
        parser.error("Select a visible --gpu for GPU policy training, or -1 for CPU")
    if args.env_gpu is None:
        args.env_gpu = max(args.gpu, 0)
    if args.env_backend == "mjwarp" and not 0 <= args.env_gpu < len(gpus):
        parser.error("Select a visible --env-gpu for MJWarp simulation")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(args.seed)
    from tf2rl.algos.ppo import PPO

    env_kwargs = dict(args.env_kwargs)
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    backend_kwargs = (
        {"nconmax": args.nconmax, "njmax": args.njmax} if args.env_backend == "mjwarp" else {}
    )
    device = f"/GPU:{args.gpu}" if args.gpu >= 0 else "/CPU:0"
    env = make_vector_env(
        args.env_name,
        args.num_envs,
        backend=args.env_backend,
        tensors=True,
        device=f"cuda:{args.env_gpu}"
        if args.env_backend == "mjwarp"
        else (f"cuda:{args.gpu}" if args.gpu >= 0 else device),
        **backend_kwargs,
        **env_kwargs,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    writer = tf.summary.create_file_writer(str(args.output / "events"))
    try:
        space = env.single_action_space
        if (
            not isinstance(space, gym.spaces.Box)
            or len(space.shape) != 1
            or not np.allclose(space.high, -space.low)
            or not np.allclose(space.high, space.high[0])
        ):
            parser.error(
                "The existing Gaussian actor requires uniform symmetric vector action bounds"
            )
        if (
            not isinstance(env.single_observation_space, gym.spaces.Box)
            or len(env.single_observation_space.shape) != 1
        ):
            parser.error("This PPO entry point requires vector Box observations")
        with tf.device(device):
            policy = PPO(
                state_shape=env.single_observation_space.shape,
                action_dim=space.shape[0],
                is_discrete=False,
                max_action=float(space.high[0]),
                horizon=count,
                batch_size=args.batch_size,
                n_epoch=args.epochs,
                actor_units=args.units,
                critic_units=args.units,
                hidden_activation_actor="tanh",
                hidden_activation_critic="tanh",
                lr_actor=args.learning_rate,
                lr_critic=args.learning_rate,
                entropy_coef=0.0,
                target_kl=args.target_kl,
                gpu=args.gpu,
            )
            trainer = VectorOnPolicyTrainer(policy, env, args.rollout_steps, args.seed)
            policy.get_action_and_val_tensor(trainer.obs)
            policy.actor_optimizer.build(policy.actor.trainable_variables)
            policy.critic_optimizer.build(policy.critic.trainable_variables)
            checkpoint = tf.train.Checkpoint(policy=policy)
            manager = tf.train.CheckpointManager(
                checkpoint, str(args.output / "checkpoints"), max_to_keep=5
            )
            initial = evaluate(
                policy, args.env_name, args.eval_episodes, args.seed + 10000, env_kwargs
            )
            print(json.dumps({"initial_evaluation": initial}), flush=True)
            records = []
            for iteration in range(-args.warmup_iterations, args.iterations):
                start = time.perf_counter()
                rollout, stats = trainer.collect()
                collected = time.perf_counter()
                losses, updates = trainer.update(*rollout)
                losses = losses.numpy().tolist()
                updates = int(updates.numpy())
                end = time.perf_counter()
                if hasattr(env, "check_overflow"):
                    env.check_overflow()
                record = {
                    "iteration": iteration + 1,
                    "rollout_seconds": collected - start,
                    "update_seconds": end - collected,
                    "seconds": end - start,
                    "steps_per_second": count / (end - start),
                    "minibatch_updates": updates,
                    "actor_loss": losses[0],
                    "critic_loss": losses[1],
                    "kl": losses[2],
                    "mean_step_reward": stats["mean_step_reward"],
                    "episodes": stats["episodes"],
                    "episode_return": stats["return_sum"] / stats["episodes"]
                    if stats["episodes"]
                    else None,
                }
                if iteration >= 0:
                    records.append(record)
                    with writer.as_default(step=(iteration + 1) * count):
                        for key, value in record.items():
                            if value is not None:
                                tf.summary.scalar(f"PPO/{key}", value)
                    writer.flush()
                    if (iteration + 1) % args.checkpoint_interval == 0:
                        manager.save(checkpoint_number=iteration + 1)
                print(json.dumps({"warmup": iteration < 0, **record}), flush=True)
            final = evaluate(
                policy, args.env_name, args.eval_episodes, args.seed + 10000, env_kwargs
            )
            if args.iterations % args.checkpoint_interval:
                manager.save(checkpoint_number=args.iterations)
            checkpoint.write(str(args.output / "checkpoint"))
            versions = {}
            for package in (
                "tensorflow",
                "tensorflow-probability",
                "tf-keras",
                "gymnasium",
                "numpy",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
            ):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    pass
            summary = {
                "config": config,
                "versions": versions,
                "initial_evaluation": initial,
                "final_evaluation": final,
                "training_steps_per_second": count
                * args.iterations
                / sum(r["seconds"] for r in records),
                "rollout_steps_per_second": count
                * args.iterations
                / sum(r["rollout_seconds"] for r in records),
                "measured_transitions": count * args.iterations,
                "simulation_device": f"cuda:{args.env_gpu}"
                if args.env_backend == "mjwarp"
                else "cpu",
                "actor_devices": sorted({v.value.device for v in policy.actor.variables}),
                "critic_devices": sorted({v.value.device for v in policy.critic.variables}),
                "iterations": records,
            }
            (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                json.dumps({key: value for key, value in summary.items() if key != "iterations"}),
                flush=True,
            )
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
