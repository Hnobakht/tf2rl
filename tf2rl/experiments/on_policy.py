"""On-policy training with a shared Gymnasium/MJWarp collector and algorithm registry."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.envs.vector_env import make_vector_env
from tf2rl.experiments.vector_on_policy_trainer import VectorOnPolicyTrainer

_ALGORITHMS = {}


def register_on_policy_algorithm(name, factory, *, default_epochs=1):
    """Register an agent factory without changing the collector or environment backend.

    Factories receive the standard VPG-style model/training keywords and optional
    policy-specific keywords. They must return an OnPolicyAgent implementing its
    tensor interface. Registration before main() also makes the name a CLI choice.
    """
    if not name or name in _ALGORITHMS or not callable(factory) or default_epochs < 1:
        raise ValueError("Expected a unique algorithm name, callable factory and positive epochs")
    _ALGORITHMS[name] = {"factory": factory, "default_epochs": default_epochs}


def _ppo(**kwargs):
    from tf2rl.algos.ppo import PPO

    return PPO(**kwargs)


def _vpg(**kwargs):
    from tf2rl.algos.vpg import VPG

    return VPG(**kwargs)


register_on_policy_algorithm("ppo", _ppo, default_epochs=10)
register_on_policy_algorithm("vpg", _vpg, default_epochs=1)


def make_policy(
    algorithm,
    observation_space,
    action_space,
    *,
    horizon=2048,
    batch_size=256,
    epochs=None,
    units=(64, 64),
    learning_rate=3e-4,
    gpu=-1,
    policy_kwargs=None,
):
    """Create a registered on-policy agent using single-world environment spaces."""
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown on-policy algorithm {algorithm!r}")
    if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
        raise ValueError("The on-policy launcher requires vector Box observations")
    discrete = isinstance(action_space, gym.spaces.Discrete)
    if not discrete and (
        not isinstance(action_space, gym.spaces.Box)
        or len(action_space.shape) != 1
        or not np.allclose(action_space.high, -action_space.low)
        or not np.allclose(action_space.high, action_space.high[0])
    ):
        raise ValueError("Actors require Discrete or uniform symmetric vector Box actions")
    spec = _ALGORITHMS[algorithm]
    kwargs = {
        "state_shape": observation_space.shape,
        "action_dim": int(action_space.n) if discrete else action_space.shape[0],
        "is_discrete": discrete,
        "max_action": 1.0 if discrete else float(action_space.high[0]),
        "horizon": horizon,
        "batch_size": batch_size,
        "n_epoch": spec["default_epochs"] if epochs is None else epochs,
        "actor_units": units,
        "critic_units": units,
        "hidden_activation_actor": "tanh",
        "hidden_activation_critic": "tanh",
        "lr_actor": learning_rate,
        "lr_critic": learning_rate,
        "entropy_coef": 0.0,
        "gpu": gpu,
    }
    kwargs.update(policy_kwargs or {})
    with tf.device(f"/GPU:{gpu}" if gpu >= 0 else "/CPU:0"):
        policy = spec["factory"](**kwargs)
    if not isinstance(policy, OnPolicyAgent):
        raise TypeError("Registered factories must return an OnPolicyAgent")
    return policy


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


def main(argv=None, algorithm=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(_ALGORITHMS), default=algorithm or "ppo")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--units", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float, default=None, help="PPO KL stopping threshold")
    parser.add_argument(
        "--policy-kwargs",
        type=json.loads,
        default={},
        help="JSON keywords for the selected algorithm",
    )
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = _ALGORITHMS[args.algorithm]["default_epochs"]
    if args.output is None:
        args.output = Path("results") / args.algorithm
    if not isinstance(args.policy_kwargs, dict):
        parser.error("--policy-kwargs must be a JSON object")
    if args.target_kl is not None and args.algorithm != "ppo":
        parser.error("--target-kl is a PPO option; use --policy-kwargs for other algorithms")
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
        with tf.device(device):
            policy_kwargs = dict(args.policy_kwargs)
            if args.target_kl is not None:
                policy_kwargs["target_kl"] = args.target_kl
            policy = make_policy(
                args.algorithm,
                env.single_observation_space,
                env.single_action_space,
                horizon=count,
                batch_size=args.batch_size,
                epochs=args.epochs,
                units=args.units,
                learning_rate=args.learning_rate,
                gpu=args.gpu,
                policy_kwargs=policy_kwargs,
            )
            trainer = VectorOnPolicyTrainer(policy, env, args.rollout_steps, args.seed)
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
                                tf.summary.scalar(f"{policy.policy_name}/{key}", value)
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
                "actor_devices": sorted(
                    {
                        v.value.device
                        for v in getattr(getattr(policy, "actor", None), "variables", ())
                    }
                ),
                "critic_devices": sorted(
                    {
                        v.value.device
                        for v in getattr(getattr(policy, "critic", None), "variables", ())
                    }
                ),
                "policy_devices": sorted({v.value.device for v in policy.variables}),
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
