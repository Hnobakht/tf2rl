"""Continuous off-policy training with selectable CPU/MJWarp vector simulation.

SAC, TD3 and DDPG share the collector and optional DAC absorbing-state semantics,
using exactly the same masks as IL/IRL training.
"""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.envs.vector_env import make_vector_env
from tf2rl.experiments.demonstrations import evaluate_vector_policy
from tf2rl.experiments.runtime import ConfigurationError, configure_tensorflow, save_run_config
from tf2rl.experiments.vector_off_policy_trainer import VectorOffPolicyTrainer


def make_policy(
    algorithm,
    observation_space,
    action_space,
    *,
    absorbing=False,
    gpu=-1,
    units=(64, 64),
    learning_rate=3e-4,
    batch_size=256,
    memory_capacity=1000000,
    n_warmup=10000,
    auto_alpha=True,
    alpha=0.2,
    sigma=0.1,
    xla=False,
):
    """Construct an existing algorithm without embedding it in the environment backend."""
    if (
        not isinstance(observation_space, gym.spaces.Box)
        or len(observation_space.shape) != 1
        or not isinstance(action_space, gym.spaces.Box)
        or len(action_space.shape) != 1
    ):
        raise ValueError("This entry point requires vector Box observations and controls")
    if not np.allclose(action_space.high, -action_space.low) or not np.allclose(
        action_space.high, action_space.high[0]
    ):
        raise ValueError("Existing actors require uniform symmetric action bounds")
    if xla and algorithm != "sac":
        raise ValueError("XLA compilation is currently exposed for SAC")
    common = {
        "state_shape": observation_space.shape,
        "action_dim": action_space.shape[0],
        "max_action": float(action_space.high[0]),
        "actor_units": tuple(units),
        "critic_units": tuple(units),
        "gpu": gpu,
        "batch_size": batch_size,
        "memory_capacity": memory_capacity,
        "n_warmup": n_warmup,
    }
    with tf.device(f"/GPU:{gpu}" if gpu >= 0 else "/CPU:0"):
        if algorithm == "sac":
            if absorbing:
                # Keep the historical architecture for existing SAC checkpoints;
                # both SAC variants implement the shared absorbing-state rules.
                from tf2rl.algos.sac_irl import SAC_IRL

                cls = SAC_IRL
                common["is_absorbed_state"] = True
            else:
                from tf2rl.algos.sac import SAC

                cls = SAC
            return cls(
                **common,
                lr=learning_rate,
                lr_alpha=learning_rate,
                auto_alpha=auto_alpha,
                alpha=alpha,
                is_xla=xla,
            )
        if algorithm == "td3":
            from tf2rl.algos.td3 import TD3

            cls = TD3
        elif algorithm == "ddpg":
            from tf2rl.algos.ddpg import DDPG

            cls = DDPG
        else:
            raise ValueError(f"Unknown off-policy algorithm {algorithm!r}")
        return cls(**common, lr_actor=learning_rate, lr_critic=learning_rate, sigma=sigma,
                   is_absorbed_state=absorbing)


def checkpoint_prefix(directory):
    """Find managed checkpoints or the final prefix used by the experiment entry points."""
    directory = Path(directory)
    if Path(str(directory) + ".index").exists():
        return str(directory)
    prefix = tf.train.latest_checkpoint(str(directory / "checkpoints"))
    prefix = prefix or tf.train.latest_checkpoint(str(directory))
    if prefix is None and (directory / "checkpoint.index").exists():
        prefix = str(directory / "checkpoint")
    if prefix is None:
        raise ValueError(f"No policy checkpoint in {directory}")
    return prefix


def build_parser(algorithm=None):
    """Build the compatible CLI without starting an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default=algorithm or "sac")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--env-gpu", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--cpu-inter-threads",
        type=int,
        default=1,
        help="CPU workers for independent TensorFlow graph operations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument(
        "--collection-steps",
        type=int,
        default=16,
        help="Vector steps per progress report; updates occur after each step",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-capacity", type=int, default=1000000)
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10000,
        help="Random warmup physical transitions, aggregated over all worlds",
    )
    parser.add_argument(
        "--updates-per-transition",
        type=float,
        default=1.0,
        help="Gradient minibatch updates per new physical transition after warmup",
    )
    parser.add_argument("--units", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--fixed-alpha", action="store_true")
    parser.add_argument(
        "--xla", action="store_true", help="Compile SAC with XLA; adds startup cost"
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument(
        "--absorbing",
        action="store_true",
        help="Enable DAC absorbing observations and targets for SAC, TD3 or DDPG",
    )
    parser.add_argument("--absorbing-indicator", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--eval-env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--save-expert", action="store_true")
    parser.add_argument("--expert-episodes", type=int, default=20)
    parser.add_argument("--expert-dir", type=Path)
    parser.add_argument("--stochastic-expert", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Restore a policy and evaluate/export without training",
    )
    parser.add_argument("--output", type=Path, default=Path("results/off_policy"))
    return parser


def run(args):
    """Run one experiment from parsed options in a fresh TensorFlow process."""
    if (
        min(
            args.num_envs,
            args.max_steps,
            args.collection_steps,
            args.batch_size,
            args.memory_capacity,
            args.cpu_threads,
            args.cpu_inter_threads,
            args.checkpoint_interval,
            args.eval_interval,
            args.eval_num_envs,
            args.expert_episodes,
            *args.units,
        )
        < 1
    ):
        raise ConfigurationError("Counts must be positive")
    if args.n_warmup < 0 or args.eval_episodes < 0:
        raise ConfigurationError("Warmup and evaluation counts must be nonnegative")
    if not np.isfinite(args.updates_per_transition) or args.updates_per_transition < 0:
        raise ConfigurationError("updates-per-transition must be finite and nonnegative")
    if args.memory_capacity < args.batch_size:
        raise ConfigurationError("memory-capacity must be at least batch-size")
    if args.xla and args.algorithm != "sac":
        raise ConfigurationError("--xla currently supports SAC")
    if args.evaluate and args.model_dir is None:
        raise ConfigurationError("--evaluate requires --model-dir")
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpu >= len(gpus):
        raise ConfigurationError("--gpu must name a visible GPU or be -1 for CPU")
    if "mjwarp" in (args.env_backend, args.eval_env_backend) and not 0 <= args.env_gpu < len(gpus):
        raise ConfigurationError("MJWarp requires a visible --env-gpu")
    configure_tensorflow(args.seed, args.cpu_threads, args.cpu_inter_threads)
    env_kwargs = dict(args.env_kwargs)
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    def environment(backend, count):
        sim_kwargs = {"nconmax": args.nconmax, "njmax": args.njmax} if backend == "mjwarp" else {}
        device = (
            f"cuda:{args.env_gpu}"
            if backend == "mjwarp"
            else (f"/GPU:{args.gpu}" if args.gpu >= 0 else "/CPU:0")
        )
        return make_vector_env(
            args.env_name,
            count,
            backend=backend,
            tensors=True,
            device=device,
            absorbing=args.absorbing,
            absorbing_indicator=args.absorbing_indicator,
            **sim_kwargs,
            **env_kwargs,
        )

    env = environment(args.env_backend, args.num_envs)
    evaluation_env = None
    writer = None
    try:
        policy = make_policy(
            args.algorithm,
            env.single_observation_space,
            env.single_action_space,
            absorbing=args.absorbing,
            gpu=args.gpu,
            units=args.units,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            memory_capacity=args.memory_capacity,
            n_warmup=args.n_warmup,
            auto_alpha=not args.fixed_alpha,
            alpha=args.alpha,
            sigma=args.sigma,
            xla=args.xla,
        )
        restored = None
        if args.model_dir is not None:
            restored = checkpoint_prefix(args.model_dir)
            tf.train.Checkpoint(policy=policy).read(
                restored
            ).assert_existing_objects_matched().expect_partial()
        config = save_run_config(args)
        writer = tf.summary.create_file_writer(str(args.output / "events"))
        checkpoint = tf.train.Checkpoint(policy=policy)
        manager = tf.train.CheckpointManager(
            checkpoint, str(args.output / "checkpoints"), max_to_keep=5
        )
        trainer = (
            None
            if args.evaluate
            else VectorOffPolicyTrainer(
                policy,
                env,
                absorbing=args.absorbing,
                seed=args.seed,
                updates_per_transition=args.updates_per_transition,
            )
        )
        records, evaluations = [], []
        if args.eval_episodes or args.save_expert:
            evaluation_env = environment(
                args.eval_env_backend,
                min(
                    args.eval_num_envs,
                    max(args.eval_episodes, args.expert_episodes if args.save_expert else 1),
                ),
            )
        next_save, next_eval = args.checkpoint_interval, args.eval_interval
        with writer.as_default(step=0), tf.device(policy.device):
            while trainer is not None and trainer.total_steps < args.max_steps:
                start = time.perf_counter()
                collected = []
                updates = 0
                collection_seconds = update_seconds = 0.0
                for index in range(args.collection_steps):
                    if trainer.total_steps >= args.max_steps:
                        break
                    before = time.perf_counter()
                    collected.append(trainer.collect_step())
                    collection_seconds += time.perf_counter() - before
                    before = time.perf_counter()
                    report = (
                        index == args.collection_steps - 1 or trainer.total_steps >= args.max_steps
                    )
                    with writer.as_default(step=trainer.total_steps):
                        updates += trainer.update(record_summaries=report)["updates"]
                    update_seconds += time.perf_counter() - before
                if hasattr(env, "check_overflow"):
                    env.check_overflow()
                elapsed = time.perf_counter() - start
                transitions = sum(row["transitions"] for row in collected)
                episodes = sum(row["episodes"] for row in collected)
                record = {
                    "total_steps": trainer.total_steps,
                    "transitions": transitions,
                    "replay_size": trainer.replay_buffer.get_stored_size(),
                    "total_updates": trainer.total_updates,
                    "updates": updates,
                    "collection_seconds": collection_seconds,
                    "update_seconds": update_seconds,
                    "steps_per_second": transitions / elapsed,
                    "seconds": elapsed,
                    "episodes": episodes,
                    "mean_step_reward": sum(row["reward_sum"] for row in collected) / transitions,
                    "episode_return": sum(row["return_sum"] for row in collected) / episodes
                    if episodes
                    else None,
                    "terminations": sum(row["terminations"] for row in collected),
                    "truncations": sum(row["truncations"] for row in collected),
                }
                records.append(record)
                with writer.as_default(step=trainer.total_steps):
                    for key, value in record.items():
                        if value is not None:
                            tf.summary.scalar(f"OffPolicy/{key}", value)
                if trainer.total_steps >= next_save:
                    manager.save(checkpoint_number=trainer.total_steps)
                    next_save = (
                        trainer.total_steps // args.checkpoint_interval + 1
                    ) * args.checkpoint_interval
                if args.eval_episodes and trainer.total_steps >= next_eval:
                    evaluation = evaluate_vector_policy(
                        policy, evaluation_env, args.eval_episodes, seed=args.seed + 10000
                    )
                    evaluations.append(dict(total_steps=trainer.total_steps, **evaluation))
                    with writer.as_default(step=trainer.total_steps):
                        tf.summary.scalar("OffPolicy/evaluation_return", evaluation["mean_return"])
                    next_eval = (trainer.total_steps // args.eval_interval + 1) * args.eval_interval
                writer.flush()
                print(json.dumps(record), flush=True)
            steps = trainer.total_steps if trainer is not None else 0
            if trainer is not None:
                manager.save(checkpoint_number=steps)
                checkpoint.write(str(args.output / "checkpoint"))
            final_evaluation = (
                evaluate_vector_policy(
                    policy, evaluation_env, args.eval_episodes, seed=args.seed + 10000
                )
                if args.eval_episodes
                else None
            )
            experts = None
            if args.save_expert:
                experts = evaluate_vector_policy(
                    policy,
                    evaluation_env,
                    args.expert_episodes,
                    seed=args.seed + 20000,
                    output_dir=args.expert_dir or args.output / "expert",
                    total_steps=steps,
                    deterministic=not args.stochastic_expert,
                    metadata={
                        "algorithm": args.algorithm,
                        "env_name": args.env_name,
                        "backend": args.eval_env_backend,
                        "absorbing": args.absorbing,
                        "absorbing_indicator": args.absorbing_indicator,
                    },
                )
            models = {"actor": policy.actor}
            for name in ("qf1", "qf2", "qf1_target", "qf2_target", "critic", "critic_target"):
                if hasattr(policy, name):
                    models[name] = getattr(policy, name)
            summary = {
                "config": config,
                "restored_checkpoint": restored,
                "total_steps": steps,
                "total_updates": trainer.total_updates if trainer is not None else 0,
                "replay_rows": trainer.replay_rows if trainer is not None else 0,
                "mask_encoding": policy.replay_mask_encoding,
                "model_devices": {
                    name: sorted({v.value.device for v in model.variables})
                    for name, model in models.items()
                },
                "iterations": records,
                "evaluations": evaluations,
                "final_evaluation": final_evaluation,
                "expert_export": experts,
            }
            (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in summary.items()
                        if key not in ("iterations", "evaluations")
                    }
                ),
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()
        if evaluation_env is not None:
            evaluation_env.close()
        env.close()


def main(argv=None, algorithm=None):
    """Keep existing command-line and example entry points compatible."""
    parser = build_parser(algorithm)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ConfigurationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
