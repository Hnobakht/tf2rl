"""Evaluation and complete trajectory collection, separate from learning."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.data.demonstrations import save_path
from tf2rl.utils.images import frames_to_gif


def evaluate_policy(trainer, total_steps):
    """Evaluate the policy in the test environment and return its performance.

    Args:
        trainer (OffPolicyTrainer): Initialized trainer supplying policy, evaluation settings, and
            output paths.
        total_steps (int): Physical environment-step count used to label evaluation summaries and
            trajectory files.

    Returns:
        tuple[float, float]: Mean undiscounted return and mean length over completed scalar
            evaluation episodes; configured trajectories, videos, and summaries are also saved.
    """
    returns, lengths = [], []
    horizon = trainer._horizon(trainer._test_env)
    for episode in range(trainer._test_episodes):
        obs, _ = trainer._test_env.reset()
        rows, frames = [], []
        episode_return = 0.0
        for step in range(1, horizon + 1):
            result = trainer._policy.get_action(
                trainer._policy_observation(np.asarray(obs, np.float32)), test=True
            )
            action = trainer._action(result)
            next_obs, reward, terminated, truncated, _ = trainer._test_env.step(action)
            truncated = truncated or step >= horizon
            row = dict(
                obs=obs,
                act=action,
                next_obs=next_obs,
                rew=reward,
                done=terminated or truncated,
                terminated=terminated,
                truncated=truncated,
            )
            if isinstance(result, tuple):
                row["logp"] = result[1]
            rows.append(row)
            if trainer._save_test_movie:
                frames.append(trainer._test_env.render())
            elif trainer._show_test_progress:
                trainer._test_env.render()
            episode_return += reward
            obs = next_obs
            if terminated or truncated:
                break
        prefix = f"step_{total_steps:08d}_epi_{episode:02d}_return_{episode_return:010.4f}"
        if trainer._save_test_path:
            samples = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            samples["format_version"] = 2
            save_path(samples, str(Path(trainer._output_dir) / (prefix + ".pkl")))
        if trainer._save_test_movie:
            frames_to_gif(frames, prefix, trainer._output_dir)
        returns.append(episode_return)
        lengths.append(len(rows))
    return float(np.mean(returns)), float(np.mean(lengths))


def evaluate_on_policy(policy, env_id, episodes, seed, env_kwargs):
    """Evaluate deterministic actions in independent native Gymnasium episodes.

    Args:
        policy (Policy): Agent supplying action selection and learning operations.
        env_id (str): Registered Gymnasium environment identifier.
        episodes (int): Number of complete episodes to evaluate or collect.
        seed (int or None): Random seed; None leaves initialization to the underlying random
            generator.
        env_kwargs (dict): Keyword options passed to the Gymnasium environment constructor.

    Returns:
        dict or None: Episode returns and mean_return, or None when episodes is zero. The temporary
            environment is closed after evaluation.
    """
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


def evaluate_vector_policy(
    policy,
    env,
    episodes=10,
    *,
    seed=10000,
    output_dir=None,
    total_steps=0,
    deterministic=True,
    metadata=None,
):
    """Evaluate a frozen policy and optionally export complete per-world episodes.

    Store physical final observations and explicit termination/truncation flags.
    DAC augmentation remains the loader's responsibility, so no absorbing loops
    are serialized or duplicated. Stochastic policies save densities evaluated
    at the exact executed actions, including deterministic evaluation actions.

    Args:
        policy (Policy): Agent supplying action selection and learning operations.
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.
        episodes (int): Number of complete episodes to evaluate or collect.
        seed (int or None): Random seed; None leaves initialization to the underlying random
            generator.
        output_dir (str or Path or None): Directory receiving metrics, trajectories, or rendered
            artifacts.
        total_steps (int): Physical environment-step count used to label evaluation summaries and
            trajectory files.
        deterministic (bool): Use the policy evaluation action instead of sampling during trajectory
            collection.
        metadata (dict or None): Additional provenance recorded with trajectories or exported
            equations.

    Returns:
        dict: mean_return, mean_length, per-episode returns/lengths, and exported trajectory paths.
            Physical final observations are used in saved transitions.

    Raises:
        ValueError: episodes must be positive. Cannot export nonfinite action log probabilities.
            Cannot export nonfinite expert transitions.
    """
    import tensorflow as tf

    from tf2rl.training.collectors import numpy, physical_next_observations

    if episodes < 1:
        raise ValueError("episodes must be positive")
    directory = Path(output_dir) if output_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    buffers = [[] for _ in range(env.num_envs)] if directory is not None else None
    running_returns = np.zeros(env.num_envs, np.float64)
    running_lengths = np.zeros(env.num_envs, np.int64)
    returns, lengths, paths = [], [], []
    while len(returns) < episodes:
        states = numpy(obs)
        with tf.device(policy.device):
            result = policy.get_action_tensor(
                tf.convert_to_tensor(obs, tf.float32), test=deterministic
            )
            action = result[0] if isinstance(result, tuple) else result
            action = tf.clip_by_value(
                action, env.single_action_space.low, env.single_action_space.high
            )
            logp = None
            if directory is not None and hasattr(policy, "compute_action_log_probs"):
                logp = numpy(
                    policy.compute_action_log_probs(tf.convert_to_tensor(obs, tf.float32), action)
                ).reshape(-1)
                if not np.isfinite(logp).all():
                    raise ValueError("Cannot export nonfinite action log probabilities")
        next_obs, reward, terminated, truncated, info = env.step(action)
        terminated, truncated = numpy(terminated, bool), numpy(truncated, bool)
        final = physical_next_observations(next_obs, terminated, truncated, info)
        actions, rewards = numpy(action), numpy(reward)
        if not all(np.isfinite(value).all() for value in (states, actions, final, rewards)):
            raise ValueError("Cannot export nonfinite expert transitions")
        running_returns += rewards
        running_lengths += 1
        for world in range(env.num_envs):
            boundary = bool(terminated[world] or truncated[world])
            if buffers is not None:
                row = {
                    "obs": states[world].copy(),
                    "act": actions[world].copy(),
                    "next_obs": final[world].copy(),
                    "rew": rewards[world],
                    "done": boundary,
                    "terminated": bool(terminated[world]),
                    "truncated": bool(truncated[world]),
                }
                if logp is not None:
                    row["logp"] = logp[world]
                buffers[world].append(row)
            if boundary:
                if len(returns) < episodes:
                    episode_return = float(running_returns[world])
                    if buffers is not None:
                        rows = buffers[world]
                        samples = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                        samples.update(
                            format_version=2,
                            mask_encoding="gymnasium",
                            metadata=dict(
                                metadata or {},
                                deterministic=deterministic,
                                collection_step=total_steps,
                                world=world,
                            ),
                        )
                        filename = directory / (
                            f"step_{total_steps:012d}_epi_{len(returns):04d}_"
                            f"return_{episode_return:010.4f}.pkl"
                        )
                        save_path(samples, str(filename))
                        paths.append(str(filename))
                    returns.append(episode_return)
                    lengths.append(int(running_lengths[world]))
                if buffers is not None:
                    buffers[world] = []
                running_returns[world] = 0.0
                running_lengths[world] = 0
        obs = next_obs
    if hasattr(env, "check_overflow"):
        env.check_overflow()
    return {
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
        "paths": paths,
    }


import argparse
import json
import time


def view_policy(argv=None):
    """Parse viewer options, restore the requested policy, and run scalar evaluation with
    rendering.

    Args:
        argv (sequence[str] or None): Command-line tokens; None reads the current process arguments.

    Returns:
        None.

    Raises:
        RuntimeError: OpenCV could not open the MP4 encoder.
    """
    from tf2rl.experiments.build import make_on_policy_agent as make_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint prefix, without .index")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path, help="Export an MP4 instead of opening a viewer")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Episode and step counts must be positive")
    config_path = args.run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = json.loads((args.run_dir / "metrics.json").read_text())["config"]
    prefix = (
        str(args.checkpoint)
        if args.checkpoint
        else tf.train.latest_checkpoint(str(args.run_dir / "checkpoints"))
    )
    if prefix is None:
        prefix = str(args.run_dir / "checkpoint")
    if not Path(prefix + ".index").exists():
        parser.error(f"No checkpoint found at {prefix}; wait for the first checkpoint save")
    kwargs = dict(config.get("env_kwargs", {}))
    if config.get("max_episode_steps") is not None:
        kwargs["max_episode_steps"] = config["max_episode_steps"]
    if args.max_steps is not None:
        kwargs["max_episode_steps"] = args.max_steps
    if args.output:
        kwargs.update(width=args.width, height=args.height)
        if args.output.suffix.lower() != ".mp4":
            parser.error("--output must be an .mp4 file")
    env = gym.make(
        config["env_name"], render_mode="rgb_array" if args.output else "human", **kwargs
    )
    video = None
    try:
        with tf.device("/CPU:0"):
            policy_kwargs = dict(config.get("policy_kwargs", {}))
            if config.get("target_kl") is not None:
                policy_kwargs["target_kl"] = config["target_kl"]
            policy = make_policy(
                config.get("algorithm", "ppo"),
                env.observation_space,
                env.action_space,
                units=config["units"],
                horizon=config["num_envs"] * config["rollout_steps"],
                batch_size=config["batch_size"],
                gpu=-1,
                policy_kwargs=policy_kwargs,
            )
            policy.get_action_and_val_tensor(tf.zeros((1, *env.observation_space.shape)))
            # The viewer restores model weights; optimizer slots are unnecessary.
            tf.train.Checkpoint(policy=policy).read(
                prefix
            ).assert_existing_objects_matched().expect_partial()
            returns = []
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode)
                total = 0.0
                while True:
                    start = time.perf_counter()
                    if args.output:
                        import cv2

                        frame = env.render()
                        if video is None:
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            video = cv2.VideoWriter(
                                str(args.output),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                env.metadata["render_fps"],
                                (frame.shape[1], frame.shape[0]),
                            )
                            if not video.isOpened():
                                raise RuntimeError("OpenCV could not open the MP4 encoder")
                        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
                    if args.output is None:
                        time.sleep(
                            max(
                                0.0,
                                1.0 / env.metadata["render_fps"] - (time.perf_counter() - start),
                            )
                        )
                returns.append(total)
                print(json.dumps({"episode": episode + 1, "return": total}), flush=True)
            print(
                json.dumps(
                    {
                        "checkpoint": prefix,
                        "mean_return": float(np.mean(returns)),
                        "video": str(args.output) if args.output else None,
                    }
                ),
                flush=True,
            )
    finally:
        if video is not None:
            video.release()
        env.close()


if __name__ == "__main__":
    view_policy()
