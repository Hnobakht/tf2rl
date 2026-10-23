"""Play a registered on-policy checkpoint in native Gymnasium or export an MP4."""

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.experiments.on_policy import make_policy


def main(argv=None):
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
    main()
