"""Trajectory I/O and consistent DAC augmentation for expert demonstrations."""
import re
import warnings
from pathlib import Path

import joblib
import numpy as np

from tf2rl.misc.dac_ops import dac_transitions


def save_path(samples, filename):
    """Save trajectory samples as a compressed joblib file."""
    joblib.dump(samples, filename, compress=3)


def get_filenames(dirname, n_path=None):
    """List trajectory filenames by descending training step and episode, optionally limiting the
    count.
    """
    pattern = re.compile(r"step_(\d+)_epi_(\d+)_return_.*\.pkl$")
    matches = [(int(match[1]), int(match[2]), path) for path in Path(dirname).iterdir()
               if (match := pattern.fullmatch(path.name))]
    paths = [str(path) for _, _, path in sorted(matches, reverse=True)]
    return paths if n_path is None else paths[:n_path]


def _augment_observation(obs, env):
    """Add the absorbing indicator only if the observation is not already augmented."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape == env.observation_space.shape:
        # Already augmented exports must not acquire a second indicator.
        return obs
    return env.get_wrapper_attr("get_non_absorbing_state")(obs)


def _load(filenames, max_steps=None, env=None, with_logps=False, absorbing=True):
    """Load and validate expert transitions, optionally adding DAC masks and log probabilities.

    The returned arrays use the historical keys, including ``not_dones`` for boolean done flags or
    three-valued DAC masks when an environment is supplied.
    """
    if not filenames:
        raise ValueError("No expert trajectory files found")
    rows = []
    for filename in filenames:
        path = joblib.load(filename)
        required = {"obs", "act", "next_obs", "done"} | ({"logp"} if with_logps else set())
        missing = required - path.keys()
        if missing:
            raise ValueError(f"{filename}: missing trajectory fields {sorted(missing)}")
        length = len(path["obs"])
        if not length or any(len(path[key]) != length for key in required):
            raise ValueError(f"{filename}: trajectory arrays must have equal nonzero lengths")
        length = min(length, max_steps) if max_steps is not None else length
        explicit = "terminated" in path and "truncated" in path
        dac_encoded = path.get("mask_encoding") == "dac"
        if env is not None and not explicit and not dac_encoded:
            warnings.warn("Legacy demonstration lacks termination/truncation metadata; "
                          "done at the environment horizon is treated as truncation.", UserWarning)
        horizon = env.spec.max_episode_steps if env is not None and env.spec is not None else None
        episode_step = 0
        for index in range(length):
            episode_step += 1
            obs, next_obs = path["obs"][index], path["next_obs"][index]
            if env is not None:
                obs, next_obs = _augment_observation(obs, env), _augment_observation(next_obs, env)
            logp = np.asarray(path["logp"][index]).reshape(-1)[0] if with_logps else None
            if dac_encoded:
                row = dict(obs=obs, act=path["act"][index], next_obs=next_obs,
                           done=float(np.asarray(path["done"][index]).item()))
                if with_logps:
                    row["logp"] = logp
                rows.append(row)
                continue
            if explicit:
                terminated = bool(np.asarray(path["terminated"][index]).item())
                truncated = bool(np.asarray(path["truncated"][index]).item())
            else:
                done = bool(np.asarray(path["done"][index]).item())
                truncated = done and horizon is not None and episode_step >= horizon
                terminated = done and not truncated
            if env is None:
                # Preserve the old raw-loader API (ordinary boolean done fields).
                row = dict(obs=obs, act=path["act"][index], next_obs=next_obs,
                           done=bool(np.asarray(path["done"][index]).item()))
                if with_logps:
                    row["logp"] = logp
                rows.append(row)
            else:
                rows.extend(dac_transitions(env, obs, path["act"][index], next_obs, 0.,
                                            terminated, truncated, absorbing=absorbing, logp=logp))
            if terminated or truncated:
                episode_step = 0
    output = {"obses": np.asarray([row["obs"] for row in rows], np.float32),
              "next_obses": np.asarray([row["next_obs"] for row in rows], np.float32),
              "acts": np.asarray([row["act"] for row in rows], np.float32),
              "not_dones": np.asarray([row["done"] for row in rows], np.float32).reshape(-1, 1)}
    if with_logps:
        output["logps"] = np.asarray([row["logp"] for row in rows], np.float32).reshape(-1, 1)
    if not all(np.isfinite(array).all() for array in output.values()):
        raise ValueError("Expert trajectories contain non-finite values")
    return output


def load_trajectories(filenames, max_steps=None, env=None, absorbing=True):
    """Load expert transition arrays, optionally augmenting observations and DAC masks."""
    return _load(filenames, max_steps, env, absorbing=absorbing)


def load_trajectories_with_logps(filenames, max_steps=None, env=None, absorbing=True):
    """Load expert transitions and action log probabilities, optionally applying DAC augmentation.
    """
    return _load(filenames, max_steps, env, with_logps=True, absorbing=absorbing)


def restore_latest_n_traj(dirname, n_path=10, max_steps=None, is_logps=False, env=None,
                          absorbing=True):
    """Load the newest trajectory files with optional log probabilities and DAC augmentation."""
    filenames = get_filenames(dirname, n_path)
    return _load(filenames, max_steps, env, with_logps=is_logps, absorbing=absorbing)


def evaluate_vector_policy(policy, env, episodes=10, *, seed=10000, output_dir=None,
                           total_steps=0, deterministic=True, metadata=None):
    """Evaluate a frozen policy and optionally export complete per-world episodes.

    Store physical final observations and explicit termination/truncation flags.
    DAC augmentation remains the loader's responsibility, so no absorbing loops
    are serialized or duplicated. Stochastic policies save densities evaluated
    at the exact executed actions, including deterministic evaluation actions.
    """
    import tensorflow as tf

    from tf2rl.experiments.vector_off_policy_trainer import numpy, physical_next_observations

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
            result = policy.get_action_tensor(tf.convert_to_tensor(obs, tf.float32),
                                               test=deterministic)
            action = result[0] if isinstance(result, tuple) else result
            action = tf.clip_by_value(action, env.single_action_space.low,
                                      env.single_action_space.high)
            logp = None
            if directory is not None and hasattr(policy, "compute_action_log_probs"):
                logp = numpy(policy.compute_action_log_probs(
                    tf.convert_to_tensor(obs, tf.float32), action)).reshape(-1)
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
                row = {"obs": states[world].copy(), "act": actions[world].copy(),
                       "next_obs": final[world].copy(), "rew": rewards[world], "done": boundary,
                       "terminated": bool(terminated[world]), "truncated": bool(truncated[world])}
                if logp is not None:
                    row["logp"] = logp[world]
                buffers[world].append(row)
            if boundary:
                if len(returns) < episodes:
                    episode_return = float(running_returns[world])
                    if buffers is not None:
                        rows = buffers[world]
                        samples = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                        samples.update(format_version=2, mask_encoding="gymnasium",
                                       metadata=dict(metadata or {}, deterministic=deterministic,
                                                     collection_step=total_steps, world=world))
                        filename = directory / (f"step_{total_steps:012d}_epi_{len(returns):04d}_"
                                                f"return_{episode_return:010.4f}.pkl")
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
    return {"mean_return": float(np.mean(returns)), "mean_length": float(np.mean(lengths)),
            "returns": returns, "lengths": lengths, "paths": paths}
