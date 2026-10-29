"""Trajectory I/O and consistent DAC augmentation for expert demonstrations."""

import re
import warnings
from pathlib import Path

import joblib
import numpy as np

from tf2rl.data.transitions import dac_transitions


def save_path(samples, filename):
    """Save trajectory samples as a compressed joblib file.

    Args:
        samples (dict[str, array_like]): Trajectory fields to serialize, including observations,
            actions, rewards, and boundary metadata as available.
        filename (str or Path): Destination path for the compressed joblib trajectory file.

    Returns:
        None.
    """
    joblib.dump(samples, filename, compress=3)


def get_filenames(dirname, n_path=None):
    """List trajectory filenames by descending training step and episode, optionally limiting the
    count.

    Args:
        dirname (str or Path): Directory containing saved expert trajectories.
        n_path (int or None): Maximum number of newest trajectory files to select; None selects all
            files.

    Returns:
        list[str]: Trajectory paths sorted newest first by numeric step and episode indices, limited
            by n_path.
    """
    pattern = re.compile(r"step_(\d+)_epi_(\d+)_return_.*\.pkl$")
    matches = [
        (int(match[1]), int(match[2]), path)
        for path in Path(dirname).iterdir()
        if (match := pattern.fullmatch(path.name))
    ]
    paths = [str(path) for _, _, path in sorted(matches, reverse=True)]
    return paths if n_path is None else paths[:n_path]


def _augment_observation(obs, env):
    """Add the absorbing indicator only if the observation is not already augmented.

    Args:
        obs (array_like): Current observation, or batch of observations with a leading batch axis.
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.

    Returns:
        np.ndarray: Original observation when already compatible, otherwise the observation
            augmented by the environment absorbing wrapper.
    """
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape == env.observation_space.shape:
        # Already augmented exports must not acquire a second indicator.
        return obs
    return env.get_wrapper_attr("get_non_absorbing_state")(obs)


def _load(filenames, max_steps=None, env=None, with_logps=False, absorbing=True):
    """Load and validate expert transitions, optionally adding DAC masks and log probabilities.

    The returned arrays use the historical keys, including ``not_dones`` for boolean done flags or
    three-valued DAC masks when an environment is supplied.

    Args:
        filenames (sequence[str or Path]): Trajectory files to load, in processing order.
        max_steps (int or None): Maximum transitions loaded from each trajectory file; None reads
            every transition.
        env (gym.Env or None): Environment with absorbing-observation helpers used for DAC
            conversion; None preserves raw trajectory observations and flags.
        with_logps (bool): Require and load stored expert action log probabilities.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        dict[str, np.ndarray]: Float32 obses, next_obses, acts, and column-vector not_dones, plus
            column-vector logps when requested. Raw legacy loading preserves done flags; supplying
            env converts ordinary transitions to DAC masks and optional absorbing rows. Already-DAC-
            encoded files retain their stored masks/rows.

    Raises:
        ValueError: No expert trajectory files found. Expert trajectories contain non-finite values.
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
            warnings.warn(
                "Legacy demonstration lacks termination/truncation metadata; "
                "done at the environment horizon is treated as truncation.",
                UserWarning,
            )
        horizon = env.spec.max_episode_steps if env is not None and env.spec is not None else None
        episode_step = 0
        for index in range(length):
            episode_step += 1
            obs, next_obs = path["obs"][index], path["next_obs"][index]
            if env is not None:
                obs, next_obs = _augment_observation(obs, env), _augment_observation(next_obs, env)
            logp = np.asarray(path["logp"][index]).reshape(-1)[0] if with_logps else None
            if dac_encoded:
                row = dict(
                    obs=obs,
                    act=path["act"][index],
                    next_obs=next_obs,
                    done=float(np.asarray(path["done"][index]).item()),
                )
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
                row = dict(
                    obs=obs,
                    act=path["act"][index],
                    next_obs=next_obs,
                    done=bool(np.asarray(path["done"][index]).item()),
                )
                if with_logps:
                    row["logp"] = logp
                rows.append(row)
            else:
                rows.extend(
                    dac_transitions(
                        env,
                        obs,
                        path["act"][index],
                        next_obs,
                        0.0,
                        terminated,
                        truncated,
                        absorbing=absorbing,
                        logp=logp,
                    )
                )
            if terminated or truncated:
                episode_step = 0
    output = {
        "obses": np.asarray([row["obs"] for row in rows], np.float32),
        "next_obses": np.asarray([row["next_obs"] for row in rows], np.float32),
        "acts": np.asarray([row["act"] for row in rows], np.float32),
        "not_dones": np.asarray([row["done"] for row in rows], np.float32).reshape(-1, 1),
    }
    if with_logps:
        output["logps"] = np.asarray([row["logp"] for row in rows], np.float32).reshape(-1, 1)
    if not all(np.isfinite(array).all() for array in output.values()):
        raise ValueError("Expert trajectories contain non-finite values")
    return output


def load_trajectories(filenames, max_steps=None, env=None, absorbing=True):
    """Load expert transition arrays, optionally augmenting observations and DAC masks.

    Args:
        filenames (sequence[str or Path]): Trajectory files to load, in processing order.
        max_steps (int or None): Maximum transitions loaded from each trajectory file; None reads
            every transition.
        env (gym.Env or None): Environment with absorbing-observation helpers for DAC conversion;
            None loads raw trajectory fields.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        dict[str, np.ndarray]: Float32 obses, next_obses, acts, and column-vector not_dones, plus
            column-vector logps when requested. Raw legacy loading preserves done flags; supplying
            env converts ordinary transitions to DAC masks and optional absorbing rows. Already-DAC-
            encoded files retain their stored masks/rows.
    """
    return _load(filenames, max_steps, env, absorbing=absorbing)


def load_trajectories_with_logps(filenames, max_steps=None, env=None, absorbing=True):
    """Load expert transitions and action log probabilities, optionally applying DAC
    augmentation.

    Args:
        filenames (sequence[str or Path]): Trajectory files to load, in processing order.
        max_steps (int or None): Maximum transitions loaded from each trajectory file; None reads
            every transition.
        env (gym.Env or None): Environment with absorbing-observation helpers for DAC conversion;
            None loads raw trajectory fields.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        dict[str, np.ndarray]: Float32 obses, next_obses, acts, and column-vector not_dones, plus
            column-vector logps when requested. Raw legacy loading preserves done flags; supplying
            env converts ordinary transitions to DAC masks and optional absorbing rows. Already-DAC-
            encoded files retain their stored masks/rows.
    """
    return _load(filenames, max_steps, env, with_logps=True, absorbing=absorbing)


def restore_latest_n_traj(
    dirname, n_path=10, max_steps=None, is_logps=False, env=None, absorbing=True
):
    """Load the newest trajectory files with optional log probabilities and DAC augmentation.

    Args:
        dirname (str or Path): Directory containing saved expert trajectories.
        n_path (int or None): Maximum number of newest trajectory files to select; None selects all
            files.
        max_steps (int or None): Maximum transitions loaded from each trajectory file; None reads
            every transition.
        is_logps (bool): Require and load stored expert action log probabilities.
        env (gym.Env or None): Environment with absorbing-observation helpers for DAC conversion;
            None loads raw trajectory fields.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        dict[str, np.ndarray]: Float32 obses, next_obses, acts, and column-vector not_dones, plus
            column-vector logps when requested. Raw legacy loading preserves done flags; supplying
            env converts ordinary transitions to DAC masks and optional absorbing rows. Already-DAC-
            encoded files retain their stored masks/rows.
    """
    filenames = get_filenames(dirname, n_path)
    return _load(filenames, max_steps, env, with_logps=is_logps, absorbing=absorbing)
