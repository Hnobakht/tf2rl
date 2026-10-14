"""Trajectory I/O and consistent DAC augmentation for expert demonstrations."""
from pathlib import Path
import re
import warnings

import joblib
import numpy as np

from tf2rl.misc.dac_ops import dac_transitions


def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def get_filenames(dirname, n_path=None):
    pattern = re.compile(r"step_(\d+)_epi_(\d+)_return_.*\.pkl$")
    matches = [(int(match[1]), int(match[2]), path) for path in Path(dirname).iterdir()
               if (match := pattern.fullmatch(path.name))]
    paths = [str(path) for _, _, path in sorted(matches, reverse=True)]
    return paths if n_path is None else paths[:n_path]


def _augment_observation(obs, env):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape == env.observation_space.shape:
        # Already augmented exports must not acquire a second indicator.
        return obs
    return env.get_wrapper_attr("get_non_absorbing_state")(obs)


def _load(filenames, max_steps=None, env=None, with_logps=False, absorbing=True):
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
    return _load(filenames, max_steps, env, absorbing=absorbing)


def load_trajectories_with_logps(filenames, max_steps=None, env=None, absorbing=True):
    return _load(filenames, max_steps, env, with_logps=True, absorbing=absorbing)


def restore_latest_n_traj(dirname, n_path=10, max_steps=None, is_logps=False, env=None,
                          absorbing=True):
    filenames = get_filenames(dirname, n_path)
    return _load(filenames, max_steps, env, with_logps=is_logps, absorbing=absorbing)
