"""Replay semantics shared by online collection and expert demonstrations."""

from enum import Enum

import numpy as np


class Done(Enum):
    """Three-valued DAC replay masks: absorbing -1, terminal 0, and continuing 1."""

    ABSORBING = -1.0
    DONE = 0.0
    NOT_DONE = 1.0


def dac_transitions(
    env, obs, action, next_obs, reward, terminated, truncated=False, *, absorbing=True, logp=None
):
    """Return a physical transition and, on termination, one absorbing self-loop.

    ``done`` is the historical three-valued DAC mask, not Gymnasium's termination
    flag. True termination wins if both flags are set. Truncation alone bootstraps.
    Observations are already augmented by the wrapper. No flags are inferred from
    episode length here; legacy demonstration loading handles that separately.

    Args:
        env (gym.Env): Environment exposing get_absorbing_state through its wrappers when
            absorbing=True.
        obs (array_like): Current observation, or batch of observations with a leading batch axis.
        action (array_like): Single action in the environment action space.
        next_obs (array_like): Physical successor observation or observation batch, before episode
            autoreset.
        reward (float): Scalar reward of the physical environment transition.
        terminated (array_like): True MDP termination flags; time-limit truncations must be supplied
            separately.
        truncated (array_like): Time-limit or external episode-boundary flags; these stop return
            recursion but permit value bootstrapping. This implementation does not read this
            argument.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.
        logp (array_like or None): Action log probability for each transition; also stored on
            synthetic rows when supplied.

    Returns:
        list[dict]: One physical transition, plus a zero-action absorbing self-loop after true
            termination when absorbing is enabled. done stores a DAC mask, not a boolean;
            truncations stay continuing.
    """
    use_absorbing = absorbing and bool(terminated)
    transition = dict(
        obs=obs,
        act=action,
        next_obs=next_obs,
        rew=reward,
        done=Done.DONE.value if terminated else Done.NOT_DONE.value,
    )
    if logp is not None:
        transition["logp"] = logp
    if use_absorbing:
        state = env.get_wrapper_attr("get_absorbing_state")()
        transition["next_obs"] = state
        loop = dict(
            obs=state.copy(),
            act=np.zeros(env.action_space.shape, dtype=np.float32),
            next_obs=state.copy(),
            rew=0.0,
            done=Done.ABSORBING.value,
        )
        if logp is not None:
            loop["logp"] = 0.0
        return [transition, loop]
    return [transition]


def dac_batch_transitions(env, obs, actions, next_obs, rewards, terminated, *, absorbing=True):
    """Vector equivalent of dac_transitions, with unbatched spaces and no per-world loop.

    next_obs must contain final physical observations, never autoreset observations.
    Time-limit truncation is continuing; only true terminations create self-loops.
    Physical rows precede synthetic rows. Added loops do not count as environment steps.

    Args:
        env (gym.vector.VectorEnv): Vector environment exposing a canonical absorbing observation
            when absorbing=True.
        obs (array_like): Current observation, or batch of observations with a leading batch axis.
        actions (array_like): Actions aligned with the observation batch; continuous actions have a
            final action_dim axis.
        next_obs (array_like): Physical successor observation or observation batch, before episode
            autoreset.
        rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
            have a trailing singleton axis.
        terminated (array_like): True MDP termination flags; time-limit truncations must be supplied
            separately.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        dict[str, np.ndarray]: Replay fields with physical rows followed by any terminal absorbing
            loops. rew and done have shape (rows, 1); done uses DAC masks (1, 0, -1).
    """
    terminated = np.asarray(terminated, bool).reshape(-1)
    rows = {
        "obs": np.asarray(obs, np.float32),
        "act": np.asarray(actions, np.float32),
        "next_obs": np.asarray(next_obs, np.float32).copy(),
        "rew": np.asarray(rewards, np.float32).reshape(-1, 1),
        "done": np.where(terminated, Done.DONE.value, Done.NOT_DONE.value)
        .astype(np.float32)
        .reshape(-1, 1),
    }
    count = int(terminated.sum()) if absorbing else 0
    if count:
        state = env.get_absorbing_state()
        rows["next_obs"][terminated] = state
        loops = {
            "obs": np.repeat(state[None], count, axis=0),
            "act": np.zeros((count, *env.single_action_space.shape), np.float32),
            "next_obs": np.repeat(state[None], count, axis=0),
            "rew": np.zeros((count, 1), np.float32),
            "done": np.full((count, 1), Done.ABSORBING.value, np.float32),
        }
        rows = {key: np.concatenate((value, loops[key]), axis=0) for key, value in rows.items()}
    return rows
