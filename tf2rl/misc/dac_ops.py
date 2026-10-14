"""Replay semantics shared by online collection and expert demonstrations."""
from enum import Enum

import numpy as np


class Done(Enum):
    ABSORBING = -1.0
    DONE = 0.0
    NOT_DONE = 1.0


def dac_transitions(env, obs, action, next_obs, reward, terminated, truncated=False,
                    *, absorbing=True, logp=None):
    """Return a physical transition and, on termination, one absorbing self-loop.

    ``done`` is the historical three-valued DAC mask, not Gymnasium's termination
    flag. True termination wins if both flags are set. Truncation alone bootstraps.
    Observations are already augmented by the wrapper. No flags are inferred from
    episode length here; legacy demonstration loading handles that separately.
    """
    use_absorbing = absorbing and bool(terminated)
    transition = dict(obs=obs, act=action, next_obs=next_obs, rew=reward,
                      done=Done.DONE.value if terminated else Done.NOT_DONE.value)
    if logp is not None:
        transition["logp"] = logp
    if use_absorbing:
        state = env.get_wrapper_attr("get_absorbing_state")()
        transition["next_obs"] = state
        loop = dict(obs=state.copy(), act=np.zeros(env.action_space.shape, dtype=np.float32),
                    next_obs=state.copy(), rew=0.0, done=Done.ABSORBING.value)
        if logp is not None:
            loop["logp"] = 0.0
        return [transition, loop]
    return [transition]
