import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete

from cpprb import ReplayBuffer, PrioritizedReplayBuffer

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.envs.utils import is_discrete


def get_space_size(space):
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [1]
    else:
        raise NotImplementedError("Assuming to use Box or Discrete, not {}".format(type(space)))


def get_default_rb_dict(size, env):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "obs": {
                "shape": get_space_size(env.observation_space)},
            "next_obs": {
                "shape": get_space_size(env.observation_space)},
            "act": {
                "shape": get_space_size(env.action_space)},
            "rew": {},
            "done": {}}}


def get_replay_buffer(policy, env, irl=None, save_logp=False, use_prioritized_rb=False,
                      use_nstep_rb=False, n_step=1, size=None):
    if policy is None or env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(policy.memory_capacity, env)

    if irl is not None and getattr(irl, "requires_log_probs", False):
        kwargs["env_dict"]["logp"] = {}

    if save_logp:
        kwargs["env_dict"]["logp"] = {}

    if size is not None:
        kwargs["size"] = size

    if not issubclass(type(policy), OffPolicyAgent):
        kwargs["size"] = policy.horizon
        kwargs["env_dict"].pop("next_obs") if irl is None else None
        kwargs["env_dict"].pop("rew")
        kwargs["env_dict"]["logp"] = {}
        kwargs["env_dict"]["ret"] = {}
        kwargs["env_dict"]["adv"] = {}
        if is_discrete(env.action_space):
            kwargs["env_dict"]["act"]["dtype"] = np.int32
        return ReplayBuffer(**kwargs)

    if use_prioritized_rb and use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return PrioritizedReplayBuffer(**kwargs)

    if len(obs_shape) == 3:
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte

    if use_prioritized_rb:
        return PrioritizedReplayBuffer(**kwargs)

    if use_nstep_rb:
        kwargs["Nstep"] = {"size": n_step,
                           "gamma": policy.discount,
                           "rew": "rew",
                           "next": "next_obs"}
        return ReplayBuffer(**kwargs)

    return ReplayBuffer(**kwargs)

def get_replay_buffer_no_policy(env, save_logp=True, size=None):
    if env is None:
        return None

    obs_shape = get_space_size(env.observation_space)
    kwargs = get_default_rb_dict(int(1e6), env)

    if save_logp:
        kwargs["env_dict"]["logp"] = {}

    if size is not None:
        kwargs["size"] = size

    if len(obs_shape) == 3:
        kwargs["env_dict"]["obs"]["dtype"] = np.ubyte
        kwargs["env_dict"]["next_obs"]["dtype"] = np.ubyte

    return ReplayBuffer(**kwargs)
