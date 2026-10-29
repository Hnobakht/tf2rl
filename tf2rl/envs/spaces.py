from gymnasium.spaces import Box, Discrete


def is_discrete(space):
    """Return whether an action space is Discrete; reject spaces other than Discrete or Box."""
    if isinstance(space, Discrete):
        return True
    elif isinstance(space, Box):
        return False
    else:
        raise NotImplementedError


def get_act_dim(action_space):
    """Return the number of discrete actions or the continuous action-vector dimension."""
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.low.size
    else:
        raise NotImplementedError


def is_mujoco_env(env):
    """Return whether the environment uses a MuJoCo base environment."""
    try:
        from gymnasium.envs.mujoco import MujocoEnv
    except ImportError:
        return False
    return isinstance(env.unwrapped, MujocoEnv)


def is_atari_env(env):
    """Return whether the environment is an Atari environment."""
    return hasattr(env.unwrapped, "ale")


def get_replay_act_dim(env):
    """
    Return replay action width: one for discrete actions, otherwise the number of Box coordinates.
    """
    if isinstance(env.action_space, Discrete):
        return 1  # env.action_space.n
    elif isinstance(env.action_space, Box):
        return env.action_space.low.size
    else:
        raise NotImplementedError
