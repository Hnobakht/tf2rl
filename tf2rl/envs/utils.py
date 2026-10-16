from gymnasium.spaces import Discrete, Box


def is_discrete(space):
    if isinstance(space, Discrete):
        return True
    elif isinstance(space, Box):
        return False
    else:
        raise NotImplementedError


def get_act_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.low.size
    else:
        raise NotImplementedError


def is_mujoco_env(env):
    try:
        from gymnasium.envs.mujoco import MujocoEnv
    except ImportError:
        return False
    return isinstance(env.unwrapped, MujocoEnv)


def is_atari_env(env):
    return hasattr(env.unwrapped, "ale")
