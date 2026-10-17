from gymnasium.spaces import Discrete, Box


def get_act_dim(env):
    """
    Return replay action width: one for discrete actions, otherwise the number of Box coordinates.
    """
    if isinstance(env.action_space, Discrete):
        return 1  # env.action_space.n
    elif isinstance(env.action_space, Box):
        return env.action_space.low.size
    else:
        raise NotImplementedError
