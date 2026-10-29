from gymnasium.spaces import Box, Discrete


def is_discrete(space):
    """Return whether an action space is Discrete; reject spaces other than Discrete or Box.

    Args:
        space (gym.Space): Observation or action space whose storage shape is required.

    Returns:
        bool: True for supported discrete action spaces and False for continuous Box spaces.

    Raises:
        NotImplementedError: The requested operation or argument combination is unsupported by this
            component.
    """
    if isinstance(space, Discrete):
        return True
    elif isinstance(space, Box):
        return False
    else:
        raise NotImplementedError


def get_act_dim(action_space):
    """Return the number of discrete actions or the continuous action-vector dimension.

    Args:
        action_space (gym.Space): Single-environment action space used for validation or model
            construction.

    Returns:
        int: Number of discrete actions or total number of continuous action coordinates.

    Raises:
        NotImplementedError: The requested operation or argument combination is unsupported by this
            component.
    """
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.low.size
    else:
        raise NotImplementedError


def is_mujoco_env(env):
    """Return whether the environment uses a MuJoCo base environment.

    Args:
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.

    Returns:
        bool: Whether the unwrapped environment derives from Gymnasium MujocoEnv; False if MuJoCo
            support is unavailable.
    """
    try:
        from gymnasium.envs.mujoco import MujocoEnv
    except ImportError:
        return False
    return isinstance(env.unwrapped, MujocoEnv)


def is_atari_env(env):
    """Return whether the environment is an Atari environment.

    Args:
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.

    Returns:
        bool: Whether the unwrapped environment exposes the ALE emulator interface.
    """
    return hasattr(env.unwrapped, "ale")


def get_replay_act_dim(env):
    """Return replay action width: one for discrete actions, otherwise the number of Box
    coordinates.

    Args:
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.

    Returns:
        int: One stored coordinate for discrete action indices, otherwise the continuous action
            width.

    Raises:
        NotImplementedError: The requested operation or argument combination is unsupported by this
            component.
    """
    if isinstance(env.action_space, Discrete):
        return 1  # env.action_space.n
    elif isinstance(env.action_space, Box):
        return env.action_space.low.size
    else:
        raise NotImplementedError
