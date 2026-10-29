"""Select a batched environment backend without coupling it to an algorithm."""

import gymnasium as gym
import numpy as np


def make_vector_env(
    env_id,
    num_envs=1,
    *,
    backend="gymnasium",
    tensors=False,
    device="cuda:0",
    absorbing=False,
    absorbing_indicator=0.1,
    **kwargs,
):
    """Create same-step-autoreset environments with a consistent five-value API.

    MJWarp is optional and imported only when selected. kwargs are forwarded to
    the selected backend (Gymnasium constructor options for the CPU backend).

    Args:
        env_id (str): Registered Gymnasium environment identifier.
        num_envs (int): Number of independent simulation worlds.
        backend (str): Simulation backend: gymnasium or mjwarp.
        tensors (bool): Export observations and step results as TensorFlow tensors instead of NumPy
            arrays.
        device (str): TensorFlow or Warp device identifier appropriate to this operation.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.
        absorbing_indicator (float): Positive final-coordinate value used to identify the canonical
            absorbing observation.
        **kwargs (dict): Additional keyword arguments forwarded to the wrapped operation.

    Returns:
        gym.vector.VectorEnv: Same-step-autoreset batched environment on the requested backend,
            optionally wrapped for tensors and absorbing observations.

    Raises:
        ImportError: MJWarp requires the optional 'mjwarp' dependencies.
    """
    if backend == "mjwarp":
        try:
            from tf2rl.envs.mjwarp import MJWarpVectorEnv
        except ImportError as error:
            raise ImportError("MJWarp requires the optional 'mjwarp' dependencies") from error
        env = MJWarpVectorEnv(env_id, num_envs, tensors=tensors, device=device, **kwargs)
    elif backend == "gymnasium":
        env = gym.vector.SyncVectorEnv(
            [lambda: gym.make(env_id, **kwargs)] * num_envs,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        if tensors:
            env = TensorVectorEnv(env, device)
    else:
        raise ValueError(f"Unknown environment backend {backend!r}")
    if absorbing:
        from tf2rl.envs.wrappers.absorbing import VectorAbsorbingWrapper

        try:
            env = VectorAbsorbingWrapper(env, absorbing_indicator)
        except BaseException:
            env.close()
            raise
    return env


class TensorVectorEnv(gym.vector.VectorWrapper):
    """Expose native CPU vector environments using the same eager tensor API."""

    def __init__(self, env, device):
        """Initialize TensorVectorEnv and its configuration.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            device (str): TensorFlow or Warp device identifier appropriate to this operation.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(env)
        self.device = device.replace("cuda:", "/GPU:")

    def _tensor(self, value):
        """Convert a wrapped environment array to a boolean or float32 tensor on the selected
        device.

        Args:
            value (array_like): Value to convert, normalize, or store in the representation required by
                this operation.

        Returns:
            tf.Tensor: Input converted on the configured device to bool for boolean arrays, otherwise
                float32.
        """
        import tensorflow as tf

        value = np.asarray(value)
        with tf.device(self.device):
            return tf.convert_to_tensor(value, tf.bool if value.dtype == bool else tf.float32)

    def reset(self, **kwargs):
        """Reset all worlds and convert their initial observations to tensors.

        Args:
            **kwargs (dict): Reset options forwarded to the wrapped environment, such as seed and
                options.

        Returns:
            tuple[tf.Tensor, dict]: Batched initial observations and reset information returned by the
                wrapped vector environment.
        """
        obs, info = self.env.reset(**kwargs)
        return self._tensor(obs), info

    def step(self, actions):
        """Clip and execute batched tensor actions, then expose tensor results and physical
        successor observations.

        Args:
            actions (tf.Tensor): Batched actions exposing .numpy(), with shape (num_envs,
                *action_shape). Continuous actions are clipped before stepping.

        Returns:
            tuple: Tensor observations, rewards, terminations, truncations, and info. info["final_obs"]
                contains dense physical successor observations before autoreset.
        """
        actions = actions.numpy()
        if isinstance(self.single_action_space, gym.spaces.Box):
            actions = np.clip(actions, self.single_action_space.low, self.single_action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(actions)
        # Gymnasium provides object arrays for final observations. Supply a dense
        # batch for value bootstrapping, with regular observations on other rows.
        final = obs.copy()
        if "final_obs" in info:
            mask = info["_final_obs"]
            final[mask] = np.stack(info["final_obs"][mask])
        info["final_obs"] = self._tensor(final)
        return (
            self._tensor(obs),
            self._tensor(reward),
            self._tensor(terminated),
            self._tensor(truncated),
            info,
        )
