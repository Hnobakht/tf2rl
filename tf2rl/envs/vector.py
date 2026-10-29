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
        super().__init__(env)
        self.device = device.replace("cuda:", "/GPU:")

    def _tensor(self, value):
        import tensorflow as tf

        value = np.asarray(value)
        with tf.device(self.device):
            return tf.convert_to_tensor(value, tf.bool if value.dtype == bool else tf.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._tensor(obs), info

    def step(self, actions):
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
