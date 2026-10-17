"""Native Gymnasium observation augmentation for DAC absorbing states."""
import gymnasium as gym
import numpy as np


class AbsorbingWrapper(gym.ObservationWrapper):
    """Append zero to ordinary observations; represent absorption as [0, ..., indicator].

    The wrapper never synthesizes an environment step. Replay collection adds the
    terminal-to-absorbing transition and one self-loop using ``dac_transitions``.
    The historical experiment indicator is 0.1; the general default remains 1.
    """

    def __init__(self, env, indicator=1):
        """
        Validate vector observations and a positive absorbing indicator, then extend the observation
        space.
        """
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise ValueError("AbsorbingWrapper requires a one-dimensional Box observation space")
        if not np.isfinite(indicator) or indicator <= 0:
            raise ValueError("indicator must be finite and positive")
        self._indicator = indicator
        # Include the zero vector even when the original bounds do not contain it.
        self.observation_space = gym.spaces.Box(
            low=np.append(np.minimum(space.low, 0), 0).astype(space.dtype),
            high=np.append(np.maximum(space.high, 0), indicator).astype(space.dtype),
            dtype=space.dtype,
        )

    def observation(self, observation):
        """Append the non-absorbing indicator to the observation."""
        return self.get_non_absorbing_state(observation)

    def get_non_absorbing_state(self, obs):
        """Append the non-absorbing indicator to an ordinary observation."""
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        if obs.shape[-1:] != self.env.observation_space.shape:
            raise ValueError(f"Expected raw observation width {self.env.observation_space.shape}, got {obs.shape}")
        return np.concatenate((obs, np.zeros(obs.shape[:-1] + (1,), dtype=obs.dtype)), axis=-1)

    def get_absorbing_state(self):
        """Return the canonical absorbing observation with its indicator set."""
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        obs[-1] = self._indicator
        return obs

    @property
    def _max_episode_steps(self):
        """Compatibility for old experiment code; new code uses env.spec."""
        return self.spec.max_episode_steps if self.spec is not None else None
