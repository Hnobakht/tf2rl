from collections import deque

import gymnasium as gym
import numpy as np


class FrameStack(gym.Wrapper):
    """Concatenate pixel channels, retaining the research model's input ordering."""
    def __init__(self, env, k, obs_shape=None, channel_first=False):
        """Configure the frame history length and stacked observation space."""
        super().__init__(env)
        if k < 1:
            raise ValueError("k must be positive")
        self._k = k
        self._channel_first = channel_first
        self._axis = 0 if channel_first else -1
        self._frames = deque(maxlen=k)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space.low] * k, axis=self._axis),
            high=np.concatenate([env.observation_space.high] * k, axis=self._axis),
            dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        """Reset the environment and fill the history with copies of the initial frame."""
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        self._frames.extend([obs] * self._k)
        return self._get_obs(), info

    def step(self, action):
        """Step the environment and append the new frame to the observation history."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Concatenate the current frame history along the configured channel axis."""
        return np.concatenate(list(self._frames), axis=self._axis)
