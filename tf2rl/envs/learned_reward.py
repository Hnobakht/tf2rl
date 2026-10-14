"""Evaluate a learned AIRL/GAIL reward using the native Gymnasium step API."""
import gymnasium as gym
import numpy as np


class LearnedRewardWrapper(gym.Wrapper):
    def __init__(self, env, irl):
        super().__init__(env)
        self.irl = irl
        self._observation = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._observation = observation.copy()
        return observation, info

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Call reset before stepping a learned-reward environment")
        observation, reward, terminated, truncated, info = self.env.step(action)
        learned = self.irl.inference(self._observation, action, observation)
        info = dict(info, environment_reward=reward)
        self._observation = observation.copy()
        return observation, float(np.asarray(learned).item()), terminated, truncated, info
