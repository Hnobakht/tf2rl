"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np
import OpenGL.GLUT as gl

from gym import spaces
from gym import ObservationWrapper

class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(self,
                 env,
                 render_shape=None):

        # Creating a dummy GL Windows to bupas the Error: "GLEW initalization error: Missing GL version"
        gl.glutInit()
        gl.glutInitWindowSize(500, 500)
        gl.glutCreateWindow('GLEW Testing')

        super(PixelObservationWrapper, self).__init__(env)

        self.render_shape =render_shape

        pixels = self.render(mode='rgb_array',width=render_shape[0],height=render_shape[1])
        low, high = (0, 255)
        self.observation_space = spaces.Box(shape=pixels.shape, low=low, high=high, dtype=np.uint8)
        self._max_episode_steps = env._max_episode_steps

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        return self.render(mode='rgb_array',width=self.render_shape[0],height=self.render_shape[1])


class PixelObservationWrapper_classic(ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(self,
                 env,
                 render_shape=None):
        raise NotImplementedError

        super(PixelObservationWrapper_classic, self).__init__(env)

        self.render_shape = render_shape

        low, high = (0, 255)
        self.observation_space = spaces.Box(shape=(render_shape[0],render_shape[1],3), low=low, high=high, dtype=np.uint8)
        self._max_episode_steps = env._max_episode_steps

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        img = self.render('rgb_array')#, width=self.render_shape[0], height=self.render_shape[1])
        return img

