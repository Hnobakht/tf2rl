"""Project-local pixel observations; no patched Gym pixel wrapper is required."""
import gymnasium as gym
import numpy as np

from tf2rl.envs.frame_stack_wrapper import FrameStack


class Obs2RawPixel(gym.ObservationWrapper):
    """Reverse image channels and optionally convert observations to weighted grayscale."""
    def __init__(self, env, is_gray=True):
        """Configure channel conversion and the resulting byte-valued observation space."""
        super().__init__(env)
        self._is_gray = is_gray
        shape = env.observation_space.shape[:-1] + ((1,) if is_gray else (3,))
        self.observation_space = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observation(self, image):
        # Retain the original channel reversal and weighted grayscale formula.
        """Reverse channels, optionally apply weighted grayscale, and return a byte image."""
        image = image[:, :, ::-1]
        if self._is_gray:
            image = (image[:, :, 0] * .299 + image[:, :, 1] * .587 + image[:, :, 2] * .114)[..., None]
        return image.astype(np.uint8)


class PixelObservation(gym.Wrapper):
    """Expose resized RGB renderings as Gymnasium observations."""
    def __init__(self, env, width=48, height=48):
        """Require RGB-array rendering and configure the output image size."""
        super().__init__(env)
        if env.render_mode != 'rgb_array':
            raise ValueError("Create the environment with render_mode='rgb_array'")
        self.width, self.height = width, height
        self.observation_space = gym.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)

    def _pixels(self):
        """Render the environment and resize the image to the observation dimensions."""
        import cv2
        return cv2.resize(self.env.render(), (self.width, self.height))

    def reset(self, **kwargs):
        """Reset the environment and return its rendered pixel observation and info."""
        _, info = self.env.reset(**kwargs)
        return self._pixels(), info

    def step(self, action):
        """Step the environment and replace the state observation with rendered pixels."""
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._pixels(), reward, terminated, truncated, info


def ControlPixelWrapper(env, n_stack=1, width=48, height=48, is_gray=True):
    """Replace observations with resized pixels, optionally grayscale and frame-stacked."""
    env = Obs2RawPixel(PixelObservation(env, width, height), is_gray)
    return FrameStack(env, n_stack) if n_stack > 1 else env
