"""Project-local pixel observations; no patched Gym pixel wrapper is required."""
import gymnasium as gym
import numpy as np

from tf2rl.envs.wrappers.frame_stack import FrameStack


class Obs2RawPixel(gym.ObservationWrapper):
    """Reverse image channels and optionally convert observations to weighted grayscale."""
    def __init__(self, env, is_gray=True):
        """Configure channel conversion and the resulting byte-valued observation space.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            is_gray (bool): Convert pixel observations to grayscale.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(env)
        self._is_gray = is_gray
        shape = env.observation_space.shape[:-1] + ((1,) if is_gray else (3,))
        self.observation_space = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observation(self, image):
        # Retain the original channel reversal and weighted grayscale formula.
        """Reverse channels, optionally apply weighted grayscale, and return a byte image.

        Args:
            image (np.ndarray): Rendered image to transform.

        Returns:
            np.ndarray: Pixel observation converted to the wrapper image layout, dtype, and grayscale
                convention.
        """
        image = image[:, :, ::-1]
        if self._is_gray:
            image = (image[:, :, 0] * .299 + image[:, :, 1] * .587 + image[:, :, 2] * .114)[..., None]
        return image.astype(np.uint8)


class PixelObservation(gym.Wrapper):
    """Expose resized RGB renderings as Gymnasium observations."""
    def __init__(self, env, width=48, height=48):
        """Require RGB-array rendering and configure the output image size.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            width (int): Output image width in pixels.
            height (int): Output image height in pixels.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: Create the environment with render_mode='rgb_array'.
        """
        super().__init__(env)
        if env.render_mode != 'rgb_array':
            raise ValueError("Create the environment with render_mode='rgb_array'")
        self.width, self.height = width, height
        self.observation_space = gym.spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)

    def _pixels(self):
        """Render the environment and resize the image to the observation dimensions.

        Returns:
            np.ndarray: Resized rendered RGB observation shaped (height, width, channels).
        """
        import cv2
        return cv2.resize(self.env.render(), (self.width, self.height))

    def reset(self, **kwargs):
        """Reset the environment and return its rendered pixel observation and info.

        Args:
            **kwargs (dict): Reset options forwarded to the wrapped environment, such as seed and
                options.

        Returns:
            tuple: Initial wrapped observation and reset info, following the Gymnasium reset API.
        """
        _, info = self.env.reset(**kwargs)
        return self._pixels(), info

    def step(self, action):
        """Step the environment and replace the state observation with rendered pixels.

        Args:
            action (array_like): Single action in the environment action space.

        Returns:
            tuple: Gymnasium (observation, reward, terminated, truncated, info) result with the wrapper
                observation/reward transformation.
        """
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._pixels(), reward, terminated, truncated, info


def ControlPixelWrapper(env, n_stack=1, width=48, height=48, is_gray=True):
    """Replace observations with resized pixels, optionally grayscale and frame-stacked.

    Args:
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.
        n_stack (int): Number of frames stacked by the pixel-wrapper factory.
        width (int): Output image width in pixels.
        height (int): Output image height in pixels.
        is_gray (bool): Convert pixel observations to grayscale.

    Returns:
        gym.Env: Environment with the requested preprocessing and frame-stack wrappers applied.
    """
    env = Obs2RawPixel(PixelObservation(env, width, height), is_gray)
    return FrameStack(env, n_stack) if n_stack > 1 else env
