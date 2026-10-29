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
        """Validate vector observations and a positive absorbing indicator, then extend the observation
        space.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            indicator (float): Positive value appended to the canonical absorbing observation; ordinary
                observations receive zero.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: AbsorbingWrapper requires a one-dimensional Box observation space. indicator
                must be finite and positive.
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
        """Append the non-absorbing indicator to the observation.

        Args:
            observation (array_like): Observation returned by the wrapped environment.

        Returns:
            np.ndarray: Original observation with a zero absorbing-state indicator appended.
        """
        return self.get_non_absorbing_state(observation)

    def get_non_absorbing_state(self, obs):
        """Append the non-absorbing indicator to an ordinary observation.

        Args:
            obs (array_like): Current observation, or batch of observations with a leading batch axis.

        Returns:
            np.ndarray or tf.Tensor: Input observations with a zero final indicator coordinate,
                preserving all leading dimensions.
        """
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        if obs.shape[-1:] != self.env.observation_space.shape:
            raise ValueError(f"Expected raw observation width {self.env.observation_space.shape}, got {obs.shape}")
        return np.concatenate((obs, np.zeros(obs.shape[:-1] + (1,), dtype=obs.dtype)), axis=-1)

    def get_absorbing_state(self):
        """Return the canonical absorbing observation with its indicator set.

        Returns:
            np.ndarray: One canonical absorbing observation, zero except for its configured final
                indicator coordinate.
        """
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        obs[-1] = self._indicator
        return obs

    @property
    def _max_episode_steps(self):
        """Compatibility for old experiment code; new code uses env.spec.

        Returns:
            int or None: Episode horizon supplied by the wrapped environment specification or legacy
                time-limit attribute.
        """
        return self.spec.max_episode_steps if self.spec is not None else None


class VectorAbsorbingWrapper(gym.vector.VectorWrapper):
    """Append indicators to vector observations and pre-autoreset final observations.

    No fake environment steps are produced. DAC replay/export code inserts the
    absorbing transition and self-loop, preserving physical termination flags.
    """

    def __init__(self, env, indicator=0.1):
        """Initialize VectorAbsorbingWrapper and its configuration.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            indicator (float): Positive value appended to the canonical absorbing observation; ordinary
                observations receive zero.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: Vector absorption requires vector Box observations. indicator must be finite and
                positive.
        """
        from gymnasium.vector.utils import batch_space
        super().__init__(env)
        space = env.single_observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            raise ValueError("Vector absorption requires vector Box observations")
        if not np.isfinite(indicator) or indicator <= 0:
            raise ValueError("indicator must be finite and positive")
        self._indicator = indicator
        self.single_observation_space = gym.spaces.Box(
            low=np.append(np.minimum(space.low, 0), 0).astype(np.float32),
            high=np.append(np.maximum(space.high, 0), indicator).astype(np.float32),
            dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)

    def get_non_absorbing_state(self, obs):
        """Augment a batch or final observation on its current device.

        Args:
            obs (array_like): Current observation, or batch of observations with a leading batch axis.

        Returns:
            np.ndarray or tf.Tensor: Input observations with a zero final indicator coordinate,
                preserving all leading dimensions.
        """
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32, copy=False)
            return np.concatenate((obs, np.zeros(obs.shape[:-1] + (1,), obs.dtype)), axis=-1)
        import tensorflow as tf
        with tf.device(obs.device):
            return tf.concat((obs, tf.zeros_like(obs[..., :1])), axis=-1)

    def get_absorbing_state(self):
        """Return one canonical absorbing observation for replay insertion.

        Returns:
            np.ndarray: One canonical absorbing observation, zero except for its configured final
                indicator coordinate.
        """
        obs = np.zeros(self.single_observation_space.shape, np.float32)
        obs[-1] = self._indicator
        return obs

    def reset(self, **kwargs):
        """Reset physical worlds and append their non-absorbing indicators.

        Args:
            **kwargs (dict): Reset options forwarded to the wrapped environment, such as seed and
                options.

        Returns:
            tuple: Initial wrapped observation and reset info, following the Gymnasium reset API.
        """
        obs, info = self.env.reset(**kwargs)
        return self.get_non_absorbing_state(obs), info

    def step(self, actions):
        """Augment ordinary and final observations without altering boundary flags.

        Args:
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.

        Returns:
            tuple: Gymnasium (observation, reward, terminated, truncated, info) result with the wrapper
                observation/reward transformation.
        """
        obs, reward, terminated, truncated, info = self.env.step(actions)
        info = dict(info)
        if "final_obs" in info:
            final = info["final_obs"]
            if isinstance(final, np.ndarray) and final.dtype == object:
                final = final.copy()
                for index in np.flatnonzero(info["_final_obs"]):
                    final[index] = self.get_non_absorbing_state(final[index])
                info["final_obs"] = final
            else:
                info["final_obs"] = self.get_non_absorbing_state(final)
        return self.get_non_absorbing_state(obs), reward, terminated, truncated, info

    def check_overflow(self):
        """Forward optional simulator capacity checks.

        Returns:
            None.
        """
        if hasattr(self.env, "check_overflow"):
            self.env.check_overflow()
