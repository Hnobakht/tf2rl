import threading

import numpy as np
import tensorflow as tf


class MultiThreadEnv(object):
    """
    This contains multiple environments.
    When step() is called, all of them forward one-step.

    This serve tensorflow operators to manipulate multiple environments.
    """

    def __init__(self, env_fn, batch_size, thread_pool=4, max_episode_steps=1000):
        """

        Args:
            env_fn: function
                Function to make an environment
            batch_size: int
                Batch size
            thread_pool: int
                Thread pool size
            max_episode_steps: int
                Maximum step of an episode
        """
        assert batch_size % thread_pool == 0

        self.batch_size = batch_size
        self.thread_pool = thread_pool
        self.batch_thread = batch_size // thread_pool
        self.envs = [env_fn() for _ in range(batch_size)]

        # collects environment information
        sample_env = env_fn()
        sample_obs, _ = sample_env.reset()
        self._sample_env = sample_env
        self.observation_shape = sample_obs.shape
        # episode time limit
        self.max_episode_steps = max_episode_steps
        if hasattr(sample_env.spec, "max_episode_steps"):
            self.max_episode_steps = sample_env.spec.max_episode_steps
            print("Use max steps of env {} instead of specified value {}".format(
                sample_env.spec.max_episode_steps, max_episode_steps))

        self.list_obs = [None] * self.batch_size
        self.list_rewards = [None] * self.batch_size
        self.list_done = [None] * self.batch_size
        self.list_boundary = [False] * self.batch_size
        self.list_steps = [0] * self.batch_size

        self.py_reset()

    @property
    def original_env(self):
        """Return the sample environment used to expose spaces and metadata."""
        return self._sample_env

    def step(self, actions, name=None):
        """

        Args:
            actions: tf.Tensor
                Actions whose shape is float32[batch_size, dim_action]
            name: str
                Operator name

        Returns:
            obs: tf.Tensor
                [batch_size, dim_obs]
            reward: tf.Tensor
                [batch_size]
            done: tf.Tensor
                [batch_size]
            env_info: None
        """
        assert isinstance(actions, tf.Tensor)
        # with tf.variable_scope(name, default_name="MultiStep"):
        obs, reward, done = tf.py_function(
            func=self.py_step,
            inp=[actions],
            Tout=[tf.float32, tf.float32, tf.float32],
            name="py_step")
        obs.set_shape((self.batch_size,) + self.observation_shape)
        reward.set_shape((self.batch_size,))
        done.set_shape((self.batch_size,))

        return obs, reward, done, None

    def py_step(self, actions):
        """

        Args:
            actions: np.array
                Actions whose shape is [batch_size, dim_action]

        Returns:
            obs: np.array
            reward: np.array
            done: np.array
        """
        def _process(offset):
            """Step this thread's environments and record observations, rewards, and episode
            boundaries.
            """
            for idx_env in range(offset, offset+self.batch_thread):
                new_obs, reward, terminated, truncated, _ = self.envs[idx_env].step(
                    actions[idx_env].numpy())
                self.list_obs[idx_env] = new_obs
                self.list_rewards[idx_env] = reward
                self.list_done[idx_env] = terminated
                self.list_steps[idx_env] += 1
                self.list_boundary[idx_env] = terminated or truncated

        threads = []
        for i in range(self.thread_pool):
            thread = threading.Thread(
                target=_process, args=[i*self.batch_thread])
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        obs = np.stack(self.list_obs, axis=0)
        reward = np.stack(self.list_rewards, axis=0).astype(np.float32)
        done = np.stack(self.list_done, axis=0).astype(np.float32)

        # TODO reset from multiple threads
        for i in range(self.batch_size):
            if self.list_boundary[i] or self.list_steps[i] >= self.max_episode_steps:
                self.list_obs[i], _ = self.envs[i].reset()
                self.list_steps[i] = 0

        return obs, reward, done

    def py_observation(self):
        """Return the latest observations from all environments as a float32 batch."""
        obs = np.stack(self.list_obs, axis=0).astype(np.float32)
        return obs

    def py_reset(self):
        """Reset all environments and return the stacked initial observations."""
        for idx_env, env in enumerate(self.envs):
            obs, _ = env.reset()
            # TODO: Allow flexible data type
            self.list_obs[idx_env] = obs.astype(np.float32)

        return np.stack(self.list_obs, axis=0)

    @property
    def max_action(self):
        """Return the first action coordinate's upper bound as a scalar."""
        return float(self._sample_env.action_space.high[0])

    @property
    def min_action(self):
        """Return the first action coordinate's lower bound as a scalar."""
        return float(self._sample_env.action_space.low[0])

    @property
    def state_dim(self):
        """Return the observation-vector dimension of the sample environment."""
        return self._sample_env.observation_space.shape[0]
