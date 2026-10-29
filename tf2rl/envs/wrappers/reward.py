"""Evaluate a learned AIRL/GAIL reward using the native Gymnasium step API."""
import gymnasium as gym
import numpy as np


class LearnedRewardWrapper(gym.Wrapper):
    """Replace environment rewards with learned rewards while retaining originals in info."""
    def __init__(self, env, irl):
        """Attach the learned reward model and clear the cached previous observation."""
        super().__init__(env)
        self.irl = irl
        self._observation = None

    def reset(self, **kwargs):
        """Reset the environment and cache the initial observation for reward inference."""
        observation, info = self.env.reset(**kwargs)
        self._observation = observation.copy()
        return observation, info

    def step(self, action):
        """Step with a learned reward and store the original reward in info; require reset first."""
        if self._observation is None:
            raise RuntimeError("Call reset before stepping a learned-reward environment")
        observation, reward, terminated, truncated, info = self.env.step(action)
        learned = self.irl.inference(self._observation, action, observation)
        info = dict(info, environment_reward=reward)
        self._observation = observation.copy()
        return observation, float(np.asarray(learned).item()), terminated, truncated, info


import numpy as np
import gymnasium as gym
import tensorflow as tf
from tf2rl.algos.imitation.gail import Discriminator
from tf2rl.algos.imitation.gail_eqn import Discriminator_EQL
from tf2rl.algos.imitation.airl import StateModel, StateActionModel
from tf2rl.algos.imitation.airl_eqn import StateModel_EQL, StateActionModel_EQL
from tf2rl.algos.imitation.gaifo import Discriminator as State_only_discriminator
from tf2rl.algos.imitation.gaifo_eqn import Discriminator_EQL as State_only_discriminator_EQL



class WeightAtolApply(gym.Wrapper):
    """Suppress small restored reward-network weights using an absolute or relative threshold."""
    def __init__(self, env, atol=0.01, drop = None, is_show = False, is_figure = False):
        """Suppress restored reward weights using exactly one of atol or relative drop."""
        if atol is None and drop is None:
            raise AttributeError("atol and drop cant be both NONE value")
        elif atol is not None and drop is not None:
            raise AttributeError("atol and drop cant be both selected")

        super().__init__(env)
        assert hasattr(self,"disc") , "Use this wrapper after one of the *ExternalRewad wrappers."

        weights = self.env.disc.get_weights()

        if not drop:
            weights = [np.where(np.abs(i) > atol, i, i*1e-7) for i in weights]
        else:
            new_weights = []
            for index in range(len(weights) - 2):
                tt = weights[index].std() * drop
                new_weights.append(np.array([np.where(np.abs(i) > tt, i, i*1e-7) for i in weights[index]]))
            new_weights.append(weights[-2])
            new_weights.append(weights[-1])
            weights = new_weights
        if is_show:
            print("Weights of the reward function are:")
            print(weights)
        self.env.disc.set_weights(weights)

    def step(self, action, state=None):
        """Forward the action and optional previous state to the wrapped reward environment."""
        return self.env.step(action, state)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access


class GailExternalReward(gym.Wrapper):
    """Legacy environment wrapper using rewards from a saved GAIL discriminator."""
    def __init__(self, env, path, units, gpu=0, enable_sn=False, dac_reward = False):
        """Wrap the environment and restore the reward model from the latest checkpoint in path."""
        super().__init__(env)
        self.env = env
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.disc = Discriminator(state_shape=env.observation_space.shape,
                                  action_dim=env.action_space.high.size,
                                  units=units,
                                  enable_sn=enable_sn)

        checkpoint = tf.train.Checkpoint(irl_weights=self.disc.weights)
        last_checkpoint = tf.train.latest_checkpoint(path)
        checkpoint.restore(last_checkpoint)
        print("[DEBUG] Weights of the reward network are loaded successfully.")

    def step(self, action, state=None):
        """Step the legacy environment using the previous state to compute a learned reward."""
        assert state is not None, "[ERROR] The state shoud has value to calculate reward. [GailExternalReward]"

        next_state, env_rew, done, info = self.env.step(action)
        info['reward'] = env_rew

        if state.ndim == action.ndim == 1:
            state = np.expand_dims(state, axis=0)
            action = np.expand_dims(action, axis=0)
        inputs = np.concatenate((state, action), axis=1)
        reward = self._step_body(inputs)
        return next_state, np.squeeze(reward), done, info

    @tf.function
    def _step_body(self, inputs):
        """
        Evaluate the restored reward model on a batch using this wrapper's reward sign convention.
        """
        print("[DEBUG] Initializing _step_body [GailEQLExternalReward]")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access


class GailEQLExternalReward(gym.Wrapper):
    """Legacy environment wrapper using rewards from a saved EQN-GAIL discriminator."""
    def __init__(self, env, path, num_layers, v, exclude, gpu=0, enable_sn=False, dac_reward = False):
        """Wrap the environment and restore the reward model from the latest checkpoint in path."""
        super().__init__(env)
        self.env = env
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.disc = Discriminator_EQL(state_shape=env.observation_space.shape,
                                      action_dim=env.action_space.high.size,
                                      v=v,
                                      num_layers=num_layers,
                                      enable_sn=enable_sn,
                                      constraint=None,
                                      name="External_Reward",
                                      lmbda=0,
                                      exclude=exclude,
                                      is_lmbda_dynamic=False,
                                      output_activation=tf.nn.sigmoid,
                                      )

        checkpoint = tf.train.Checkpoint(irl_weights=self.disc.weights)
        last_checkpoint = tf.train.latest_checkpoint(path)
        checkpoint.restore(last_checkpoint)
        print("[DEBUG] Weights of the reward network are loaded successfully.")

    def step(self, action, state=None):
        """Step the legacy environment using the previous state to compute a learned reward."""
        assert state is not None, "[ERROR] The state shoud has value to calculate reward. [GailEQLExternalReward]"

        next_state, env_rew, done, info = self.env.step(action)
        info['reward'] = env_rew

        if state.ndim == action.ndim == 1:
            state = np.expand_dims(state, axis=0)
            action = np.expand_dims(action, axis=0)
        inputs = np.concatenate((state, action), axis=1)
        reward = self._step_body(inputs)
        return next_state, np.squeeze(reward), done, info

    @tf.function
    def _step_body(self, inputs):
        """
        Evaluate the restored reward model on a batch using this wrapper's reward sign convention.
        """
        print("[DEBUG] Initializing _step_body [GailEQLExternalReward]")
        with tf.device(self.device):
            return -self.disc.compute_reward(inputs)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access

class AirlEQLExternalReward(gym.Wrapper):
    """Legacy environment wrapper using rewards from a saved EQN-AIRL model."""
    def __init__(self, env, path, num_layers, v, exclude, units, state_only = False, gpu=0, enable_sn=False, dac_reward = False):
        """Wrap the environment and restore the reward model from the latest checkpoint in path."""
        super().__init__(env)
        self.env = env
        self._is_state_only = state_only
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        if state_only:
            self.disc = StateModel_EQL(
                state_shape=env.observation_space.shape,
                v=v,
                num_layers=num_layers,
                enable_sn=enable_sn,
                constraint=None,
                name="External_reward_net",
                lmbda=0,
                exclude=exclude,
                is_lmbda_dynamic=False,
                output_activation=None)
        else:
            self.disc = StateActionModel_EQL(
                state_shape=env.observation_space.shape,
                action_dim=env.action_space.high.size,
                v=v,
                num_layers=num_layers,
                enable_sn=enable_sn,
                constraint=None,
                name="External_reward_net",
                lmbda=0,
                exclude=exclude,
                is_lmbda_dynamic=0,
                output_activation=None)

        self.val_net = StateModel(
            state_shape=env.observation_space.shape, units=units, enable_sn=enable_sn,
            name="val_net", output_activation="linear")

        checkpoint = tf.train.Checkpoint(irl_weights=self.disc.weights + self.val_net.weights)
        last_checkpoint = tf.train.latest_checkpoint(path)
        checkpoint.restore(last_checkpoint)
        print("[DEBUG] Weights of the reward network are loaded successfully.")

    def step(self, action, state=None):
        """Step the legacy environment using the previous state to compute a learned reward."""
        assert state is not None, "[ERROR] The state shoud has value to calculate reward. [AirlEQLExternalReward]"

        next_state, env_rew, done, info = self.env.step(action)
        info['reward'] = env_rew
        if self._is_state_only:
            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)
            inputs = state
        else:
            if state.ndim == action.ndim == 1:
                state = np.expand_dims(state, axis=0)
                action = np.expand_dims(action, axis=0)
            inputs = np.concatenate((state, action), axis=1)

        reward = self._step_body(inputs)
        return next_state, np.squeeze(reward), done, info

    @tf.function
    def _step_body(self, inputs):
        """
        Evaluate the restored reward model on a batch using this wrapper's reward sign convention.
        """
        print("[DEBUG] Initializing _step_body [AirlEQLExternalReward]")
        with tf.device(self.device):
            return -self.disc(inputs, training=False)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access


class AirlExternalReward(gym.Wrapper):
    """Legacy environment wrapper using rewards from a saved AIRL model."""
    def __init__(self, env, path, units, gpu=0, state_only = False, enable_sn=False, dac_reward = False):
        """Wrap the environment and restore the reward model from the latest checkpoint in path."""
        super().__init__(env)
        self.env = env
        self._is_state_only = state_only
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        if state_only:
            self.disc = StateModel(
                state_shape=env.observation_space.shape,
                units=units,
                name="External_reward_net",
                enable_sn=enable_sn,
                output_activation="linear")
        else:
            self.disc = StateActionModel(
                state_shape=env.observation_space.shape,
                action_dim=env.action_space.high.size,
                units=units,
                name="External_reward_net",
                enable_sn=enable_sn,
                output_activation="linear")

        self.val_net = StateModel(
            state_shape=env.observation_space.shape, units=units, enable_sn=enable_sn,
            name="val_net", output_activation="linear")

        checkpoint = tf.train.Checkpoint(irl_weights=self.disc.weights + self.val_net.weights)
        last_checkpoint = tf.train.latest_checkpoint(path)
        checkpoint.restore(last_checkpoint)
        print("[DEBUG] Weights of the reward network are loaded successfully.")

    def step(self, action, state=None):
        """Step the legacy environment using the previous state to compute a learned reward."""
        assert state is not None, "[ERROR] The state shoud has value to calculate reward. [AirlExternalReward]"

        next_state, env_rew, done, info = self.env.step(action)
        info['reward'] = env_rew
        if self._is_state_only:
            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)
            inputs = state
        else:
            if state.ndim == action.ndim == 1:
                state = np.expand_dims(state, axis=0)
                action = np.expand_dims(action, axis=0)
            inputs = np.concatenate((state, action), axis=1)

        reward = self._step_body(inputs)
        return next_state, np.squeeze(reward), done, info

    @tf.function
    def _step_body(self, inputs):
        """
        Evaluate the restored reward model on a batch using this wrapper's reward sign convention.
        """
        print("[DEBUG] Initializing _step_body [AirlExternalReward]")
        with tf.device(self.device):
            return self.disc(inputs, training=False)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access

class GaifoExternalReward(gym.Wrapper):
    """Legacy environment wrapper using a saved observation-transition discriminator."""
    def __init__(self, env, path, units, gpu=0, enable_sn=False, dac_reward = False):
        """Wrap the environment and restore the reward model from the latest checkpoint in path."""
        super().__init__(env)
        self.env = env
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.disc = State_only_discriminator(state_shape=env.observation_space.shape,
                                             units=units,
                                             enable_sn=enable_sn)

        checkpoint = tf.train.Checkpoint(irl_weights=self.disc.weights)
        last_checkpoint = tf.train.latest_checkpoint(path)
        checkpoint.restore(last_checkpoint)
        print("[DEBUG] Weights of the reward network are loaded successfully.")

    def step(self, action, state=None):
        """Step the legacy environment using the previous state to compute a learned reward."""
        assert state is not None, "[ERROR] The state shoud has value to calculate reward. [GaifoExternalReward]"

        next_state, env_rew, done, info = self.env.step(action)
        info['reward'] = env_rew

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        if next_state.ndim == 1:
            next_state = np.expand_dims(next_state, axis=0)
        inputs = np.concatenate((state, next_state), axis=1)
        reward = self._step_body(inputs)
        return next_state, np.squeeze(reward), done, info

    @tf.function
    def _step_body(self, inputs):
        """
        Evaluate the restored reward model on a batch using this wrapper's reward sign convention.
        """
        print("[DEBUG] Initializing _step_body [GaifoExternalReward]")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access


class GaifoEQLExternalReward(gym.Wrapper):
    """Legacy environment wrapper using a saved equation transition discriminator."""
    def __init__(self, env, path, num_layers, v, exclude, gpu=0, enable_sn=False, dac_reward = False):
        """Wrap the environment and restore the reward model from the latest checkpoint in path."""
        super().__init__(env)
        self.env = env
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.disc = State_only_discriminator_EQL(state_shape=env.observation_space.shape,
                                                  v=v,
                                                  num_layers=num_layers,
                                                  enable_sn=enable_sn,
                                                  constraint=None,
                                                  name="External_Reward",
                                                  lmbda=0,
                                                  exclude=exclude,
                                                  is_lmbda_dynamic=False,
                                                  output_activation=tf.nn.sigmoid,
                                                  )

        checkpoint = tf.train.Checkpoint(irl_weights=self.disc.weights)
        last_checkpoint = tf.train.latest_checkpoint(path)
        checkpoint.restore(last_checkpoint)
        print("[DEBUG] Weights of the reward network are loaded successfully.")

    def step(self, action, state=None):
        """Step the legacy environment using the previous state to compute a learned reward."""
        assert state is not None, "[ERROR] The state shoud has value to calculate reward. [GaifoEQLExternalReward]"

        next_state, env_rew, done, info = self.env.step(action)
        info['reward'] = env_rew

        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        if next_state.ndim == 1:
            next_state = np.expand_dims(next_state, axis=0)
        inputs = np.concatenate((state, next_state), axis=1)
        reward = self._step_body(inputs)
        return next_state, np.squeeze(reward), done, info

    @tf.function
    def _step_body(self, inputs):
        """
        Evaluate the restored reward model on a batch using this wrapper's reward sign convention.
        """
        print("[DEBUG] Initializing _step_body [GaifoEQLExternalReward]")
        with tf.device(self.device):
            return self.disc.compute_reward(inputs)

    @property
    def _max_episode_steps(self):
        """Return the wrapped environment's episode time limit."""
        return self.env._max_episode_steps  # pylint: disable=protected-access
