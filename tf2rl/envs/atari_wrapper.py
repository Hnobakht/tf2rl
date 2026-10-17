"""
The MIT License

Copyright (c) 2017 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
from collections import deque

import numpy as np
import cv2
from gymnasium import spaces
import gymnasium as gym

os.environ.setdefault('PATH', '')
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    """Randomize initial Atari states with a sequence of no-op actions."""
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        """Forward the action to the wrapped environment."""
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    """Start Atari games that require a FIRE action after reset."""
    def __init__(self, env):
        """
        Take action on reset for environments that are fixed until firing.
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        """Reset and issue the actions needed to start the Atari game."""
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        """Forward the action to the wrapped environment."""
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    """Treat loss of an Atari life as an episode boundary during training."""
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """Step the game and mark loss of life as an episode boundary."""
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat actions and max-pool the last two Atari frames."""
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """
        Repeat action, sum reward, and max over last observations.
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        """Reset the wrapped environment and return its initial observation."""
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to their sign while preserving the environment interface."""
    def __init__(self, env):
        """Wrap the environment and configure its observation and episode handling."""
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """Resize Atari images and optionally convert them to grayscale."""
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(
            original_space.shape) == 3

    def observation(self, obs):
        """Resize the observation and optionally convert it to grayscale."""
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """Convert raw Atari frames to 84-by-84 grayscale observations."""
    def __init__(self, env=None):
        """Declare the byte-valued 84-by-84 grayscale observation space."""
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """Convert the raw image to an 84-by-84 grayscale observation."""
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        """Convert an Atari image to the 84-by-84 grayscale observation format."""
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(
            img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack recent image observations along the channel axis."""
    def __init__(self, env, k):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See also baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        """Reset the environment and fill the history with copies of the initial frame."""
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        """Step the environment and append the new frame to the observation history."""
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        """Return the current observations as a lazy frame stack."""
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    """Convert byte image observations to float32 values scaled by 1/255."""
    def __init__(self, env):
        """Declare the floating-point image observation space for the wrapped environment."""
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        """Convert a byte image to float32 values in the range zero to one."""
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    """Delay frame concatenation so replay entries can share stored image arrays."""
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        """Concatenate frames on first access and cache the resulting array."""
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        """Materialize the stacked frames as a NumPy array with the requested dtype."""
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        """Return the length of the materialized frame stack."""
        return len(self._force())

    def __getitem__(self, i):
        """Return the requested slice of the materialized frame stack."""
        return self._force()[..., i]


class NdarrayFrames(gym.Wrapper):
    """Return stacked Atari frames as ordinary NumPy arrays."""
    def __init__(self, env):
        """Wrap a frame-stacked environment to materialize lazy observations as arrays."""
        gym.Wrapper.__init__(self, env)

    def reset(self):
        """Reset the environment and materialize its stacked observation."""
        obs = self.env.reset()
        assert isinstance(obs, LazyFrames)
        return np.array(obs)

    def step(self, action):
        """Step the wrapped environment and materialize the stacked observations."""
        next_obs, rew, done, env_info = self.env.step(action)
        assert isinstance(next_obs, LazyFrames)
        return np.array(next_obs), rew, done, env_info


def make_atari(env_id, max_episode_steps=None):
    """Create an Atari environment with no-op reset and action-repeat preprocessing."""
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """
    Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


# forked from https://github.com/Shmuma/ptan/blob/master/ptan/common/wrappers.py
def wrap_dqn(env, stack_frames=4, episodic_life=True,
             reward_clipping=True, wrap_ndarray=False):
    """
    Apply a common set of wrappers for Atari games.
    """
    assert 'NoFrameskip' in env.spec.id
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, stack_frames)
    if reward_clipping:
        env = ClipRewardEnv(env)
    if wrap_ndarray:
        env = NdarrayFrames(env)
    return env
