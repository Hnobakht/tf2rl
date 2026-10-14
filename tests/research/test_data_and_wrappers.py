import random

import gymnasium as gym
import numpy as np
import pytest

from tf2rl.experiments.expert_sampling import ExpertSampler
from tf2rl.envs.frame_stack_wrapper import FrameStack
from tf2rl.experiments.demonstrations import get_filenames


class Replay:
    def __init__(self):
        self.calls = []

    def sample(self, size):
        assert size > 0
        self.calls.append(size)
        return dict(obs=-np.ones((size, 2), np.float32), act=-np.ones((size, 1), np.float32))


def data():
    return dict(obs=np.column_stack((np.arange(10), np.arange(10))),
                act=np.arange(10).reshape(-1, 1))


def test_expert_sampling_without_replacement_and_noise_alignment():
    random.seed(0)
    sampler = ExpertSampler(data(), 8, .25)
    replay = Replay()
    result = sampler.sample(replay)
    assert replay.calls == [2]
    assert len(np.unique(result['obs'][:6, 0])) == 6
    np.testing.assert_array_equal(result['obs'][:, 0], result['act'][:, 0])
    np.testing.assert_array_equal(result['obs'][-2:], -np.ones((2, 2)))


def test_zero_noise_never_samples_zero_replay():
    replay = Replay()
    result = ExpertSampler(data(), 8).sample(replay)
    assert not replay.calls and len(result['obs']) == 8


def test_epoch_penalty_uses_warmup_replay_with_same_noise_ratio():
    regular, warmup = Replay(), Replay()
    result = ExpertSampler(data(), 8, .25).sample(regular, warmup)
    assert warmup.calls == [6] and regular.calls == [2]
    assert np.all(result['obs'] == -1)


def test_expert_sampling_validation():
    with pytest.raises(ValueError):
        ExpertSampler(data(), 11)
    with pytest.raises(ValueError):
        ExpertSampler(data(), 8, 1)
    invalid = data()
    invalid['act'] = invalid['act'][:-1]
    with pytest.raises(ValueError):
        ExpertSampler(invalid, 4)


def test_latest_trajectory_selection_numeric_not_lexical(tmp_path):
    for name in ['step_9_epi_0_return_1.pkl', 'step_10_epi_1_return_-1.0.pkl',
                 'step_10_epi_0_return_2.0.pkl', 'notes.txt']:
        (tmp_path / name).touch()
    assert [p.split('/')[-1] for p in get_filenames(tmp_path, 2)] == [
        'step_10_epi_1_return_-1.0.pkl', 'step_10_epi_0_return_2.0.pkl']


class ImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (2, 3, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.ones((2, 3, 3), np.uint8), {'seed': seed}

    def step(self, action):
        return np.full((2, 3, 3), 2, np.uint8), 0., False, True, {}


def test_frame_stack_channels_and_native_flags():
    env = FrameStack(ImageEnv(), 2)
    obs, info = env.reset(seed=7)
    assert obs.shape == (2, 3, 6) and info == {'seed': 7}
    obs, reward, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    np.testing.assert_array_equal(obs[..., :3], 1)
    np.testing.assert_array_equal(obs[..., 3:], 2)
    assert env.observation_space.contains(obs)


def test_learned_reward_tracks_previous_observation_and_flags():
    from tf2rl.envs.learned_reward import LearnedRewardWrapper

    class Reward:
        def inference(self, obs, action, next_obs):
            np.testing.assert_array_equal(obs, 1)
            np.testing.assert_array_equal(next_obs, 2)
            return [[5.]]

    env = LearnedRewardWrapper(ImageEnv(), Reward())
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 5 and info['environment_reward'] == 0
    assert truncated and not terminated
