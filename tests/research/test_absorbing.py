import joblib
import gymnasium as gym
import numpy as np
import pytest

from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.misc.dac_ops import dac_transitions
from tf2rl.experiments.demonstrations import load_trajectories_with_logps


class TinyEnv(gym.Env):
    spec = gym.envs.registration.EnvSpec("Tiny-v0")
    observation_space = gym.spaces.Box(np.array([1., -5.], np.float32),
                                       np.array([3., 7.], np.float32))
    action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([2., 1.], np.float32), {"reset": True}

    def step(self, action):
        self.steps += 1
        return np.array([2., self.steps], np.float32), 1., self.steps == 2, False, {}


def test_wrapper_api_bounds_and_dtype():
    env = AbsorbingWrapper(TinyEnv(), indicator=.1)
    obs, info = env.reset(seed=4)
    assert info['reset'] and obs.dtype == np.float32
    np.testing.assert_array_equal(obs, [2, 1, 0])
    np.testing.assert_array_equal(env.observation_space.low, [0, -5, 0])
    assert env.observation_space.contains(env.get_absorbing_state())
    assert len(env.step(np.zeros(1, np.float32))) == 5
    with pytest.raises(ValueError):
        env.get_non_absorbing_state(obs)


@pytest.mark.parametrize('terminated,truncated,count,mask', [
    (False, False, 1, 1.), (False, True, 1, 1.),
    (True, False, 2, 0.), (True, True, 2, 0.)])
def test_dac_boundary_flags(terminated, truncated, count, mask):
    env = AbsorbingWrapper(TinyEnv(), .1)
    obs, _ = env.reset()
    rows = dac_transitions(env, obs, np.array([.5]), obs, 3., terminated, truncated, logp=-.2)
    assert len(rows) == count and rows[0]['done'] == mask
    if terminated:
        np.testing.assert_array_equal(rows[0]['next_obs'], env.get_absorbing_state())
        np.testing.assert_array_equal(rows[1]['act'], [0])
        np.testing.assert_array_equal(rows[1]['obs'], rows[1]['next_obs'])
        assert rows[1]['done'] == -1 and rows[1]['rew'] == rows[1]['logp'] == 0
    else:
        np.testing.assert_array_equal(rows[0]['next_obs'], obs)


def test_absorption_can_be_disabled():
    env = AbsorbingWrapper(TinyEnv())
    obs, _ = env.reset()
    rows = dac_transitions(env, obs, np.ones(1), obs, 1, True, absorbing=False)
    assert len(rows) == 1 and rows[0]['done'] == 0
    np.testing.assert_array_equal(rows[0]['next_obs'], obs)


def trajectory(augmented=False):
    obs = np.array([[2., 0.], [2., 1.]], np.float32)
    next_obs = obs + [0., 1.]
    if augmented:
        obs, next_obs = np.pad(obs, ((0, 0), (0, 1))), np.pad(next_obs, ((0, 0), (0, 1)))
    return dict(obs=obs, next_obs=next_obs, act=np.ones((2, 1), np.float32),
                done=np.array([False, True]), logp=np.array([-.2, -.3], np.float32))


@pytest.mark.parametrize('augmented', [False, True])
def test_short_expert_episode_and_idempotent_augmentation(tmp_path, augmented):
    env = AbsorbingWrapper(gym.wrappers.TimeLimit(TinyEnv(), max_episode_steps=10), .1)
    path = trajectory(augmented)
    path.update(terminated=[False, True], truncated=[False, False])
    filename = tmp_path / 'trajectory.pkl'
    joblib.dump(path, filename)
    data = load_trajectories_with_logps([filename], env=env)
    assert data['obses'].shape == (3, 3)
    assert data['logps'].shape == (3, 1)
    np.testing.assert_array_equal(data['not_dones'].ravel(), [1, 0, -1])
    np.testing.assert_array_equal(data['next_obses'][-2], env.get_absorbing_state())


def test_explicit_terminal_at_horizon_and_legacy_inference(tmp_path):
    env = AbsorbingWrapper(gym.wrappers.TimeLimit(TinyEnv(), max_episode_steps=2))
    path = trajectory()
    filename = tmp_path / 'trajectory.pkl'
    joblib.dump(path, filename)
    with pytest.warns(UserWarning, match='Legacy'):
        legacy = load_trajectories_with_logps([filename], env=env)
    assert len(legacy['obses']) == 2
    np.testing.assert_array_equal(legacy['not_dones'].ravel(), [1, 1])
    path.update(terminated=[False, True], truncated=[False, True])
    joblib.dump(path, filename)
    explicit = load_trajectories_with_logps([filename], env=env)
    np.testing.assert_array_equal(explicit['not_dones'].ravel(), [1, 0, -1])


def test_max_steps_applied_before_augmentation(tmp_path):
    env = AbsorbingWrapper(TinyEnv())
    path = trajectory()
    path.update(terminated=[False, True], truncated=[False, False])
    filename = tmp_path / 'trajectory.pkl'
    joblib.dump(path, filename)
    data = load_trajectories_with_logps([filename], max_steps=1, env=env)
    assert data['obses'].shape == (1, 3)
    np.testing.assert_array_equal(data['not_dones'], [[1]])
