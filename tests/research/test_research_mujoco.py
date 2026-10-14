import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import tf2rl


@pytest.mark.optional
@pytest.mark.parametrize('name', [
    'pointmaze-left-v0', 'pointmaze-right-v0', 'hopper-fat-v0', 'hopper-friction-v0',
    'hopper-longleg-v0', 'halfcheetah-disabled-v0', 'halfcheetah-longleg-v0',
    'halfcheetah-longbody-v0', 'walker2d-disabled-v0', 'walker2d-longleg-v0',
    'walker2d-fat-v0', 'ant-disabled-v0', 'ant-disabled2-v0', 'ant-longlegs-v0',
    'ant-full-disabled-v0', 'reacher-simple-v0', 'antrllab-v0', 'antrllab-disabled-v0'])
def test_custom_tasks_native_mujoco(name):
    pytest.importorskip('mujoco')
    env = gym.make(name)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, _ = env.reset(seed=5)
        result = env.step(np.zeros(env.action_space.shape))
        assert result[0].shape == obs.shape and np.isfinite(result[0]).all()
        assert np.isfinite(result[1]) and len(result) == 5
    finally:
        env.close()
