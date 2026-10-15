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


@pytest.mark.optional
@pytest.mark.parametrize('name,asset', [('hopper-fat-v0', 'hopper_fat.xml'),
                                     ('walker2d-longleg-v0', 'walker2d_long_leg.xml')])
def test_global_asset_geometry_is_preserved_in_world_coordinates(name, asset):
    """Compare compiled geometry to the original global coordinates, not converted XML."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    mujoco = pytest.importorskip('mujoco')
    env = gym.make(name)
    try:
        raw = env.unwrapped
        mujoco.mj_resetData(raw.model, raw.data)
        mujoco.mj_forward(raw.model, raw.data)
        xml = ET.parse(Path(tf2rl.__file__).parent / 'envs' / 'assets' / asset)
        for body in xml.findall('.//worldbody//body'):
            position = [float(token.split('/')[0]) / (float(token.split('/')[1])
                        if '/' in token else 1) for token in body.attrib['pos'].split()]
            np.testing.assert_allclose(raw.data.body(body.attrib['name']).xpos, position, atol=1e-12)
        for geom in xml.findall('.//worldbody//geom'):
            if 'fromto' in geom.attrib:
                endpoints = np.asarray([float(token) for token in geom.attrib['fromto'].split()]).reshape(2, 3)
                np.testing.assert_allclose(raw.data.geom(geom.attrib['name']).xpos,
                                           endpoints.mean(axis=0), atol=1e-12)
    finally:
        env.close()
