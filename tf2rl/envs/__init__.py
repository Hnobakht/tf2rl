"""Environment utilities; MuJoCo tasks are loaded only when requested."""

import importlib

_ENV_MODULES = {'PointMazeEnv': 'tf2rl.envs.maze_rllab_env', 'HopperEnvFat': 'tf2rl.envs.hopper_emb', 'HopperEnvFraction': 'tf2rl.envs.hopper_emb', 'HopperEnvLongleg': 'tf2rl.envs.hopper_emb', 'ReacherSimpleEnv': 'tf2rl.envs.reacher_simple_obs', 'AntEnvDisabled': 'tf2rl.envs.ant_emb', 'AntEnvDisabled2': 'tf2rl.envs.ant_emb', 'AntEnvLonglegs': 'tf2rl.envs.ant_emb', 'AntEnvFullDisabled': 'tf2rl.envs.ant_emb', 'AntEnvRLLAB': 'tf2rl.envs.ant_rllab_env', 'HalfCheetahLongLegEnv': 'tf2rl.envs.halfcheetah_emb', 'HalfCheetahLongBodyEnv': 'tf2rl.envs.halfcheetah_emb', 'HalfCheetahDisabledEnv': 'tf2rl.envs.halfcheetah_emb', 'HalfCheetahPlot': 'tf2rl.envs.halfcheetah_emb', 'Walker2dDisabledEnv': 'tf2rl.envs.walker2d_emb', 'Walker2dLongLegEnv': 'tf2rl.envs.walker2d_emb', 'Walker2dFatEnv': 'tf2rl.envs.walker2d_emb'}

def __getattr__(name):
    if name in _ENV_MODULES:
        return getattr(importlib.import_module(_ENV_MODULES[name]), name)
    raise AttributeError(name)
