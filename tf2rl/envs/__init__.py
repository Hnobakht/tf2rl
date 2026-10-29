"""Environment utilities; MuJoCo tasks are loaded only when requested."""

import importlib

_ENV_MODULES = {'PointMazeEnv': 'tf2rl.envs.tasks.maze', 'HopperEnvFat': 'tf2rl.envs.tasks.hopper', 'HopperEnvFraction': 'tf2rl.envs.tasks.hopper', 'HopperEnvLongleg': 'tf2rl.envs.tasks.hopper', 'ReacherSimpleEnv': 'tf2rl.envs.tasks.reacher', 'AntEnvDisabled': 'tf2rl.envs.tasks.ant', 'AntEnvDisabled2': 'tf2rl.envs.tasks.ant', 'AntEnvLonglegs': 'tf2rl.envs.tasks.ant', 'AntEnvFullDisabled': 'tf2rl.envs.tasks.ant', 'AntEnvRLLAB': 'tf2rl.envs.tasks.ant_rllab', 'HalfCheetahLongLegEnv': 'tf2rl.envs.tasks.halfcheetah', 'HalfCheetahLongBodyEnv': 'tf2rl.envs.tasks.halfcheetah', 'HalfCheetahDisabledEnv': 'tf2rl.envs.tasks.halfcheetah', 'HalfCheetahPlot': 'tf2rl.envs.tasks.halfcheetah', 'Walker2dDisabledEnv': 'tf2rl.envs.tasks.walker2d', 'Walker2dLongLegEnv': 'tf2rl.envs.tasks.walker2d', 'Walker2dFatEnv': 'tf2rl.envs.tasks.walker2d'}

def __getattr__(name):
    """Lazily import supported environment wrappers when their names are requested."""
    if name in _ENV_MODULES:
        return getattr(importlib.import_module(_ENV_MODULES[name]), name)
    raise AttributeError(name)
