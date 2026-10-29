"""Registration of the original embodiment-mismatch experiment IDs."""
from gymnasium.envs.registration import register

register(
    id='pointmaze-left-v0',
    entry_point='tf2rl.envs.tasks.maze:PointMazeEnv',
    kwargs={'direction': 0},
    max_episode_steps=100,
)
register(
    id='pointmaze-right-v0',
    entry_point='tf2rl.envs.tasks.maze:PointMazeEnv',
    kwargs={'direction': 1},
    max_episode_steps=100,
)
register(
    id='pointmaze-left-long-v0',
    entry_point='tf2rl.envs.tasks.maze:PointMazeEnv',
    kwargs={'direction': 0, 'episode_length': 500},
    max_episode_steps=500,
)
register(
    id='pointmaze-right-long-v0',
    entry_point='tf2rl.envs.tasks.maze:PointMazeEnv',
    kwargs={'direction': 1, 'episode_length': 500},
    max_episode_steps=500,
)


register(
    id='hopper-fat-v0',
    entry_point='tf2rl.envs.tasks.hopper:HopperEnvFat',
    max_episode_steps=1000,
)
register(
    id='hopper-friction-v0',
    entry_point='tf2rl.envs.tasks.hopper:HopperEnvFraction',
    max_episode_steps=1000,
)
register(
    id='hopper-longleg-v0',
    entry_point='tf2rl.envs.tasks.hopper:HopperEnvLongleg',
    max_episode_steps=1000,
)


register(
    id='reacher-simple-v0',
    entry_point='tf2rl.envs.tasks.reacher:ReacherSimpleEnv',
    max_episode_steps=50,
)
register(
    id='reacher-simple-long-v0',
    entry_point='tf2rl.envs.tasks.reacher:ReacherSimpleEnv',
    max_episode_steps=200,
)
register(
    id='Reacher-long-v2',
    entry_point='gymnasium.envs.mujoco.reacher_v5:ReacherEnv',
    max_episode_steps=200,
    reward_threshold=-3.75,
)


register(
    id='ant-disabled-v0',
    entry_point='tf2rl.envs.tasks.ant:AntEnvDisabled',
    max_episode_steps=1000,
)
register(
    id='ant-disabled2-v0',
    entry_point='tf2rl.envs.tasks.ant:AntEnvDisabled2',
    max_episode_steps=1000,
)
register(
    id='ant-longlegs-v0',
    entry_point='tf2rl.envs.tasks.ant:AntEnvLonglegs',
    max_episode_steps=1000,
)
register(
    id='ant-full-disabled-v0',
    entry_point='tf2rl.envs.tasks.ant:AntEnvFullDisabled',
    max_episode_steps=1000,
)
register(
    id='ant-full-disabled-plot-v0',
    entry_point='tf2rl.envs.tasks.ant:AntEnvFullDisabled',
    kwargs={'ploting': True},
    max_episode_steps=1000,
)
register(
    id='ant-plot-v0',
    entry_point='tf2rl.envs.tasks.ant:AntEnvPlot',
    kwargs={'ploting': True},
    max_episode_steps=1000,
)


register(
    id='antrllab-rotation-v0',
    entry_point='tf2rl.envs.tasks.ant_rllab:AntEnvRLLAB',
    kwargs={'disabled': False},
    max_episode_steps=1000,
)
register(
    id='antrllab-rotation-disabled-v0',
    entry_point='tf2rl.envs.tasks.ant_rllab:AntEnvRLLAB',
    kwargs={'disabled': True},
    max_episode_steps=1000,
)
register(
    id='antrllab-v0',
    entry_point='tf2rl.envs.tasks.ant_rllab:AntEnvRLLAB',
    kwargs={'disabled': False, 'rotation': False},
    max_episode_steps=1000,
)
register(
    id='antrllab-disabled-v0',
    entry_point='tf2rl.envs.tasks.ant_rllab:AntEnvRLLAB',
    kwargs={'disabled': True, 'rotation': False},
    max_episode_steps=1000,
)


register(
    id='halfcheetah-longleg-v0',
    entry_point='tf2rl.envs.tasks.halfcheetah:HalfCheetahLongLegEnv',
    max_episode_steps=1000,
)
register(
    id='halfcheetah-longbody-v0',
    entry_point='tf2rl.envs.tasks.halfcheetah:HalfCheetahLongBodyEnv',
    max_episode_steps=1000,
)
register(
    id='halfcheetah-disabled-v0',
    entry_point='tf2rl.envs.tasks.halfcheetah:HalfCheetahDisabledEnv',
    max_episode_steps=1000,
)
register(
    id='halfcheetah-plot-v0',
    entry_point='tf2rl.envs.tasks.halfcheetah:HalfCheetahPlot',
    max_episode_steps=1000,
)

register(
    id='walker2d-disabled-v0',
    entry_point='tf2rl.envs.tasks.walker2d:Walker2dDisabledEnv',
    max_episode_steps=1000,
)
register(
    id='walker2d-longleg-v0',
    entry_point='tf2rl.envs.tasks.walker2d:Walker2dLongLegEnv',
    max_episode_steps=1000,
)
register(
    id='walker2d-fat-v0',
    entry_point='tf2rl.envs.tasks.walker2d:Walker2dFatEnv',
    max_episode_steps=1000,
)


register(
    id='pendulum-long-v0',
    entry_point='gymnasium.envs.classic_control:PendulumEnv',
    max_episode_steps=1000,
)
