"""Registration of the original embodiment-mismatch experiment IDs."""
from gymnasium.envs.registration import register

register(
    id='pointmaze-left-v0',
    entry_point='tf2rl.envs.maze_rllab_env:PointMazeEnv',
    kwargs={'direction': 0},
    max_episode_steps=100,
)
register(
    id='pointmaze-right-v0',
    entry_point='tf2rl.envs.maze_rllab_env:PointMazeEnv',
    kwargs={'direction': 1},
    max_episode_steps=100,
)
register(
    id='pointmaze-left-long-v0',
    entry_point='tf2rl.envs.maze_rllab_env:PointMazeEnv',
    kwargs={'direction': 0, 'episode_length': 500},
    max_episode_steps=500,
)
register(
    id='pointmaze-right-long-v0',
    entry_point='tf2rl.envs.maze_rllab_env:PointMazeEnv',
    kwargs={'direction': 1, 'episode_length': 500},
    max_episode_steps=500,
)


register(
    id='hopper-fat-v0',
    entry_point='tf2rl.envs.hopper_emb:HopperEnvFat',
    max_episode_steps=1000,
)
register(
    id='hopper-friction-v0',
    entry_point='tf2rl.envs.hopper_emb:HopperEnvFraction',
    max_episode_steps=1000,
)
register(
    id='hopper-longleg-v0',
    entry_point='tf2rl.envs.hopper_emb:HopperEnvLongleg',
    max_episode_steps=1000,
)


register(
    id='reacher-simple-v0',
    entry_point='tf2rl.envs.reacher_simple_obs:ReacherSimpleEnv',
    max_episode_steps=50,
)
register(
    id='reacher-simple-long-v0',
    entry_point='tf2rl.envs.reacher_simple_obs:ReacherSimpleEnv',
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
    entry_point='tf2rl.envs.ant_emb:AntEnvDisabled',
    max_episode_steps=1000,
)
register(
    id='ant-disabled2-v0',
    entry_point='tf2rl.envs.ant_emb:AntEnvDisabled2',
    max_episode_steps=1000,
)
register(
    id='ant-longlegs-v0',
    entry_point='tf2rl.envs.ant_emb:AntEnvLonglegs',
    max_episode_steps=1000,
)
register(
    id='ant-full-disabled-v0',
    entry_point='tf2rl.envs.ant_emb:AntEnvFullDisabled',
    max_episode_steps=1000,
)
register(
    id='ant-full-disabled-plot-v0',
    entry_point='tf2rl.envs.ant_emb:AntEnvFullDisabled',
    kwargs={'ploting': True},
    max_episode_steps=1000,
)
register(
    id='ant-plot-v0',
    entry_point='tf2rl.envs.ant_emb:AntEnvPlot',
    kwargs={'ploting': True},
    max_episode_steps=1000,
)


register(
    id='antrllab-rotation-v0',
    entry_point='tf2rl.envs.ant_rllab_env:AntEnvRLLAB',
    kwargs={'disabled': False},
    max_episode_steps=1000,
)
register(
    id='antrllab-rotation-disabled-v0',
    entry_point='tf2rl.envs.ant_rllab_env:AntEnvRLLAB',
    kwargs={'disabled': True},
    max_episode_steps=1000,
)
register(
    id='antrllab-v0',
    entry_point='tf2rl.envs.ant_rllab_env:AntEnvRLLAB',
    kwargs={'disabled': False, 'rotation': False},
    max_episode_steps=1000,
)
register(
    id='antrllab-disabled-v0',
    entry_point='tf2rl.envs.ant_rllab_env:AntEnvRLLAB',
    kwargs={'disabled': True, 'rotation': False},
    max_episode_steps=1000,
)


register(
    id='halfcheetah-longleg-v0',
    entry_point='tf2rl.envs.halfcheetah_emb:HalfCheetahLongLegEnv',
    max_episode_steps=1000,
)
register(
    id='halfcheetah-longbody-v0',
    entry_point='tf2rl.envs.halfcheetah_emb:HalfCheetahLongBodyEnv',
    max_episode_steps=1000,
)
register(
    id='halfcheetah-disabled-v0',
    entry_point='tf2rl.envs.halfcheetah_emb:HalfCheetahDisabledEnv',
    max_episode_steps=1000,
)
register(
    id='halfcheetah-plot-v0',
    entry_point='tf2rl.envs.halfcheetah_emb:HalfCheetahPlot',
    max_episode_steps=1000,
)

register(
    id='walker2d-disabled-v0',
    entry_point='tf2rl.envs.walker2d_emb:Walker2dDisabledEnv',
    max_episode_steps=1000,
)
register(
    id='walker2d-longleg-v0',
    entry_point='tf2rl.envs.walker2d_emb:Walker2dLongLegEnv',
    max_episode_steps=1000,
)
register(
    id='walker2d-fat-v0',
    entry_point='tf2rl.envs.walker2d_emb:Walker2dFatEnv',
    max_episode_steps=1000,
)


register(
    id='pendulum-long-v0',
    entry_point='gymnasium.envs.classic_control:PendulumEnv',
    max_episode_steps=1000,
)
