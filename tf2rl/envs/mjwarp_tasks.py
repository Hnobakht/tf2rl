"""Task adapters for the reusable MJWarp vector backend.

Physics alone does not define an RL task. Each adapter explicitly preserves its
Gymnasium observation, reward, termination and reset semantics on the GPU.
"""

import numpy as np
import warp as wp

_TASKS = {}


def register_mjwarp_task(env_class, factory):
    """Register a factory taking an unwrapped Gymnasium environment."""
    _TASKS[env_class] = factory


def get_mjwarp_task(env):
    from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
    from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv

    defaults = {HalfCheetahEnv: HalfCheetahTask, InvertedPendulumEnv: InvertedPendulumTask}
    factory = _TASKS.get(type(env), defaults.get(type(env)))
    if factory is None:
        raise NotImplementedError(
            f"No MJWarp task adapter for {type(env).__name__}; register_mjwarp_task or pass task=. "
            "MJWarpSimulator can run other compatible MuJoCo models without task semantics."
        )
    return factory(env)


@wp.kernel
def _joint_observation(
    qpos: wp.array2d[float], qvel: wp.array2d[float], exclude: int, obs: wp.array2d[float]
):
    world = wp.tid()
    for j in range(qpos.shape[1] - exclude):
        obs[world, j] = qpos[world, j + exclude]
    for j in range(qvel.shape[1]):
        obs[world, qpos.shape[1] - exclude + j] = qvel[world, j]


@wp.kernel
def _joint_reset(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    warmstart: wp.array2d[float],
    time: wp.array[float],
    qpos0: wp.array[float],
    mask: wp.array[int],
    seeds: wp.array[int],
    episodes: wp.array[int],
    noise: float,
    normal_velocity: int,
):
    world = wp.tid()
    if mask[world] != 0:
        rng = wp.rand_init(seeds[world], episodes[world])
        for j in range(qpos.shape[1]):
            qpos[world, j] = qpos0[j] + wp.randf(rng, -noise, noise)
        for j in range(qvel.shape[1]):
            velocity = wp.randf(rng, -noise, noise)
            if normal_velocity != 0:
                velocity = noise * wp.randn(rng)
            qvel[world, j] = velocity
            warmstart[world, j] = 0.0
        time[world] = 0.0
        episodes[world] += 1


class JointStateTask:
    """Shared observations/resets for stateless actuators and scalar joints only."""

    def allocate(self, sim, env):
        if sim.host_model.na or np.any(np.asarray(sim.host_model.jnt_type) < 2):
            raise ValueError("JointStateTask requires stateless actuators and slide/hinge joints")
        self.qpos0 = wp.array(env.init_qpos.astype(np.float32))
        self.noise = env._reset_noise_scale

    def observe(self, sim, output):
        wp.launch(
            _joint_observation,
            dim=sim.num_envs,
            inputs=[sim.data.qpos, sim.data.qvel, self.exclude, output],
        )

    def reset(self, sim, mask, seeds, episodes):
        # These tasks have no actuator activation, mocap, history or plugin state.
        # Reset integration state; the next physics step rebuilds derived fields.
        wp.launch(
            _joint_reset,
            dim=sim.num_envs,
            inputs=[
                sim.data.qpos,
                sim.data.qvel,
                sim.data.qacc_warmstart,
                sim.data.time,
                self.qpos0,
                mask,
                seeds,
                episodes,
                self.noise,
                self.normal_velocity,
            ],
        )


@wp.kernel
def _position(qpos: wp.array2d[float], previous: wp.array[float]):
    world = wp.tid()
    previous[world] = qpos[world, 0]


@wp.kernel
def _running_reward(
    qpos: wp.array2d[float],
    ctrl: wp.array2d[float],
    previous: wp.array[float],
    dt: float,
    run_weight: float,
    ctrl_weight: float,
    rewards: wp.array[float],
    terminated: wp.array[int],
):
    world = wp.tid()
    cost = float(0.0)  # noqa: UP018 - Warp requires a dynamic accumulator.
    for j in range(ctrl.shape[1]):
        cost += ctrl[world, j] * ctrl[world, j]
    rewards[world] = run_weight * (qpos[world, 0] - previous[world]) / dt - ctrl_weight * cost
    terminated[world] = 0


class HalfCheetahTask(JointStateTask):
    def __init__(self, env):
        self.exclude = int(env._exclude_current_positions_from_observation)
        self.normal_velocity = 1
        self.dt = env.dt
        self.run_weight = env._forward_reward_weight
        self.ctrl_weight = env._ctrl_cost_weight

    def allocate(self, sim, env):
        super().allocate(sim, env)
        self.previous = wp.zeros(sim.num_envs, dtype=float)

    def before_step(self, sim):
        wp.launch(_position, dim=sim.num_envs, inputs=[sim.data.qpos, self.previous])

    def evaluate(self, sim, reward, terminated):
        wp.launch(
            _running_reward,
            dim=sim.num_envs,
            inputs=[
                sim.data.qpos,
                sim.data.ctrl,
                self.previous,
                self.dt,
                self.run_weight,
                self.ctrl_weight,
                reward,
                terminated,
            ],
        )


@wp.kernel
def _pendulum_reward(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    rewards: wp.array[float],
    terminated: wp.array[int],
):
    world = wp.tid()
    unhealthy = wp.abs(qpos[world, 1]) > 0.2
    for j in range(qpos.shape[1]):
        unhealthy = unhealthy or not wp.isfinite(qpos[world, j])
    for j in range(qvel.shape[1]):
        unhealthy = unhealthy or not wp.isfinite(qvel[world, j])
    rewards[world] = wp.float32(not unhealthy)
    terminated[world] = wp.int32(unhealthy)


class InvertedPendulumTask(JointStateTask):
    def __init__(self, env):
        self.exclude = 0
        self.normal_velocity = 0

    def before_step(self, sim):
        pass

    def evaluate(self, sim, reward, terminated):
        wp.launch(
            _pendulum_reward,
            dim=sim.num_envs,
            inputs=[sim.data.qpos, sim.data.qvel, reward, terminated],
        )
