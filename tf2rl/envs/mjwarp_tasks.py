"""Task adapters for the reusable MJWarp vector backend.

Physics alone does not define an RL task. Each adapter explicitly preserves its
Gymnasium observation, reward, termination and reset semantics on the GPU.
"""

import numpy as np
import warp as wp

_TASKS = {}


def register_mjwarp_task(env_class, factory):
    """Register a factory taking an unwrapped Gymnasium environment.

    Args:
        env_class (type): Unwrapped Gymnasium environment class associated with a task adapter.
        factory (callable): Constructor registered for the selected agent or environment adapter.

    Returns:
        None.
    """
    _TASKS[env_class] = factory


def get_mjwarp_task(env):
    """Select the registered or built-in task adapter for an unwrapped MuJoCo environment.

    Args:
        env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
            spaces, and interaction state.

    Returns:
        object: Task adapter created by the registered factory, exposing allocate, before_step,
            observe, evaluate, and reset operations for batched simulation.

    Raises:
        NotImplementedError: This interface must be implemented by a subclass or is unsupported by
            this historical variant.
    """
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
    """Write per-world position/velocity observations into the supplied Warp output buffer.

    Args:
        qpos (wp.array2d[float]): Per-world generalized positions, indexed by world and position
            coordinate.
        qvel (wp.array2d[float]): Per-world generalized velocities, indexed by world and velocity
            coordinate.
        exclude (int): Number of leading generalized-position coordinates omitted from each
            observation.
        obs (wp.array2d[float]): Output observation matrix written in place with selected positions
            and velocities.

    Returns:
        None.
    """
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
    """Reset selected world integration state and advance their episode counters using seeded
    reset noise.

    Args:
        qpos (wp.array2d[float]): Per-world generalized positions, indexed by world and position
            coordinate.
        qvel (wp.array2d[float]): Per-world generalized velocities, indexed by world and velocity
            coordinate.
        warmstart (wp.array2d[float]): Solver warmstart accelerations cleared in place on reset.
        time (wp.array[float]): Per-world simulation times cleared in place on reset.
        qpos0 (wp.array[float]): Reference generalized positions used when resetting a world.
        mask (wp.array[int]): Per-world reset selection, with nonzero entries identifying worlds to
            reset.
        seeds (sequence[int] or wp.array[int]): Per-episode evaluation seeds or per-world random
            seeds, according to the caller.
        episodes (wp.array[int]): Per-world integer episode counters used to generate deterministic
            independent reset noise.
        noise (float): Uniform position reset-noise scale; velocity noise follows the selected task
            convention.
        normal_velocity (int): Use Gaussian velocity reset noise when nonzero; otherwise use uniform
            noise.

    Returns:
        None.
    """
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
        """Validate scalar-joint/stateless-actuator support and allocate the task reference
        reset configuration.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.

        Returns:
            None.

        Raises:
            ValueError: JointStateTask requires stateless actuators and slide/hinge joints.
        """
        if sim.host_model.na or np.any(np.asarray(sim.host_model.jnt_type) < 2):
            raise ValueError("JointStateTask requires stateless actuators and slide/hinge joints")
        self.qpos0 = wp.array(env.init_qpos.astype(np.float32))
        self.noise = env._reset_noise_scale

    def observe(self, sim, output):
        """Launch the joint-observation kernel to fill the supplied output matrix.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
            output (wp.array2d[float]): Preallocated observation buffer written in place by the task
                observation kernel.

        Returns:
            None.
        """
        wp.launch(
            _joint_observation,
            dim=sim.num_envs,
            inputs=[sim.data.qpos, sim.data.qvel, self.exclude, output],
        )

    def reset(self, sim, mask, seeds, episodes):
        # These tasks have no actuator activation, mocap, history or plugin state.
        # Reset integration state; the next physics step rebuilds derived fields.
        """Launch the task reset kernel for selected worlds, using per-world seeds and episode
        counters.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
            mask (wp.array[int]): Per-world reset selection, with nonzero entries identifying worlds to
                reset.
            seeds (sequence[int] or wp.array[int]): Per-episode evaluation seeds or per-world random
                seeds, according to the caller.
            episodes (wp.array[int]): Per-world integer episode counters used to generate deterministic
                independent reset noise.

        Returns:
            None.
        """
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
    """Copy the current forward position of each world into the previous-position buffer.

    Args:
        qpos (wp.array2d[float]): Per-world generalized positions, indexed by world and position
            coordinate.
        previous (wp.array[float]): Per-world previous forward position used by the running reward.

    Returns:
        None.
    """
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
    """Write forward-velocity-minus-control-cost rewards and clear true-termination flags.

    Args:
        qpos (wp.array2d[float]): Per-world generalized positions, indexed by world and position
            coordinate.
        ctrl (array_like): Actuator controls; Warp kernels write their clipped controls into this
            buffer.
        previous (wp.array[float]): Per-world previous forward position used by the running reward.
        dt (float): Physical time elapsed per environment step.
        run_weight (float): Coefficient multiplying forward velocity in the running reward.
        ctrl_weight (float): Coefficient multiplying the squared-control cost.
        rewards (wp.array[float]): Preallocated per-world reward buffer written in place by the task
            reward kernel.
        terminated (wp.array[int]): Per-world true-termination flags, read by boundary handling or
            written by reward kernels.

    Returns:
        None.
    """
    world = wp.tid()
    cost = float(0.0)  # noqa: UP018 - Warp requires a dynamic accumulator.
    for j in range(ctrl.shape[1]):
        cost += ctrl[world, j] * ctrl[world, j]
    rewards[world] = run_weight * (qpos[world, 0] - previous[world]) / dt - ctrl_weight * cost
    terminated[world] = 0


class HalfCheetahTask(JointStateTask):
    """HalfCheetah observation/reset adapter with forward-velocity and control-cost rewards.
    """
    def __init__(self, env):
        """Initialize HalfCheetahTask and its configuration.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.

        Returns:
            None: Initializes the instance.
        """
        self.exclude = int(env._exclude_current_positions_from_observation)
        self.normal_velocity = 1
        self.dt = env.dt
        self.run_weight = env._forward_reward_weight
        self.ctrl_weight = env._ctrl_cost_weight

    def allocate(self, sim, env):
        """Allocate shared reset data and the per-world previous-position buffer.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.

        Returns:
            None.
        """
        super().allocate(sim, env)
        self.previous = wp.zeros(sim.num_envs, dtype=float)

    def before_step(self, sim):
        """Record each world forward position before physics advancement.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.

        Returns:
            None.
        """
        wp.launch(_position, dim=sim.num_envs, inputs=[sim.data.qpos, self.previous])

    def evaluate(self, sim, reward, terminated):
        """Write the HalfCheetah velocity/control reward and termination buffers after physics
        advancement.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
            reward (wp.array[float]): Preallocated per-world reward buffer written in place by the task
                reward kernel.
            terminated (wp.array[int]): Per-world true-termination flags, read by boundary handling or
                written by reward kernels.

        Returns:
            None.
        """
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
    """Write pendulum survival rewards and mark nonfinite or excessively tilted worlds as
    terminated.

    Args:
        qpos (wp.array2d[float]): Per-world generalized positions, indexed by world and position
            coordinate.
        qvel (wp.array2d[float]): Per-world generalized velocities, indexed by world and velocity
            coordinate.
        rewards (wp.array[float]): Preallocated per-world reward buffer written in place by the task
            reward kernel.
        terminated (wp.array[int]): Per-world true-termination flags, read by boundary handling or
            written by reward kernels.

    Returns:
        None.
    """
    world = wp.tid()
    unhealthy = wp.abs(qpos[world, 1]) > 0.2
    for j in range(qpos.shape[1]):
        unhealthy = unhealthy or not wp.isfinite(qpos[world, j])
    for j in range(qvel.shape[1]):
        unhealthy = unhealthy or not wp.isfinite(qvel[world, j])
    rewards[world] = wp.float32(not unhealthy)
    terminated[world] = wp.int32(unhealthy)


class InvertedPendulumTask(JointStateTask):
    """Pendulum observation/reset adapter with survival rewards and tilt-based termination.
    """
    def __init__(self, env):
        """Initialize InvertedPendulumTask and its configuration.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state. This implementation does not read this argument.

        Returns:
            None: Initializes the instance.
        """
        self.exclude = 0
        self.normal_velocity = 0

    def before_step(self, sim):
        """Provide the no-op pre-step hook; pendulum rewards need no previous-position state.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
                This implementation does not read this argument.

        Returns:
            None.
        """
        pass

    def evaluate(self, sim, reward, terminated):
        """Write pendulum survival rewards and true-termination flags into the supplied
        buffers.

        Args:
            sim (MJWarpSimulator): Allocated batched simulator providing model, data, and device state.
            reward (wp.array[float]): Preallocated per-world reward buffer written in place by the task
                reward kernel.
            terminated (wp.array[int]): Per-world true-termination flags, read by boundary handling or
                written by reward kernels.

        Returns:
            None.
        """
        wp.launch(
            _pendulum_reward,
            dim=sim.num_envs,
            inputs=[sim.data.qpos, sim.data.qvel, reward, terminated],
        )
