"""Optional, algorithm-independent batched MuJoCo Warp simulation.

MJWarpSimulator accepts any MJWarp-compatible MjModel. MJWarpVectorEnv adds
task semantics supplied by a task adapter; see mjwarp_tasks for built-in adapters.
TensorFlow is imported only when tensor output or TensorFlow actions are used.
"""

from typing import ClassVar

import gymnasium as gym
import mujoco_warp as mjw
import numpy as np
import warp as wp
from gymnasium.vector.utils import batch_space


def as_warp(value, device):
    """Import eager TensorFlow actions without a host copy, or upload NumPy actions.

    Args:
        value (wp.array or np.ndarray or tf.Tensor): Per-world controls; NumPy values are uploaded
            as float32, while Warp/tensor inputs must already expose compatible float32 storage.
        device (str): TensorFlow or Warp device identifier appropriate to this operation.

    Returns:
        wp.array: Contiguous float32 Warp array on the selected device, using DLPack for tensor
            inputs where supported.
    """
    if isinstance(value, wp.array):
        result = value
    elif isinstance(value, np.ndarray):
        result = wp.array(value, dtype=wp.float32, device=device)
    else:
        import tensorflow as tf

        # Inspect the backing allocation, not Tensor.device: tf.function can
        # label a CPU result with its GPU caller's device. Copy controls only
        # when the DLPack payload is actually on a different device.
        result = wp.from_dlpack(tf.experimental.dlpack.to_dlpack(value))
        if result.device != device:
            with tf.device(f"/GPU:{device.ordinal}"):
                value = tf.identity(value)
            result = wp.from_dlpack(tf.experimental.dlpack.to_dlpack(value))
    if result.device != device or result.dtype != wp.float32:
        raise ValueError(f"Actions must be float32 on {device}; got {result.dtype} on {result.device}")
    return result


def export_arrays(arrays, tensors, device):
    """Export owned snapshots so later simulator steps cannot overwrite rollouts.

    Args:
        arrays (sequence[wp.array]): Simulator buffers to snapshot and export.
        tensors (bool): Export observations and step results as TensorFlow tensors instead of NumPy
            arrays.
        device (str): TensorFlow or Warp device identifier appropriate to this operation.

    Returns:
        tuple: NumPy copies or TensorFlow tensor snapshots, in the same order as arrays. Device
            buffers are copied before exposing tensors.
    """
    if not tensors:
        return tuple(array.numpy() for array in arrays)
    import tensorflow as tf

    copies = [wp.clone(array) for array in arrays]
    # TensorFlow uses the legacy capsule API, which has no stream handshake.
    wp.synchronize_stream(wp.get_stream(device))
    return tuple(tf.experimental.dlpack.from_dlpack(wp.to_dlpack(a)) for a in copies)


class MJWarpSimulator:
    """CUDA physics for arbitrary MuJoCo models, independent of reward and policy."""

    def __init__(self, model, num_envs=1, frame_skip=1, device="cuda:0", nconmax=64, njmax=128):
        """Initialize MJWarpSimulator and its configuration.

        Args:
            model (mujoco.MjModel): Compiled single-world MuJoCo model to transfer to the batched Warp
                simulator.
            num_envs (int): Number of independent simulation worlds.
            frame_skip (int): Number of physics integration steps advanced for each environment step.
            device (str): TensorFlow or Warp device identifier appropriate to this operation.
            nconmax (int): Allocated per-backend contact-buffer capacity.
            njmax (int): Allocated per-backend constraint-buffer capacity.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: Simulation sizes must be positive. The MJWarp backend requires an NVIDIA CUDA
                device.
        """
        if min(num_envs, frame_skip, nconmax, njmax) < 1:
            raise ValueError("Simulation sizes must be positive")
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("The MJWarp backend requires an NVIDIA CUDA device")
        self.host_model = model
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        with wp.ScopedDevice(self.device):
            self.model = mjw.put_model(model)
            self.data = mjw.make_data(model, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            self.advance()
            wp.synchronize_device(self.device)
            with wp.ScopedCapture() as capture:
                self.advance()
            self.graph = capture.graph
            mjw.reset_data(self.model, self.data)

    def advance(self):
        """Launch physics kernels; also usable inside a larger CUDA graph capture.

        Returns:
            None.
        """
        for _ in range(self.frame_skip):
            mjw.step(self.model, self.data)

    def step(self, actions):
        """Set controls and advance all worlds; return the mutable MJWarp Data.

        Args:
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.

        Returns:
            mujoco_warp.Data: Mutable simulator data after assigning controls and advancing frame_skip
                physics steps.

        Raises:
            ValueError: Action shape must be (num_envs, model.nu).
        """
        with wp.ScopedDevice(self.device):
            source = as_warp(actions, self.device)
            if source.shape != (self.num_envs, self.host_model.nu):
                raise ValueError("Action shape must be (num_envs, model.nu)")
            wp.copy(self.data.ctrl, source)
            wp.capture_launch(self.graph)
            return self.data

    def reset(self, mask=None):
        """Reset all worlds or a per-world Warp mask, retaining allocated buffers.

        Args:
            mask (wp.array[int] or None): Per-world reset selection; None resets all simulator worlds.

        Returns:
            None.
        """
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.model, self.data, reset=mask)

    def check_overflow(self):
        """Check contact/constraint capacity at rollout boundaries.

        Returns:
            None.

        Raises:
            RuntimeError: MJWarp buffer overflow: increase nconmax/njmax.
        """
        if np.any(self.data.overflow.numpy()):
            raise RuntimeError("MJWarp buffer overflow: increase nconmax/njmax")

    def close(self):
        """Synchronize pending Warp device work before the simulator is released.

        Returns:
            None.
        """
        wp.synchronize_device(self.device)


@wp.kernel
def _clip_actions(
    actions: wp.array2d[float], low: wp.array[float], high: wp.array[float], ctrl: wp.array2d[float]
):
    """Clip per-world actions to actuator bounds and write the simulator control buffer.

    Args:
        actions (array_like): Actions aligned with the observation batch; continuous actions have a
            final action_dim axis.
        low (array_like): Lower continuous-action bounds, one per actuator.
        high (array_like): Upper continuous-action bounds, one per actuator.
        ctrl (array_like): Actuator controls; Warp kernels write their clipped controls into this
            buffer.

    Returns:
        None.
    """
    world, actuator = wp.tid()
    ctrl[world, actuator] = wp.clamp(actions[world, actuator], low[actuator], high[actuator])


@wp.kernel
def _boundaries(
    terminated: wp.array[int],
    steps: wp.array[int],
    limit: int,
    truncated: wp.array[int],
    reset: wp.array[int],
):
    """Advance episode lengths and write time-limit and autoreset masks for each world.

    Args:
        terminated (wp.array[int]): Per-world true-termination flags, read by boundary handling or
            written by reward kernels.
        steps (wp.array[int]): Per-world episode step counters incremented in place.
        limit (int): Maximum episode length used to set truncation flags.
        truncated (wp.array[int]): Output per-world time-limit flags written in place by the
            boundary kernel.
        reset (wp.array[int]): Output reset mask set for terminated or truncated worlds.

    Returns:
        None.
    """
    world = wp.tid()
    steps[world] += 1
    truncated[world] = wp.int32(steps[world] >= limit)
    reset[world] = wp.int32(terminated[world] != 0 or truncated[world] != 0)
    if reset[world] != 0:
        steps[world] = 0


class MJWarpVectorEnv(gym.vector.VectorEnv):
    """Gymnasium five-value vector API, with same-step autoreset and optional GPU tensors.

    A task implements allocate(sim, env), before_step(sim), observe(sim, output),
    evaluate(sim, reward, terminated), and reset(sim, mask, seeds, episodes).
    Those methods launch Warp kernels and must support CUDA graph capture.
    Register adapters with register_mjwarp_task, or pass task= explicitly.
    """

    metadata: ClassVar[dict] = {
        "render_modes": [],
        "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
    }

    def __init__(
        self,
        env_id,
        num_envs=1,
        *,
        task=None,
        tensors=False,
        seed=None,
        device="cuda:0",
        nconmax=64,
        njmax=128,
        **env_kwargs,
    ):
        """Initialize MJWarpVectorEnv and its configuration.

        Args:
            env_id (str): Registered Gymnasium environment identifier.
            num_envs (int): Number of independent simulation worlds.
            task (object or None): Task adapter supplying observation, reward, termination, and reset
                kernels.
            tensors (bool): Export observations and step results as TensorFlow tensors instead of NumPy
                arrays.
            seed (int or None): Random seed; None leaves initialization to the underlying random
                generator.
            device (str): TensorFlow or Warp device identifier appropriate to this operation.
            nconmax (int): Allocated per-backend contact-buffer capacity.
            njmax (int): Allocated per-backend constraint-buffer capacity.
            **env_kwargs (dict): Additional keyword arguments forwarded to gym.vector.VectorEnv
                construction.

        Returns:
            None: Initializes the instance.

        Raises:
            TypeError: The MJWarp backend requires a MuJoCo environment. The MJWarp backend requires Box
                controls.
            ValueError: Specify max_episode_steps for the vector environment.
        """
        from tf2rl.envs.mjwarp_tasks import get_mjwarp_task

        self.num_envs = num_envs
        self.tensors = tensors
        self.closed = True
        self.render_mode = None
        self._reference = gym.make(env_id, **env_kwargs)
        try:
            env = self._reference.unwrapped
            if not isinstance(env, gym.envs.mujoco.MujocoEnv):
                raise TypeError("The MJWarp backend requires a MuJoCo environment")
            if not isinstance(env.action_space, gym.spaces.Box):
                raise TypeError("The MJWarp backend requires Box controls")
            self.single_observation_space = gym.spaces.Box(
                env.observation_space.low.astype(np.float32),
                env.observation_space.high.astype(np.float32),
                dtype=np.float32,
            )
            self.single_action_space = env.action_space
            self.observation_space = batch_space(self.single_observation_space, num_envs)
            self.action_space = batch_space(env.action_space, num_envs)
            self.spec = self._reference.spec
            self.max_episode_steps = self.spec.max_episode_steps
            if self.max_episode_steps is None:
                raise ValueError("Specify max_episode_steps for the vector environment")
            self.task = task if task is not None else get_mjwarp_task(env)
            self.sim = MJWarpSimulator(env.model, num_envs, env.frame_skip, device, nconmax, njmax)
            with wp.ScopedDevice(self.sim.device):
                self.task.allocate(self.sim, env)
                self._actions = wp.zeros((num_envs, env.model.nu), dtype=float)
                self._low = wp.array(env.action_space.low.astype(np.float32))
                self._high = wp.array(env.action_space.high.astype(np.float32))
                self._obs = wp.zeros((num_envs, *env.observation_space.shape), dtype=float)
                self._final = wp.zeros_like(self._obs)
                self._rewards = wp.zeros(num_envs, dtype=float)
                self._terminated = wp.zeros(num_envs, dtype=int)
                self._truncated = wp.zeros(num_envs, dtype=int)
                self._reset = wp.zeros(num_envs, dtype=int)
                self._steps = wp.zeros(num_envs, dtype=int)
                self._episodes = wp.zeros(num_envs, dtype=int)
                self._seeds = wp.zeros(num_envs, dtype=int)
                self.reset(seed=seed)
                self._advance()
                wp.synchronize_device(self.sim.device)
                with wp.ScopedCapture() as capture:
                    self._advance()
                self._graph = capture.graph
                self.reset(seed=seed)
            self.closed = False
        except BaseException:
            if hasattr(self, "sim"):
                self.sim.close()
            self._reference.close()
            raise

    def _advance(self):
        """Advance batched physics through the simulator, using captured execution when
        available.

        Returns:
            None.
        """
        sim = self.sim
        wp.launch(
            _clip_actions,
            dim=self._actions.shape,
            inputs=[self._actions, self._low, self._high, sim.data.ctrl],
        )
        self.task.before_step(sim)
        sim.advance()
        self.task.observe(sim, self._final)
        self.task.evaluate(sim, self._rewards, self._terminated)
        wp.launch(
            _boundaries,
            dim=self.num_envs,
            inputs=[
                self._terminated,
                self._steps,
                self.max_episode_steps,
                self._truncated,
                self._reset,
            ],
        )
        self.task.reset(sim, self._reset, self._seeds, self._episodes)
        self.task.observe(sim, self._obs)

    def reset(self, *, seed=None, options=None):
        """Reset every world, optional per-world seeds, and episode counters, then produce
        initial observations.

        Args:
            seed (int or None): Random seed; None leaves initialization to the underlying random
                generator.
            options (dict or None): Additional reset settings accepted by the underlying environment.

        Returns:
            tuple: Initial observations shaped (num_envs, observation_width) and an empty info
                dictionary; observation type follows tensors.

        Raises:
            ValueError: MJWarp reset does not support reset options. A seed list must have num_envs
                entries.
        """
        if options:
            raise ValueError("MJWarp reset does not support reset options")
        with wp.ScopedDevice(self.sim.device):
            if seed is not None or not hasattr(self, "_seeded"):
                if isinstance(seed, (list, tuple)):
                    if len(seed) != self.num_envs:
                        raise ValueError("A seed list must have num_envs entries")
                    seeds = np.asarray(seed, dtype=np.uint32)
                else:
                    seeds = np.random.SeedSequence(seed).generate_state(
                        self.num_envs, dtype=np.uint32
                    )
                self._seeds.assign(seeds.view(np.int32))
                self._episodes.zero_()
                self._seeded = True
            self.sim.reset()
            self._steps.zero_()
            self._reset.fill_(1)
            self.task.reset(self.sim, self._reset, self._seeds, self._episodes)
            self.task.observe(self.sim, self._obs)
            return export_arrays([self._obs], self.tensors, self.sim.device)[0], {}

    def step(self, actions):
        """Advance all worlds, snapshot physical outcomes, and autoreset completed episodes
        before returning observations.

        Args:
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.

        Returns:
            tuple: Observations, rewards, terminated, truncated, and info. Observation rows autoreset
                immediately; info["final_obs"] preserves physical successors and info["_final_obs"]
                marks episode boundaries. Array/tensor types follow tensors.
        """
        with wp.ScopedDevice(self.sim.device):
            source = as_warp(actions, self.sim.device)
            if source.shape != self._actions.shape:
                raise ValueError(f"Expected actions with shape {self._actions.shape}")
            wp.copy(self._actions, source)
            wp.capture_launch(self._graph)
            obs, reward, terminated, truncated, final, reset = export_arrays(
                [
                    self._obs,
                    self._rewards,
                    self._terminated,
                    self._truncated,
                    self._final,
                    self._reset,
                ],
                self.tensors,
                self.sim.device,
            )
        if self.tensors:
            import tensorflow as tf

            terminated, truncated, reset = [
                tf.cast(a, tf.bool) for a in (terminated, truncated, reset)
            ]
        else:
            terminated, truncated, reset = [a.astype(bool) for a in (terminated, truncated, reset)]
        return obs, reward, terminated, truncated, {"final_obs": final, "_final_obs": reset}

    def check_overflow(self):
        """Delegate contact/constraint capacity checks to the simulator.

        Returns:
            None.
        """
        self.sim.check_overflow()

    def close_extras(self, **kwargs):
        """Close the simulator by synchronizing its pending device work.

        Args:
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            None.
        """
        self.sim.close()
        self._reference.close()
