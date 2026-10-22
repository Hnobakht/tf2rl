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
    """Import eager TensorFlow actions without a host copy, or upload NumPy actions."""
    if isinstance(value, wp.array):
        result = value
    elif isinstance(value, np.ndarray):
        result = wp.array(value, dtype=wp.float32, device=device)
    else:
        import tensorflow as tf

        # A CPU policy may drive GPU physics. Transfer controls explicitly before
        # exporting them; GPU policies retain the existing DLPack path.
        if not value.device.endswith(f"GPU:{device.ordinal}"):
            with tf.device(f"/GPU:{device.ordinal}"):
                value = tf.identity(value)
        result = wp.from_dlpack(tf.experimental.dlpack.to_dlpack(value))
    if result.device != device or result.dtype != wp.float32:
        raise ValueError("Actions must be float32 on the simulator's device")
    return result


def export_arrays(arrays, tensors, device):
    """Export owned snapshots so later simulator steps cannot overwrite rollouts."""
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
        """Launch physics kernels; also usable inside a larger CUDA graph capture."""
        for _ in range(self.frame_skip):
            mjw.step(self.model, self.data)

    def step(self, actions):
        """Set controls and advance all worlds; return the mutable MJWarp Data."""
        with wp.ScopedDevice(self.device):
            source = as_warp(actions, self.device)
            if source.shape != (self.num_envs, self.host_model.nu):
                raise ValueError("Action shape must be (num_envs, model.nu)")
            wp.copy(self.data.ctrl, source)
            wp.capture_launch(self.graph)
            return self.data

    def reset(self, mask=None):
        """Reset all worlds or a per-world Warp mask, retaining allocated buffers."""
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.model, self.data, reset=mask)

    def check_overflow(self):
        """Check contact/constraint capacity at rollout boundaries."""
        if np.any(self.data.overflow.numpy()):
            raise RuntimeError("MJWarp buffer overflow: increase nconmax/njmax")

    def close(self):
        wp.synchronize_device(self.device)


@wp.kernel
def _clip_actions(
    actions: wp.array2d[float], low: wp.array[float], high: wp.array[float], ctrl: wp.array2d[float]
):
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
        self.sim.check_overflow()

    def close_extras(self, **kwargs):
        self.sim.close()
        self._reference.close()
