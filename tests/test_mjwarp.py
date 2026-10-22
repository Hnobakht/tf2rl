"""GPU integration checks; skipped when the optional MJWarp runtime is absent."""

import importlib.util
import unittest

import gymnasium as gym
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "MJWarp extra is not installed")
class MJWarpIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import warp as wp

        if not tf.config.list_physical_devices("GPU"):
            raise unittest.SkipTest("A CUDA GPU is required")
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        wp.config.quiet = True
        from tf2rl.envs.vector_env import make_vector_env

        cls.env = make_vector_env("HalfCheetah-v5", 4, backend="mjwarp", max_episode_steps=8)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_native_physics_reward_and_action_clipping(self):
        env = self.env
        env.tensors = False
        env.reset(seed=7)
        native = gym.make("HalfCheetah-v5", max_episode_steps=8)
        try:
            native.reset(seed=7)
            native.unwrapped.set_state(
                env.sim.data.qpos.numpy()[0].astype(np.float64),
                env.sim.data.qvel.numpy()[0].astype(np.float64),
            )
            for step in range(8):
                action = 2.0 * np.sin(np.arange(env.sim.host_model.nu) + step).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(np.tile(action, (4, 1)))
                expected, expected_reward, expected_term, expected_trunc, _ = native.step(
                    np.clip(action, -1, 1)
                )
                np.testing.assert_allclose(info["final_obs"][0], expected, atol=3e-3, rtol=3e-3)
                np.testing.assert_allclose(reward[0], expected_reward, atol=3e-3, rtol=3e-3)
                self.assertEqual(terminated[0], expected_term)
                self.assertEqual(truncated[0], expected_trunc)
            self.assertTrue(truncated.all())
            self.assertFalse(np.array_equal(obs, info["final_obs"]))
            env.check_overflow()
        finally:
            native.close()

    def test_seed_snapshot_and_partial_autoreset(self):
        import tensorflow as tf

        env = self.env
        env.tensors = True
        try:
            first, _ = env.reset(seed=17)
            snapshot = first.numpy().copy()
            np.testing.assert_array_equal(env.reset(seed=17)[0].numpy(), snapshot)
            self.assertFalse(np.array_equal(env.reset(seed=18)[0].numpy(), snapshot))
            env.reset(seed=17)
            env._steps.assign(np.array([7, 0, 7, 0], np.int32))
            with tf.device("/GPU:0"):
                action = tf.zeros((4, env.sim.host_model.nu))
            obs, _, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(truncated.numpy(), [True, False, True, False])
            self.assertFalse(terminated.numpy().any())
            np.testing.assert_array_equal(obs.numpy()[[1, 3]], info["final_obs"].numpy()[[1, 3]])
            self.assertFalse(np.array_equal(obs.numpy()[0], info["final_obs"].numpy()[0]))
            np.testing.assert_array_equal(first.numpy(), snapshot)
            # Verify that a reset(seed=...) updates the RNG used by captured resets.
            episode_reset = obs.numpy().copy()
            env.reset(seed=17)
            env._steps.assign(np.array([7, 0, 7, 0], np.int32))
            np.testing.assert_array_equal(env.step(action)[0].numpy(), episode_reset)
        finally:
            env.tensors = False

    def test_other_model_and_termination(self):
        from tf2rl.envs.vector_env import make_vector_env

        env = make_vector_env("InvertedPendulum-v5", 2, backend="mjwarp")
        try:
            obs, _ = env.reset(seed=11)
            self.assertEqual(obs.shape, (2, 4))
            self.assertEqual(env.single_action_space.shape, (1,))
            positions = env.sim.data.qpos.numpy()
            positions[0, 1] = 0.3
            env.sim.data.qpos.assign(positions)
            obs, rewards, terminated, truncated, info = env.step(np.zeros((2, 1), np.float32))
            self.assertTrue(terminated[0])
            self.assertEqual(rewards[0], 0.0)
            self.assertFalse(truncated.any())
            self.assertLess(abs(obs[0, 1]), 0.02)
            self.assertGreater(abs(info["final_obs"][0, 1]), 0.2)
            env.check_overflow()
        finally:
            env.close()

    def test_custom_task_registration_and_options(self):
        from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

        from tf2rl.envs.mjwarp_tasks import HalfCheetahTask, register_mjwarp_task
        from tf2rl.envs.vector_env import make_vector_env

        class CustomCheetah(HalfCheetahEnv):
            pass

        gym.register("MJWarpCustomTest-v0", entry_point=CustomCheetah, max_episode_steps=8)
        register_mjwarp_task(CustomCheetah, HalfCheetahTask)
        kwargs = {
            "exclude_current_positions_from_observation": False,
            "forward_reward_weight": 2.0,
            "ctrl_cost_weight": 0.3,
            "reset_noise_scale": 0.0,
        }
        env = make_vector_env("MJWarpCustomTest-v0", 2, backend="mjwarp", **kwargs)
        native = gym.make("HalfCheetah-v5", **kwargs)
        try:
            obs, _ = env.reset(seed=[3, 4])
            self.assertEqual(obs.shape, (2, 18))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(env.single_observation_space.contains(obs[0]))
            native.reset(seed=3)
            action = np.full((2, env.single_action_space.shape[0]), 0.1, np.float32)
            obs, reward, _, _, _ = env.step(action)
            expected, expected_reward, _, _, _ = native.step(action[0])
            np.testing.assert_allclose(obs[0], expected, atol=3e-3, rtol=3e-3)
            np.testing.assert_allclose(reward[0], expected_reward, atol=3e-3, rtol=3e-3)
        finally:
            env.close()
            native.close()
            gym.envs.registry.pop("MJWarpCustomTest-v0")

    def test_raw_simulator_without_registered_task(self):
        import mujoco

        from tf2rl.envs.mjwarp import MJWarpSimulator

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        sim = MJWarpSimulator(model, num_envs=3)
        try:
            before = sim.data.qpos.numpy().copy()
            sim.step(np.zeros((3, 0), np.float32))
            self.assertTrue(np.all(sim.data.qpos.numpy()[:, 2] < before[:, 2]))
            sim.reset()
            np.testing.assert_array_equal(sim.data.qpos.numpy(), before)
        finally:
            sim.close()

    def test_ppo_update_on_gpu(self):
        import tensorflow as tf

        from tf2rl.algos.ppo import PPO
        from tf2rl.experiments.vector_on_policy_trainer import VectorOnPolicyTrainer

        env = self.env
        env.tensors = True
        tf.keras.utils.set_random_seed(7)
        try:
            with tf.device("/GPU:0"):
                policy = PPO(
                    state_shape=env.single_observation_space.shape,
                    action_dim=env.single_action_space.shape[0],
                    is_discrete=False,
                    horizon=64,
                    batch_size=16,
                    n_epoch=2,
                    actor_units=(32, 32),
                    critic_units=(32, 32),
                    entropy_coef=0.0,
                    gpu=0,
                )
                trainer = VectorOnPolicyTrainer(policy, env, 16, seed=7)
                action, logp, _ = policy.get_action_and_val_tensor(trainer.obs)
                np.testing.assert_allclose(
                    logp.numpy(),
                    policy.actor.compute_log_probs(trainer.obs, action).numpy(),
                    atol=1e-5,
                )
                before = [variable.numpy().copy() for variable in policy.actor.trainable_variables]
                rollout, stats = trainer.collect()
                losses, updates = trainer.update(*rollout)
                self.assertTrue(np.isfinite(losses.numpy()).all())
                self.assertGreater(int(updates.numpy()), 0)
                self.assertEqual(stats["episodes"], 8)
                self.assertTrue(
                    any(
                        not np.array_equal(old, variable.numpy())
                        for old, variable in zip(before, policy.actor.trainable_variables)
                    )
                )
                for variable in policy.actor.variables + policy.critic.variables:
                    self.assertIn("GPU:0", variable.value.device)
                env.check_overflow()
        finally:
            env.tensors = False

    def test_cpu_policy_with_gpu_simulation(self):
        import tensorflow as tf

        from tf2rl.algos.ppo import PPO
        from tf2rl.experiments.vector_on_policy_trainer import VectorOnPolicyTrainer

        env = self.env
        env.tensors = True
        try:
            with tf.device("/CPU:0"):
                policy = PPO(
                    state_shape=env.single_observation_space.shape,
                    action_dim=env.single_action_space.shape[0],
                    is_discrete=False,
                    horizon=32,
                    batch_size=16,
                    n_epoch=1,
                    actor_units=(32, 32),
                    critic_units=(32, 32),
                    entropy_coef=0.0,
                    gpu=-1,
                )
                trainer = VectorOnPolicyTrainer(policy, env, 8, seed=3)
                action, _, _ = policy.get_action_and_val_tensor(trainer.obs)
                self.assertIn("CPU:0", action.device)
                self.assertIn("GPU:0", trainer.obs.device)
                rollout, _ = trainer.collect()
                losses, updates = trainer.update(*rollout)
                self.assertIn("CPU:0", losses.device)
                self.assertTrue(np.isfinite(losses.numpy()).all())
                self.assertGreater(int(updates.numpy()), 0)
                variables = (
                    policy.actor.variables
                    + policy.critic.variables
                    + policy.actor_optimizer.variables
                    + policy.critic_optimizer.variables
                )
                for variable in variables:
                    self.assertIn("CPU:0", variable.value.device)
                env.check_overflow()
        finally:
            env.tensors = False


class RolloutTargets(unittest.TestCase):
    def test_terminal_time_limit_and_rollout_boundary(self):
        from tf2rl.experiments.vector_on_policy_trainer import rollout_targets

        rng = np.random.default_rng(7)
        rewards, values, next_values = [
            rng.normal(size=(5, 3)).astype(np.float32) for _ in range(3)
        ]
        terminated = np.zeros((5, 3), bool)
        truncated = np.zeros((5, 3), bool)
        terminated[2, 0] = True
        truncated[1, 1] = True
        expected_adv = np.zeros_like(rewards)
        expected_ret = np.zeros_like(rewards)
        adv = np.zeros(3)
        ret = next_values[-1].copy()
        for step in reversed(range(5)):
            bootstrap = np.where(terminated[step], 0.0, next_values[step])
            boundary = terminated[step] | truncated[step]
            adv = (
                rewards[step]
                + 0.99 * bootstrap
                - values[step]
                + 0.99 * 0.95 * np.where(boundary, 0.0, adv)
            )
            ret = rewards[step] + 0.99 * np.where(boundary, bootstrap, ret)
            expected_adv[step], expected_ret[step] = adv, ret
        advantages, returns = rollout_targets(
            rewards, values, next_values, terminated, truncated, 0.99, 0.95
        )
        np.testing.assert_allclose(advantages.numpy(), expected_adv, atol=1e-6)
        np.testing.assert_allclose(returns.numpy(), expected_ret, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
