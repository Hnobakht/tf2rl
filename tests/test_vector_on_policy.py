"""The vector path must honor every on-policy agent's objective and tensor interface."""

import importlib.util
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OnPolicyAgent
from tf2rl.envs.vector_env import make_vector_env
from tf2rl.experiments.on_policy import make_policy, register_on_policy_algorithm
from tf2rl.experiments.vector_on_policy_trainer import VectorOnPolicyTrainer

for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)


class HookAgent(OnPolicyAgent):
    """Exercise the public contract without actor/critic attributes or PPO ancestry."""

    def __init__(self, stop_after=None, **kwargs):
        super().__init__(name="HookAgent", gpu=-1, **kwargs)
        self.stop_after = stop_after
        self.updates = self.add_weight(shape=(), initializer="zeros", trainable=False)

    def get_action_and_val_tensor(self, states, test=False):
        count = tf.shape(states)[0]
        return tf.zeros((count, 1)), tf.zeros(count), tf.zeros(count)

    def train_tensor(self, states, actions, advantages, logp_olds, returns):
        self.updates.assign_add(1)
        return tf.stack((tf.convert_to_tensor(self.updates), 0.0, 1000000.0))

    def should_stop_update(self, diagnostics):
        if self.stop_after is None:
            return super().should_stop_update(diagnostics)
        return self.updates >= self.stop_after


class SharedActorCritic(tf.keras.Model):
    """Small Gaussian shared network for the architecture-independent value hook."""

    def __init__(self):
        super().__init__()
        self.features = tf.keras.layers.Dense(8, activation="tanh")
        self.mean = tf.keras.layers.Dense(1)
        self.value = tf.keras.layers.Dense(1)

    def call(self, states, test=False):
        features = self.features(states)
        mean = self.mean(features)
        action = mean if test else mean + 0.5 * tf.random.normal(tf.shape(mean))
        return action, self.compute_log_probs(states, action), self.value(features)[:, 0]

    def compute_log_probs(self, states, actions):
        mean = self.mean(self.features(states))
        return tf.reduce_sum(-0.5 * ((actions - mean) / 0.5) ** 2, axis=-1)

    def compute_entropy(self, states):
        return tf.ones(tf.shape(states)[0])


class OnPolicyContracts(unittest.TestCase):
    def test_ppo_and_vpg_preserve_their_actor_objectives(self):
        obs = gym.spaces.Box(-10, 10, (3,), dtype=np.float32)
        act = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        states = tf.constant(np.random.default_rng(12).normal(size=(8, 3)), tf.float32)
        actions = tf.zeros((8, 1))
        for algorithm in ("ppo", "vpg"):
            with self.subTest(algorithm=algorithm):
                policy = make_policy(algorithm, obs, act, horizon=8, batch_size=8, units=(8, 8))
                policy.initialize_tensor_training(states)
                logp = policy.actor.compute_log_probs(states, actions)
                old_logp = logp - tf.math.log(3.0)
                expected = -1.2 if algorithm == "ppo" else -float(tf.reduce_mean(logp))
                advantages = tf.ones(8)
                returns = policy.get_value_tensor(states) + 1
                losses = policy.train_tensor(states, actions, advantages, old_logp, returns)
                self.assertAlmostEqual(float(losses[0]), expected, places=5)
                self.assertTrue(np.isfinite(losses.numpy()).all())
                self.assertEqual(policy.n_epoch, 10 if algorithm == "ppo" else 1)

    def test_continuous_and_discrete_rollouts_updates_and_restore(self):
        for algorithm in ("ppo", "vpg"):
            for env_id in ("Pendulum-v1", "CartPole-v1"):
                with self.subTest(algorithm=algorithm, env=env_id):
                    env = make_vector_env(
                        env_id, 2, tensors=True, device="/CPU:0", max_episode_steps=4
                    )
                    try:
                        kwargs = {"horizon": 16, "batch_size": 8, "epochs": 2, "units": (8, 8)}
                        policy = make_policy(
                            algorithm,
                            env.single_observation_space,
                            env.single_action_space,
                            **kwargs,
                        )
                        if algorithm == "ppo":
                            policy.target_kl = 1e9
                        trainer = VectorOnPolicyTrainer(policy, env, 8, seed=3)
                        action, logp, _ = policy.get_action_and_val_tensor(trainer.obs)
                        np.testing.assert_allclose(
                            logp, policy.actor.compute_log_probs(trainer.obs, action), atol=1e-5
                        )
                        before = [v.numpy().copy() for v in policy.actor.trainable_variables]
                        rollout, stats = trainer.collect()
                        losses, updates = trainer.update(*rollout)
                        self.assertEqual(int(updates), 4)
                        self.assertGreaterEqual(stats["episodes"], 4)
                        self.assertTrue(np.isfinite(losses.numpy()).all())
                        self.assertTrue(
                            any(
                                not np.array_equal(old, v.numpy())
                                for old, v in zip(before, policy.actor.trainable_variables)
                            )
                        )
                        with tempfile.TemporaryDirectory() as directory:
                            prefix = tf.train.Checkpoint(policy=policy).write(directory + "/model")
                            restored = make_policy(
                                algorithm,
                                env.single_observation_space,
                                env.single_action_space,
                                **kwargs,
                            )
                            restored.initialize_tensor_training(trainer.obs)
                            tf.train.Checkpoint(policy=restored).read(
                                prefix
                            ).assert_existing_objects_matched().expect_partial()
                            expected = policy.get_action_and_val_tensor(trainer.obs, test=True)
                            actual = restored.get_action_and_val_tensor(trainer.obs, test=True)
                            for first, second in zip(expected, actual):
                                np.testing.assert_allclose(first, second, atol=1e-5)
                    finally:
                        env.close()

    def test_shared_actor_critic(self):
        for algorithm in ("ppo", "vpg"):
            with self.subTest(algorithm=algorithm):
                env = make_vector_env("Pendulum-v1", 2, tensors=True, device="/CPU:0")
                try:
                    with tf.device("/CPU:0"):
                        shared = SharedActorCritic()
                        policy = make_policy(
                            algorithm,
                            env.single_observation_space,
                            env.single_action_space,
                            horizon=16,
                            batch_size=8,
                            policy_kwargs={"actor_critic": shared},
                            epochs=1,
                        )
                        trainer = VectorOnPolicyTrainer(policy, env, 8)
                        before = [v.numpy().copy() for v in shared.trainable_variables]
                        rollout, _ = trainer.collect()
                        losses, updates = trainer.update(*rollout)
                        self.assertIsNone(policy.actor)
                        self.assertTrue(np.isfinite(losses.numpy()).all())
                        self.assertGreater(int(updates), 0)
                        self.assertTrue(
                            any(
                                not np.array_equal(old, v.numpy())
                                for old, v in zip(before, shared.trainable_variables)
                            )
                        )
                finally:
                    env.close()

    def test_custom_agent_and_algorithm_owned_stopping(self):
        for limit, expected in ((None, 6), (2, 2)):
            with self.subTest(limit=limit):
                env = make_vector_env("Pendulum-v1", 2, tensors=True, device="/CPU:0")
                try:
                    policy = HookAgent(stop_after=limit, horizon=16, batch_size=8, n_epoch=3)
                    trainer = VectorOnPolicyTrainer(policy, env, 8)
                    rollout, _ = trainer.collect()
                    _, updates = trainer.update(*rollout)
                    self.assertEqual(int(updates), expected)
                    self.assertEqual(int(policy.updates), expected)
                    self.assertFalse(hasattr(policy, "actor"))
                finally:
                    env.close()
        from tf2rl.algos.ppo import PPO
        from tf2rl.algos.vpg import VPG

        kwargs = {
            "state_shape": (3,),
            "action_dim": 1,
            "is_discrete": False,
            "gpu": -1,
            "horizon": 8,
            "batch_size": 8,
        }
        self.assertTrue(bool(PPO(**kwargs).should_stop_update(tf.constant([0.0, 0.0, 1.0]))))
        self.assertFalse(bool(VPG(**kwargs).should_stop_update(tf.constant([0.0, 0.0, 1.0]))))

    def test_custom_algorithm_registration(self):
        def factory(**kwargs):
            return HookAgent(
                horizon=kwargs["horizon"],
                batch_size=kwargs["batch_size"],
                n_epoch=kwargs["n_epoch"],
            )

        register_on_policy_algorithm("contract_test", factory, default_epochs=3)
        policy = make_policy(
            "contract_test",
            gym.spaces.Box(-1, 1, (3,), dtype=np.float32),
            gym.spaces.Box(-1, 1, (1,), dtype=np.float32),
            horizon=8,
            batch_size=8,
        )
        self.assertIsInstance(policy, HookAgent)
        self.assertEqual(policy.n_epoch, 3)


@unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "MJWarp extra is not installed")
class OnPolicyGPUIntegration(unittest.TestCase):
    def test_cpu_function_result_with_gpu_caller(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("A CUDA GPU is required")
        env = make_vector_env("HalfCheetah-v5", 4, backend="mjwarp", tensors=True)
        try:

            @tf.function
            def cpu_controls(inputs):
                with tf.device("/CPU:0"):
                    return inputs + 0.125

            with tf.device("/GPU:0"):
                action = cpu_controls(tf.zeros((4, env.single_action_space.shape[0])))
            env.reset(seed=7)
            obs, rewards, _, _, _ = env.step(action)
            self.assertIn("GPU:0", obs.device)
            self.assertTrue(np.isfinite(rewards.numpy()).all())
            env.check_overflow()
        finally:
            env.close()

    def test_all_algorithms_with_cpu_and_gpu_learners(self):
        if not tf.config.list_physical_devices("GPU"):
            self.skipTest("A CUDA GPU is required")
        env = make_vector_env(
            "HalfCheetah-v5", 4, backend="mjwarp", tensors=True, max_episode_steps=4
        )
        try:
            for algorithm in ("ppo", "vpg"):
                for gpu in (-1, 0):
                    with self.subTest(algorithm=algorithm, gpu=gpu):
                        policy = make_policy(
                            algorithm,
                            env.single_observation_space,
                            env.single_action_space,
                            horizon=32,
                            batch_size=8,
                            epochs=2,
                            units=(16, 16),
                            gpu=gpu,
                        )
                        if algorithm == "ppo":
                            policy.target_kl = 1e9
                        else:
                            # A PPO-looking attribute must not change VPG's update schedule.
                            policy.target_kl = -1e9
                        trainer = VectorOnPolicyTrainer(policy, env, 8, seed=5)
                        action, _, _ = policy.get_action_and_val_tensor(trainer.obs)
                        self.assertIn("CPU:0" if gpu < 0 else "GPU:0", action.device)
                        before = [v.numpy().copy() for v in policy.actor.trainable_variables]
                        rollout, stats = trainer.collect()
                        losses, updates = trainer.update(*rollout)
                        self.assertIn("GPU:0", trainer.obs.device)
                        self.assertEqual(stats["episodes"], 8)
                        self.assertEqual(int(updates), 8)
                        self.assertTrue(np.isfinite(losses.numpy()).all())
                        self.assertTrue(
                            any(
                                not np.array_equal(old, v.numpy())
                                for old, v in zip(before, policy.actor.trainable_variables)
                            )
                        )
                        for v in (
                            policy.actor.variables
                            + policy.critic.variables
                            + policy.actor_optimizer.variables
                            + policy.critic_optimizer.variables
                        ):
                            self.assertIn("CPU:0" if gpu < 0 else "GPU:0", v.value.device)
            env.check_overflow()
        finally:
            env.close()


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    unittest.main()
