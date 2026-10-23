"""Replay, absorbing-state and expert-export contracts shared by both backends."""

import importlib.util
import tempfile
import unittest

import gymnasium as gym
import joblib
import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper, VectorAbsorbingWrapper
from tf2rl.envs.vector_env import TensorVectorEnv, make_vector_env
from tf2rl.experiments.demonstrations import (
    evaluate_vector_policy,
    load_trajectories_with_logps,
)
from tf2rl.experiments.off_policy import checkpoint_prefix, make_policy
from tf2rl.experiments.vector_off_policy_trainer import VectorOffPolicyTrainer
from tf2rl.misc.dac_ops import dac_batch_transitions, dac_transitions

# Configure before the CPU algorithm tests initialize the TensorFlow runtime.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)


class BoundaryEnv(gym.Env):
    """Each world has a distinct identity and a known terminal/timeout boundary."""

    def __init__(self, world=0):
        self.world = world
        self.observation_space = gym.spaces.Box(-100, 100, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([self.world, 0], np.float32), {}

    def step(self, action):
        self.steps += 1
        boundary = self.steps == self.world + 2
        return (
            np.array([self.world, self.steps], np.float32),
            float(self.world + 1),
            boundary and self.world != 1,
            boundary and self.world != 0,
            {},
        )


def scripted_env(absorbing=False, tensors=True):
    env = gym.vector.SyncVectorEnv(
        [lambda i=i: BoundaryEnv(i) for i in range(3)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    if tensors:
        env = TensorVectorEnv(env, "/CPU:0")
    return VectorAbsorbingWrapper(env) if absorbing else env


class CountingPolicy(OffPolicyAgent):
    """A deterministic collector oracle, including a supplied-action log-density oracle."""

    def __init__(self, absorbing=False, warmup=0):
        super().__init__(
            name="Counting", memory_capacity=128, batch_size=3, n_warmup=warmup, gpu=-1
        )
        self._is_absorbed_state = absorbing
        self.replay_mask_encoding = "dac" if absorbing else "done"
        self.calls = 0

    def get_action_tensor(self, states, test=False):
        return tf.fill((tf.shape(states)[0], 1), 0.5)

    def compute_action_log_probs(self, states, actions):
        return tf.reduce_sum(states, axis=-1) + tf.reduce_sum(actions, axis=-1)

    def train(self, states, actions, next_states, rewards, dones):
        self.calls += 1
        return np.zeros_like(rewards)


class VectorReplayContracts(unittest.TestCase):
    def test_warmup_fractional_budget_and_final_states(self):
        env = scripted_env()
        try:
            policy = CountingPolicy(warmup=4)
            trainer = VectorOffPolicyTrainer(policy, env, updates_per_transition=0.3, seed=7)
            for _ in range(4):
                trainer.collect_step()
                trainer.update()
            # Twelve physical steps, four warmup: floor(8 * .3) = 2 updates.
            self.assertEqual(trainer.total_steps, 12)
            self.assertEqual(policy.calls, 2)
            self.assertAlmostEqual(trainer.update_credit, 0.4)
            rows = trainer.replay_buffer.get_all_transitions()
            np.testing.assert_array_equal(rows["act"][4:], 0.5)
            self.assertFalse(np.all(rows["act"][:4] == 0.5))
            # Same-step autoreset must never enter the replay's final next state.
            np.testing.assert_array_equal(rows["next_obs"][[3, 7, 11]], [[0, 2], [1, 3], [2, 4]])
            np.testing.assert_array_equal(rows["done"][[3, 7, 11], 0], [1, 0, 1])
        finally:
            env.close()

    def test_batched_dac_matches_scalar_oracle(self):
        env = scripted_env(absorbing=True)
        scalar = AbsorbingWrapper(BoundaryEnv(), indicator=0.1)
        try:
            trainer = VectorOffPolicyTrainer(
                CountingPolicy(True), env, absorbing=True, updates_per_transition=0.25
            )
            for _ in range(4):
                trainer.collect_step()
                trainer.update()
            # Two world-0 terminals and one world-2 simultaneous terminal/timeout.
            self.assertEqual(trainer.total_steps, 12)
            self.assertEqual(trainer.replay_rows, 15)
            self.assertEqual(trainer.total_updates, 3)
            rows = trainer.replay_buffer.get_all_transitions()
            self.assertEqual(np.count_nonzero(rows["done"] == -1), 3)
            terminal = rows["done"][:, 0] == 0
            np.testing.assert_array_equal(
                rows["next_obs"][terminal], np.repeat([[0, 0, 0.1]], 3, axis=0).astype(np.float32)
            )
            timeout = np.flatnonzero((rows["obs"][:, 0] == 1) & (rows["obs"][:, 1] == 2))
            np.testing.assert_array_equal(rows["done"][timeout], [[1]])
            np.testing.assert_array_equal(rows["next_obs"][timeout], [[1, 3, 0]])

            obs = np.array([[0, 1, 0], [1, 2, 0], [2, 3, 0]], np.float32)
            actions, final, rewards = np.ones((3, 1)), obs + [0, 1, 0], np.arange(3)
            batch = dac_batch_transitions(env, obs, actions, final, rewards, [True, False, True])
            oracle = [
                dac_transitions(scalar, obs[i], actions[i], final[i], rewards[i], i != 1, i != 0)
                for i in range(3)
            ]
            expected = [rows[0] for rows in oracle] + [rows[1] for rows in oracle if len(rows) == 2]
            for key in batch:
                np.testing.assert_allclose(
                    batch[key].reshape(5, -1),
                    np.asarray([row[key] for row in expected]).reshape(5, -1),
                )
        finally:
            env.close()
            scalar.close()

    def test_export_episodes_and_load_once_for_il_irl(self):
        for absorbing in (False, True):
            with self.subTest(absorbing=absorbing), tempfile.TemporaryDirectory() as directory:
                env = scripted_env(absorbing=absorbing)
                scalar = AbsorbingWrapper(BoundaryEnv(), indicator=0.1)
                try:
                    policy = CountingPolicy(absorbing)
                    result = evaluate_vector_policy(policy, env, 6, output_dir=directory)
                    self.assertEqual(len(result["paths"]), 6)
                    physical, terminals = 0, 0
                    for filename in result["paths"]:
                        path = joblib.load(filename)
                        world = path["metadata"]["world"]
                        self.assertEqual(len(path["obs"]), world + 2)
                        self.assertEqual(path["done"].sum(), 1)
                        self.assertTrue(path["done"][-1])
                        np.testing.assert_array_equal(path["obs"][:, 0], world)
                        np.testing.assert_array_equal(path["next_obs"][:-1], path["obs"][1:])
                        self.assertEqual(path["next_obs"][-1, 1], world + 2)
                        np.testing.assert_array_equal(
                            path["logp"], path["obs"].sum(-1) + path["act"].sum(-1)
                        )
                        physical += len(path["obs"])
                        terminals += int(path["terminated"][-1])
                    loaded = load_trajectories_with_logps(result["paths"], env=scalar)
                    self.assertEqual(loaded["obses"].shape, (physical + terminals, 3))
                    self.assertEqual(np.count_nonzero(loaded["not_dones"] == -1), terminals)
                    self.assertEqual(np.count_nonzero(loaded["not_dones"] == 0), terminals)
                    loops = loaded["not_dones"][:, 0] == -1
                    np.testing.assert_array_equal(loaded["logps"][loops], 0)
                    self.assertTrue(np.isfinite(loaded["logps"]).all())
                finally:
                    env.close()
                    scalar.close()

    def test_native_object_final_observations(self):
        env = scripted_env(absorbing=True, tensors=False)
        try:
            env.reset(seed=3)
            augmented = env.get_non_absorbing_state(np.ones((3, 2), np.float64))
            self.assertTrue(env.single_observation_space.contains(augmented[0]))
            for _ in range(3):
                _, _, _, _, info = env.step(np.zeros((3, 1), np.float32))
            np.testing.assert_array_equal(info["final_obs"][1], [1, 3, 0])
        finally:
            env.close()

    def test_reject_absorbing_masks_with_boolean_agent(self):
        env = scripted_env(absorbing=True)
        try:
            with self.assertRaises(ValueError):
                VectorOffPolicyTrainer(CountingPolicy(), env, absorbing=True)
        finally:
            env.close()


class ExistingAlgorithmUpdates(unittest.TestCase):
    def test_weights_change_targets_initialize_and_checkpoint_restores(self):
        observation = gym.spaces.Box(-10, 10, (4,), dtype=np.float32)
        action = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        rng = np.random.default_rng(13)
        states, actions, next_states = [
            rng.normal(size=(8, width)).astype(np.float32) for width in (4, 2, 4)
        ]
        rewards = np.ones((8, 1), np.float32)
        for algorithm, absorbing, xla in (
            ("sac", False, False),
            ("sac", False, True),
            ("sac", True, False),
            ("sac", True, True),
            ("td3", False, False),
            ("ddpg", False, False),
        ):
            with self.subTest(algorithm=algorithm, absorbing=absorbing, xla=xla):
                kwargs = {
                    "units": (16, 16),
                    "gpu": -1,
                    "absorbing": absorbing,
                    "batch_size": 8,
                    "xla": xla,
                }
                policy = make_policy(algorithm, observation, action, **kwargs)
                if algorithm != "sac":
                    for model in ("actor", "critic"):
                        for source, target in zip(
                            getattr(policy, model).weights,
                            getattr(policy, model + "_target").weights,
                        ):
                            np.testing.assert_array_equal(source.numpy(), target.numpy())
                before = [v.numpy().copy() for v in policy.actor.trainable_variables]
                masks = np.ones((8, 1), np.float32) if absorbing else np.zeros((8, 1), np.float32)
                for index in range(2):
                    diagnostic = policy.train(states, actions, next_states, rewards, masks)
                    self.assertTrue(np.isfinite(np.asarray(diagnostic)).all())
                    if algorithm == "td3" and index == 0:
                        for old, variable in zip(before, policy.actor.trainable_variables):
                            np.testing.assert_array_equal(old, variable.numpy())
                self.assertTrue(
                    any(
                        not np.array_equal(old, variable.numpy())
                        for old, variable in zip(before, policy.actor.trainable_variables)
                    )
                )
                with tempfile.TemporaryDirectory() as directory:
                    tf.train.Checkpoint(policy=policy).write(directory + "/checkpoint")
                    restored = make_policy(algorithm, observation, action, **kwargs)
                    tf.train.Checkpoint(policy=restored).read(
                        checkpoint_prefix(directory)
                    ).assert_existing_objects_matched().expect_partial()
                    actual = policy.get_action_tensor(states, test=True)
                    expected = restored.get_action_tensor(states, test=True)
                    if isinstance(actual, tuple):
                        actual, expected = actual[0], expected[0]
                    np.testing.assert_allclose(actual, expected)


@unittest.skipUnless(importlib.util.find_spec("mujoco_warp"), "MJWarp extra is not installed")
class GPUOffPolicyIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not tf.config.list_physical_devices("GPU"):
            raise unittest.SkipTest("A CUDA GPU is required")
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

    def test_cpu_agents_collect_and_update_with_gpu_halfcheetah(self):
        env = make_vector_env(
            "HalfCheetah-v5", 4, backend="mjwarp", tensors=True, max_episode_steps=3
        )
        try:
            for algorithm in ("sac", "td3", "ddpg"):
                with self.subTest(algorithm=algorithm):
                    policy = make_policy(
                        algorithm,
                        env.single_observation_space,
                        env.single_action_space,
                        units=(16, 16),
                        batch_size=4,
                        memory_capacity=128,
                        n_warmup=4,
                    )
                    trainer = VectorOffPolicyTrainer(policy, env, updates_per_transition=0.25)
                    before = [v.numpy().copy() for v in policy.actor.trainable_variables]
                    for _ in range(4):
                        trainer.collect_step()
                        trainer.update()
                    self.assertEqual(trainer.total_updates, 3)
                    self.assertTrue(
                        any(
                            not np.array_equal(old, variable.numpy())
                            for old, variable in zip(before, policy.actor.trainable_variables)
                        )
                    )
                    self.assertIn("GPU:0", trainer.obs.device)
                    for variable in policy.actor.variables:
                        self.assertIn("CPU:0", variable.value.device)
        finally:
            env.close()

    def test_gpu_dac_terminal_timeout_and_expert_roundtrip(self):
        env = make_vector_env(
            "InvertedPendulum-v5",
            3,
            backend="mjwarp",
            tensors=True,
            absorbing=True,
            max_episode_steps=4,
        )
        scalar = AbsorbingWrapper(gym.make("InvertedPendulum-v5"), indicator=0.1)
        try:
            policy = make_policy(
                "sac",
                env.single_observation_space,
                env.single_action_space,
                absorbing=True,
                xla=True,
                units=(16, 16),
                batch_size=3,
                n_warmup=0,
            )
            trainer = VectorOffPolicyTrainer(
                policy, env, absorbing=True, updates_per_transition=1 / 3
            )
            raw = env.env
            positions = raw.sim.data.qpos.numpy()
            positions[[0, 2], 1] = 0.3
            raw.sim.data.qpos.assign(positions)
            raw._steps.assign(np.array([0, 3, 3], np.int32))
            stats = trainer.collect_step()
            self.assertEqual(stats["terminations"], 2)
            self.assertEqual(stats["truncations"], 2)
            self.assertEqual(trainer.replay_rows, 5)
            batch = trainer.replay_buffer.get_all_transitions()
            np.testing.assert_array_equal(batch["done"][:, 0], [0, 1, 0, -1, -1])
            self.assertEqual(trainer.update()["updates"], 1)
            with tempfile.TemporaryDirectory() as directory:
                result = evaluate_vector_policy(policy, env, 3, output_dir=directory)
                loaded = load_trajectories_with_logps(result["paths"], env=scalar)
                self.assertEqual(loaded["obses"].shape[1], 5)
                for filename in result["paths"]:
                    path = joblib.load(filename)
                    np.testing.assert_allclose(
                        path["logp"],
                        policy.compute_action_log_probs(path["obs"], path["act"]).numpy(),
                        rtol=1e-5,
                        atol=1e-5,
                    )
            env.check_overflow()
        finally:
            env.close()
            scalar.close()


if __name__ == "__main__":
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    unittest.main()
