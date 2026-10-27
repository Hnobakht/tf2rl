"""Algorithm-independent vector collection and uniform replay for off-policy agents."""

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from tf2rl.algos.policy_base import OffPolicyAgent
from tf2rl.misc.dac_ops import dac_batch_transitions
from tf2rl.misc.get_replay_buffer import get_replay_buffer


def numpy(value, dtype=np.float32):
    """Convert eager tensors and arrays to replay-compatible host arrays."""
    return np.asarray(value.numpy() if tf.is_tensor(value) else value, dtype=dtype)


def physical_next_observations(next_obs, terminated, truncated, info):
    """Replace autoreset states with the last physical states, supporting either backend."""
    final = numpy(next_obs).copy()
    boundary = numpy(terminated, bool) | numpy(truncated, bool)
    if boundary.any():
        if "final_obs" not in info:
            raise ValueError("Same-step autoreset must supply final_obs for completed worlds")
        source = info["final_obs"]
        # Native NumPy VectorEnv uses object arrays; tensor backends use dense batches.
        if isinstance(source, np.ndarray) and source.dtype == object:
            final[boundary] = np.stack(source[boundary])
        else:
            final[boundary] = numpy(source)[boundary]
    return final


class VectorOffPolicyTrainer:
    """Collect every world's transitions, then update from uniformly sampled replay.

    updates_per_transition counts gradient minibatch updates per new physical
    transition, not per vector step. Fractional credit survives collection calls.
    Synthetic absorbing loops never increase environment or update-budget counts.
    """

    def __init__(
        self,
        policy,
        env,
        *,
        updates_per_transition=1.0,
        absorbing=False,
        seed=0,
        replay_buffer=None,
    ):
        if not isinstance(policy, OffPolicyAgent):
            raise TypeError("Vector off-policy collection requires an OffPolicyAgent")
        if not np.isfinite(updates_per_transition) or updates_per_transition < 0:
            raise ValueError("updates_per_transition must be finite and nonnegative")
        if policy.memory_capacity < policy.batch_size:
            raise ValueError("Replay capacity must be at least batch_size")
        self.mask_encoding = getattr(policy, "replay_mask_encoding", "done")
        if self.mask_encoding not in ("done", "dac"):
            raise ValueError("Unsupported replay mask encoding")
        if absorbing and (self.mask_encoding != "dac" or not hasattr(env, "get_absorbing_state")):
            raise ValueError(
                "Absorbing-state training needs an absorbing vector wrapper and DAC-aware policy"
            )
        if bool(getattr(policy, "_is_absorbed_state", False)) != absorbing:
            raise ValueError("Collector absorbing mode must match the policy's target semantics")
        self.policy, self.env = policy, env
        self.absorbing = absorbing
        self.updates_per_transition = updates_per_transition
        self.rng = np.random.default_rng(seed)
        spaces = SimpleNamespace(
            observation_space=env.single_observation_space, action_space=env.single_action_space
        )
        self.replay_buffer = (
            get_replay_buffer(policy, spaces) if replay_buffer is None else replay_buffer
        )
        self.obs, _ = env.reset(seed=seed)
        self.total_steps = 0
        self.total_updates = 0
        self.replay_rows = 0
        self.update_credit = 0.0
        self.episode_returns = np.zeros(env.num_envs, np.float64)
        self.episode_lengths = np.zeros(env.num_envs, np.int64)

    def collect_step(self):
        """Advance all worlds once and insert a batch with correct termination semantics."""
        count = self.env.num_envs
        low, high = self.env.single_action_space.low, self.env.single_action_space.high
        warmup = self.total_steps + np.arange(count) < self.policy.n_warmup
        with tf.device(self.policy.device):
            if warmup.all():
                action = tf.convert_to_tensor(
                    self.rng.uniform(low, high, (count, *low.shape)), tf.float32
                )
            else:
                result = self.policy.get_action_tensor(tf.convert_to_tensor(self.obs, tf.float32))
                action = result[0] if isinstance(result, tuple) else result
                if warmup.any():
                    random = tf.convert_to_tensor(
                        self.rng.uniform(low, high, (count, *low.shape)), tf.float32
                    )
                    action = tf.where(warmup[:, None], random, action)
            action = tf.clip_by_value(action, low, high)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        terminated, truncated = numpy(terminated, bool), numpy(truncated, bool)
        final = physical_next_observations(next_obs, terminated, truncated, info)
        states, actions, rewards = numpy(self.obs), numpy(action), numpy(reward)
        if not all(np.isfinite(value).all() for value in (states, actions, final, rewards)):
            raise FloatingPointError("Nonfinite off-policy environment transition")
        if self.mask_encoding == "dac":
            rows = dac_batch_transitions(
                self.env, states, actions, final, rewards, terminated, absorbing=self.absorbing
            )
        else:
            rows = {
                "obs": states,
                "act": actions,
                "next_obs": final,
                "rew": rewards[:, None],
                "done": terminated.astype(np.float32)[:, None],
            }
        self.replay_buffer.add(**rows)
        self.replay_rows += len(rows["obs"])
        self.obs = next_obs
        self.episode_returns += rewards
        self.episode_lengths += 1
        boundary = terminated | truncated
        stats = {
            "transitions": count,
            "replay_rows": len(rows["obs"]),
            "reward_sum": float(rewards.sum()),
            "episodes": int(boundary.sum()),
            "return_sum": float(self.episode_returns[boundary].sum()),
            "length_sum": int(self.episode_lengths[boundary].sum()),
            "terminations": int(terminated.sum()),
            "truncations": int(truncated.sum()),
        }
        self.episode_returns[boundary] = 0
        self.episode_lengths[boundary] = 0
        self.total_steps += count
        self.update_credit += int((~warmup).sum()) * self.updates_per_transition
        return stats

    def update(self, record_summaries=False):
        """Spend accumulated update credit; return counts and a finite diagnostic."""
        if self.replay_buffer.get_stored_size() < self.policy.batch_size:
            return {"updates": 0, "mean_update_diagnostic": None}
        count = int(self.update_credit + 1e-9)
        diagnostics = []
        with tf.device(self.policy.device):
            for index in range(count):
                batch = self.replay_buffer.sample(self.policy.batch_size)
                with tf.summary.record_if(record_summaries and index == count - 1):
                    result = self.policy.train(
                        batch["obs"], batch["act"], batch["next_obs"], batch["rew"], batch["done"]
                    )
                values = numpy(result)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite off-policy learner diagnostic")
                diagnostics.append(float(values.mean()))
        self.update_credit -= count
        self.total_updates += count
        return {
            "updates": count,
            "mean_update_diagnostic": float(np.mean(diagnostics)) if diagnostics else None,
        }
