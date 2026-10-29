"""Stateful environment collection. No optimizer updates or experiment setup."""

import numpy as np
import tensorflow as tf

from tf2rl.algos.base import OffPolicyAgent, OnPolicyAgent
from tf2rl.data.transitions import dac_batch_transitions, dac_transitions


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


class StepCollector:
    """Collect vector transitions with physical final observations."""

    def __init__(
        self,
        policy,
        env,
        *,
        updates_per_transition=1.0,
        absorbing=False,
        seed=0,
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
        self.obs, _ = env.reset(seed=seed)
        self.total_steps = 0
        self.episode_returns = np.zeros(env.num_envs, np.float64)
        self.episode_lengths = np.zeros(env.num_envs, np.int64)

    def collect(self):
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
        stats["learning_transitions"] = int((~warmup).sum())
        return rows, stats


class RolloutCollector:
    """Collect tensor rollouts for PPO/VPG without host copies."""

    def __init__(self, policy, env, rollout_steps, seed=None):
        if not isinstance(policy, OnPolicyAgent):
            raise TypeError("Vector on-policy collection requires an OnPolicyAgent")
        self.policy = policy
        self.env = env
        self.rollout_steps = rollout_steps
        self.count = env.num_envs * rollout_steps
        if self.count != policy.horizon or self.count % policy.batch_size:
            raise ValueError(
                "Policy horizon must equal num_envs * rollout_steps and be divisible by batch_size"
            )
        self.obs, _ = env.reset(seed=seed)
        policy.initialize_tensor_training(self.obs)
        self.episode_returns = tf.zeros(env.num_envs, tf.float32)

    def collect(self):
        rows = [[] for _ in range(8)]
        return_sum = tf.constant(0.0)
        episodes = tf.constant(0)
        for _ in range(self.rollout_steps):
            action, logp, value = self.policy.get_action_and_val_tensor(self.obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            for column, value_to_store in zip(
                rows,
                (self.obs, action, logp, value, reward, terminated, truncated, info["final_obs"]),
            ):
                column.append(value_to_store)
            done = tf.logical_or(terminated, truncated)
            self.episode_returns += reward
            return_sum += tf.reduce_sum(tf.where(done, self.episode_returns, 0.0))
            episodes += tf.reduce_sum(tf.cast(done, tf.int32))
            self.episode_returns = tf.where(done, 0.0, self.episode_returns)
            self.obs = next_obs
        rollout = tuple(tf.stack(column) for column in rows)
        tf.debugging.assert_all_finite(rollout[4], "Nonfinite environment rewards")
        stats = {
            "mean_step_reward": float(tf.reduce_mean(rollout[4]).numpy()),
            "return_sum": float(return_sum.numpy()),
            "episodes": int(episodes.numpy()),
        }
        return rollout, stats


class SingleStepCollector:
    """Single-environment collection for expert and historical imitation recipes."""

    def __init__(
        self,
        policy,
        env,
        horizon,
        observation_transform,
        *,
        absorbing=False,
        mask_encoding="done",
        record_logp=False,
    ):
        self.policy, self.env = policy, env
        self.horizon = horizon
        self.transform = observation_transform
        self.absorbing, self.mask_encoding = absorbing, mask_encoding
        self.record_logp = record_logp
        self.reset_episode()

    def reset_episode(self):
        self.obs, _ = self.env.reset()
        self.episode_steps, self.episode_return = 0, 0.0

    def collect(self, step):
        obs = self.transform(np.asarray(self.obs, np.float32), update=True)
        if step - 1 < self.policy.n_warmup:
            action = self.env.action_space.sample()
            logp = self.policy.get_logp(obs) if self.record_logp else None
        else:
            result = self.policy.get_action(obs)
            action = result[0] if isinstance(result, tuple) else result
            logp = result[1] if self.record_logp else None
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        self.episode_steps += 1
        truncated = truncated or self.episode_steps >= self.horizon
        final = self.transform(np.asarray(next_obs, np.float32))
        if self.mask_encoding == "dac" or self.record_logp:
            rows = dac_transitions(
                self.env,
                obs,
                action,
                final,
                reward,
                terminated,
                truncated,
                absorbing=self.absorbing,
                logp=logp,
            )
            if self.mask_encoding == "done":
                for row in rows:
                    row["done"] = 1.0 - row["done"]
        else:
            rows = [dict(obs=obs, act=action, next_obs=final, rew=reward, done=float(terminated))]
        self.obs = next_obs
        self.episode_return += reward
        return rows, {"boundary": terminated or truncated, "episode_return": self.episode_return}
