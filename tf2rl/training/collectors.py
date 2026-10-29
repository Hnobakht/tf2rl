"""Stateful environment collection. No optimizer updates or experiment setup."""

import numpy as np
import tensorflow as tf

from tf2rl.algos.base import OffPolicyAgent, OnPolicyAgent
from tf2rl.data.transitions import dac_batch_transitions, dac_transitions


def numpy(value, dtype=np.float32):
    """Convert eager tensors and arrays to replay-compatible host arrays.

    Args:
        value (array_like): Value to convert, normalize, or store in the representation required by
            this operation.
        dtype (numpy.dtype or type): Requested array element type.

    Returns:
        np.ndarray: Host array converted to the requested dtype; eager tensors are copied from their
            device if necessary.
    """
    return np.asarray(value.numpy() if tf.is_tensor(value) else value, dtype=dtype)


def physical_next_observations(next_obs, terminated, truncated, info):
    """Replace autoreset states with the last physical states, supporting either backend.

    Args:
        next_obs (array_like): Physical successor observation or observation batch, before episode
            autoreset.
        terminated (array_like): True MDP termination flags; time-limit truncations must be supplied
            separately.
        truncated (array_like): Time-limit or external episode-boundary flags; these stop return
            recursion but permit value bootstrapping.
        info (dict): Vector-step metadata, including final physical observations before autoreset.

    Returns:
        np.ndarray: Copy of next_obs with boundary rows replaced by info["final_obs"], shaped
            (environments, *observation_shape).

    Raises:
        ValueError: Same-step autoreset must supply final_obs for completed worlds.
    """
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
        """Initialize StepCollector and its configuration.

        Construction resets the vector environment and initializes warmup, episode, and
        physical-step counters. Optimizer updates and replay insertion remain the trainer
        responsibility.

        Args:
            policy (OffPolicyAgent): Agent whose replay encoding and absorbing mode match the collector
                environment.
            env (gym.vector.VectorEnv): Same-step-autoreset environment exposing physical final
                observations for finished worlds.
            updates_per_transition (float): Gradient minibatch updates budgeted per physical transition
                after random warmup.
            absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
                interpretation.
            seed (int or None): Random seed; None leaves initialization to the underlying random
                generator.

        Returns:
            None: Initializes the instance.

        Raises:
            TypeError: Vector off-policy collection requires an OffPolicyAgent.
            ValueError: updates_per_transition must be finite and nonnegative. Replay capacity must be
                at least batch_size. Unsupported replay mask encoding. Absorbing-state training needs an
                absorbing vector wrapper and DAC-aware policy. Collector absorbing mode must match the
                policy's target semantics.
        """
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
        """Advance all worlds once and assemble replay rows with physical final observations.

        Returns:
            tuple[dict, dict]: Replay-ready arrays (obs, act, next_obs, rew, done) and collection
                counts/sums, including learning_transitions after warmup. This method advances collector
                state but does not insert rows into replay.

        Raises:
            FloatingPointError: Nonfinite off-policy environment transition.
        """
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
        """Initialize RolloutCollector and its configuration.

        Construction resets all worlds and builds the policy tensor inference/training state.
        The full rollout size must equal policy.horizon.

        Args:
            policy (OnPolicyAgent): Agent supporting batched action/log-density/value inference and
                tensor minibatch learning.
            env (gym.vector.VectorEnv): Tensor-backed vector environment returning dense physical final
                observations in info["final_obs"].
            rollout_steps (int): Number of sequential steps collected per environment in one rollout.
            seed (int or None): Random seed; None leaves initialization to the underlying random
                generator.

        Returns:
            None: Initializes the instance.

        Raises:
            TypeError: Vector on-policy collection requires an OnPolicyAgent.
            ValueError: Policy horizon must equal num_envs * rollout_steps and be divisible by
                batch_size.
        """
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
        """Collect one fixed-length vector rollout, retaining physical final observations and
        episode statistics.

        Returns:
            tuple[tuple[tf.Tensor, ...], dict]: Eight time-major tensors (states, actions, logps,
                values, rewards, terminated, truncated, final_obs) and collection statistics. Each
                tensor starts with (rollout_steps, environments).
        """
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
        """Initialize SingleStepCollector and its configuration.

        Construction resets the environment through reset_episode. The caller must reset again
        after collect reports an episode boundary.

        Args:
            policy (Policy): Agent supplying action selection and learning operations.
            env (gym.Env): Scalar Gymnasium environment; absorbing collection requires the absorbing-
                state helper methods.
            horizon (int): Number of transitions in one on-policy rollout batch; must be divisible by
                batch_size.
            observation_transform (callable): Function accepting an observation and update flag and
                returning the policy observation.
            absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
                interpretation.
            mask_encoding (str or None): Replay convention: done for termination flags or dac for masks
                (1, 0, -1).
            record_logp (bool): Store collection-time action densities, including the historical warmup
                convention.

        Returns:
            None: Initializes the instance.
        """
        self.policy, self.env = policy, env
        self.horizon = horizon
        self.transform = observation_transform
        self.absorbing, self.mask_encoding = absorbing, mask_encoding
        self.record_logp = record_logp
        self.reset_episode()

    def reset_episode(self):
        """Reset the scalar environment, episode return, and episode-length counters.

        Returns:
            None.
        """
        self.obs, _ = self.env.reset()
        self.episode_steps, self.episode_return = 0, 0.0

    def collect(self, step):
        """Advance one physical scalar transition and assemble its replay rows without
        resetting the episode.

        Args:
            step (int or None): Physical environment-step counter used for warmup, summaries, or
                checkpoint numbering.

        Returns:
            tuple[list[dict], dict]: Replay rows and boundary/episode_return statistics. Rows include
                any absorbing loop and optional logp; resetting a finished episode is the caller
                responsibility.
        """
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
