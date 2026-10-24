"""On-policy vector rollouts for CPU or GPU environment backends.

The environment implements the Gymnasium same-step vector API; the policy
provides tensor inference, tensor minibatch updates, and a value function.
No task-specific dimensions, rewards or simulation calls belong in this trainer.
"""

import tensorflow as tf

from tf2rl.algos.policy_base import OnPolicyAgent


@tf.function(reduce_retracing=True)
def rollout_targets(
    rewards, values, next_values, terminated, truncated, discount, lam, enable_gae=True
):
    """Per-world advantages and discounted rewards-to-go with correct boundaries.

    Truncations bootstrap their final observation; true terminations bootstrap
    zero. Both stop recursion so returns never leak into an autoreset episode.
    The final rollout step bootstraps even if no episode boundary was reached.

    All inputs have shape ``(time, worlds)``. With ``b=1-terminated`` and
    ``c=1-(terminated or truncated)``, the TD residual is
    ``delta_t=r_t+gamma*b_t*V(next_t)-V_t`` and GAE obeys
    ``A_t=delta_t+gamma*lambda*c_t*A_(t+1)``. ``next_values`` must use physical
    final observations, never reset observations. See Schulman et al. (2016),
    https://arxiv.org/abs/1506.02438.

    Value regression uses bootstrapped discounted rewards-to-go, not ``A+V``;
    these differ for lambda < 1. Disabling GAE selects one-step TD advantages.
    """
    bootstrap = (1.0 - tf.cast(terminated, tf.float32)) * next_values
    boundary = tf.logical_or(terminated, truncated)
    continuation = 1.0 - tf.cast(boundary, tf.float32)
    deltas = rewards + discount * bootstrap - values
    if enable_gae:
        advantages = tf.scan(
            lambda carry, item: item[0] + discount * lam * item[1] * carry,
            (deltas, continuation),
            initializer=tf.zeros_like(values[0]),
            reverse=True,
        )
    else:
        advantages = deltas
    # Preserve the existing OnPolicyTrainer's discounted return targets.
    boundary_value = tf.where(boundary, bootstrap, tf.zeros_like(bootstrap))
    returns = tf.scan(
        lambda carry, item: item[0] + discount * (item[1] + item[2] * carry),
        (rewards, boundary_value, continuation),
        initializer=bootstrap[-1],
        reverse=True,
    )
    return advantages, returns


class VectorOnPolicyTrainer:
    """Train on tensor vector rollouts using an existing on-policy agent."""

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

    def update(self, *rollout):
        """Place the rollout on the learner device before preprocessing and updates."""
        with tf.device(self.policy.device):
            rollout = tuple(tf.identity(value) for value in rollout)
            return self._update(*rollout)

    @tf.function(reduce_retracing=True)
    def _update(self, states, actions, logps, values, rewards, terminated, truncated, final_obs):
        policy = self.policy
        state_shape = self.env.single_observation_space.shape
        action_shape = self.env.single_action_space.shape
        states = tf.reshape(states, (self.count, *state_shape))
        actions = tf.reshape(actions, (self.count, *action_shape))
        logps = tf.reshape(logps, (self.count,))
        next_values = tf.reshape(
            policy.get_value_tensor(tf.reshape(final_obs, (self.count, *state_shape))),
            tf.shape(values),
        )
        advantages, returns = rollout_targets(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            policy.discount,
            policy.lam,
            policy.enable_gae,
        )
        advantages = tf.reshape(advantages, (self.count,))
        returns = tf.reshape(returns, (self.count,))
        if policy.normalize_adv:
            mean, variance = tf.nn.moments(advantages, axes=[0])
            advantages = (advantages - mean) / tf.sqrt(variance + 1e-8)
        tf.debugging.assert_all_finite(states, "Nonfinite rollout observations")
        tf.debugging.assert_all_finite(returns, "Nonfinite return targets")
        diagnostics = tf.zeros(3)
        updates = tf.constant(0)
        stop = tf.constant(False)
        for _ in tf.range(policy.n_epoch):
            if not stop:
                order = tf.random.shuffle(tf.range(self.count))
                for offset in tf.range(0, self.count, policy.batch_size):
                    if not stop:
                        index = order[offset : offset + policy.batch_size]
                        result = policy.train_tensor(
                            tf.gather(states, index),
                            tf.gather(actions, index),
                            tf.gather(advantages, index),
                            tf.gather(logps, index),
                            tf.gather(returns, index),
                        )
                        tf.debugging.assert_all_finite(result, "Nonfinite policy losses")
                        diagnostics += result
                        updates += 1
                        stop = policy.should_stop_update(result)
        return diagnostics / tf.cast(updates, tf.float32), updates
