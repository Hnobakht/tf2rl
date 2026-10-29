import numpy as np
from scipy.signal import lfilter


def discount_cumsum(x, discount):
    """Forked from rllab for computing discounted cumulative sums of vectors.

    Args:
        x: np.ndarray or tf.Tensor
            Vector of inputs
        discount: float
            Discount factor

    Returns:
        Discounted cumulative summation. If input is [x0, x1, x2], then the output is:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
    """
    assert isinstance(x, np.ndarray) and x.ndim == 1
    return lfilter(b=[1], a=[1, float(-discount)], x=x[::-1], axis=0)[::-1]


import tensorflow as tf


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
