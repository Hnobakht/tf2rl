"""Shared off-policy Bellman and actor rules for optional DAC absorbing states.

Collection owns observation augmentation and synthetic transitions. These rules
only interpret replay rows; callers supply their algorithm's Q estimate and,
for entropy-regularized algorithms, the next-state entropy contribution.
"""

import tensorflow as tf


class ReplaySemantics:
    """Interpret boolean termination flags or DAC masks (1, 0, -1).

    Absorbing mode requires DAC masks. Terminal rows and self-loops bootstrap
    at zero action without entropy, giving Q_abs = r_abs / (1 - discount).
    Truncations remain continuing rows. Synthetic loops have no actor objective.
    """

    def __init__(self, absorbing=False, mask_encoding=None):
        self.absorbing = bool(absorbing)
        self.mask_encoding = mask_encoding or ("dac" if absorbing else "done")
        if self.mask_encoding not in ("done", "dac"):
            raise ValueError("Replay mask encoding must be 'done' or 'dac'")
        if self.absorbing and self.mask_encoding != "dac":
            raise ValueError("Absorbing-state training requires DAC replay masks")

    def masks(self, dones):
        """Return flat DAC masks, converting ordinary termination flags if needed."""
        values = tf.reshape(tf.cast(dones, tf.float32), (-1,))
        return values if self.mask_encoding == "dac" else 1.0 - values

    def next_actions(self, actions, dones):
        """Use zero action at absorbing successors, after any target-policy noise."""
        if not self.absorbing:
            return actions
        return tf.where(self.masks(dones)[:, None] > 0, actions, tf.zeros_like(actions))

    def target(self, rewards, next_q, dones, discount, entropy=0.0):
        """Build aligned, flat Bellman targets; entropy is alpha*logpi for SAC."""
        rewards = tf.reshape(tf.cast(rewards, tf.float32), (-1,))
        next_q = tf.reshape(tf.cast(next_q, tf.float32), (-1,))
        entropy = tf.reshape(tf.cast(entropy, tf.float32), (-1,))
        continuation = tf.maximum(self.masks(dones), 0.0)
        if self.absorbing:
            entropy = tf.where(continuation > 0, entropy, tf.zeros_like(entropy))
            return rewards + discount * (next_q - entropy)
        return rewards + discount * continuation * (next_q - entropy)

    def actor_mean(self, values, dones):
        """Average per-row actor losses over physical transitions only."""
        values = tf.reshape(values, (-1,))
        if not self.absorbing:
            return tf.reduce_mean(values)
        physical = self.masks(dones) >= 0
        return tf.math.divide_no_nan(
            tf.reduce_sum(tf.where(physical, values, tf.zeros_like(values))),
            tf.reduce_sum(tf.cast(physical, values.dtype)),
        )

    def has_actor_samples(self, dones):
        """Skip optimizer momentum updates when a batch contains only self-loops."""
        return not self.absorbing or tf.reduce_any(self.masks(dones) >= 0)


def dac_target(rewards, masks, next_q, next_logp, alpha, discount, absorbing):
    """SAC entropy-regularized Bellman target for callers supplying DAC masks."""
    return ReplaySemantics(absorbing, "dac").target(
        rewards, next_q, masks, discount, entropy=alpha * next_logp
    )


def dac_actor_loss(logp, q, masks, alpha, absorbing):
    """SAC actor objective for callers supplying DAC masks."""
    return ReplaySemantics(absorbing, "dac").actor_mean(alpha * logp - q, masks)
