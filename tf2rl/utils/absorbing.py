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
        """Initialize ReplaySemantics and its configuration.

        Args:
            absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
                interpretation.
            mask_encoding (str or None): Replay convention: done for termination flags or dac for masks
                (1, 0, -1).

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: Replay mask encoding must be 'done' or 'dac'. Absorbing-state training requires
                DAC replay masks.
        """
        self.absorbing = bool(absorbing)
        self.mask_encoding = mask_encoding or ("dac" if absorbing else "done")
        if self.mask_encoding not in ("done", "dac"):
            raise ValueError("Replay mask encoding must be 'done' or 'dac'")
        if self.absorbing and self.mask_encoding != "dac":
            raise ValueError("Absorbing-state training requires DAC replay masks")

    def masks(self, dones):
        """Return flat DAC masks, converting ordinary termination flags if needed.

        Args:
            dones (array_like): Replay termination indicators. DAC-aware agents instead accept masks 1
                (continuing), 0 (terminal), and -1 (absorbing loop).

        Returns:
            tf.Tensor: Flat float32 DAC masks of shape (batch,), converting done flags to 1-done when
                needed.
        """
        values = tf.reshape(tf.cast(dones, tf.float32), (-1,))
        return values if self.mask_encoding == "dac" else 1.0 - values

    def next_actions(self, actions, dones):
        """Use zero action at absorbing successors, after any target-policy noise.

        Args:
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            dones (array_like): Replay termination indicators. DAC-aware agents instead accept masks 1
                (continuing), 0 (terminal), and -1 (absorbing loop).

        Returns:
            tf.Tensor: Actions with the original shape; terminal and loop successors use zero actions in
                absorbing mode.
        """
        if not self.absorbing:
            return actions
        return tf.where(self.masks(dones)[:, None] > 0, actions, tf.zeros_like(actions))

    def target(self, rewards, next_q, dones, discount, entropy=0.0):
        """Build aligned, flat Bellman targets; entropy is alpha*logpi for SAC.

        Args:
            rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
                have a trailing singleton axis.
            next_q (array_like): Q estimates at successor states, one per transition.
            dones (array_like): Replay termination indicators. DAC-aware agents instead accept masks 1
                (continuing), 0 (terminal), and -1 (absorbing loop).
            discount (float): Reward discount factor gamma.
            entropy (array_like or float): SAC alpha times successor log probability; zero for
                algorithms without an entropy term.

        Returns:
            tf.Tensor: Flat float32 Bellman targets of shape (batch,). Absorbing mode bootstraps all
                rows but removes entropy at absorbing successors.
        """
        rewards = tf.reshape(tf.cast(rewards, tf.float32), (-1,))
        next_q = tf.reshape(tf.cast(next_q, tf.float32), (-1,))
        entropy = tf.reshape(tf.cast(entropy, tf.float32), (-1,))
        continuation = tf.maximum(self.masks(dones), 0.0)
        if self.absorbing:
            entropy = tf.where(continuation > 0, entropy, tf.zeros_like(entropy))
            return rewards + discount * (next_q - entropy)
        return rewards + discount * continuation * (next_q - entropy)

    def actor_mean(self, values, dones):
        """Average per-row actor losses over physical transitions only.

        Args:
            values (array_like): Values aligned with observations or transitions.
            dones (array_like): Replay termination indicators. DAC-aware agents instead accept masks 1
                (continuing), 0 (terminal), and -1 (absorbing loop).

        Returns:
            tf.Tensor: Scalar mean actor loss, excluding synthetic self-loops in absorbing mode; zero if
                no physical samples remain.
        """
        values = tf.reshape(values, (-1,))
        if not self.absorbing:
            return tf.reduce_mean(values)
        physical = self.masks(dones) >= 0
        return tf.math.divide_no_nan(
            tf.reduce_sum(tf.where(physical, values, tf.zeros_like(values))),
            tf.reduce_sum(tf.cast(physical, values.dtype)),
        )

    def has_actor_samples(self, dones):
        """Skip optimizer momentum updates when a batch contains only self-loops.

        Args:
            dones (array_like): Replay termination indicators. DAC-aware agents instead accept masks 1
                (continuing), 0 (terminal), and -1 (absorbing loop).

        Returns:
            bool or tf.Tensor: Whether at least one row contributes to the actor objective; always True
                outside absorbing mode.
        """
        return not self.absorbing or tf.reduce_any(self.masks(dones) >= 0)


def dac_target(rewards, masks, next_q, next_logp, alpha, discount, absorbing):
    """SAC entropy-regularized Bellman target for callers supplying DAC masks.

    Args:
        rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
            have a trailing singleton axis.
        masks (array_like): DAC masks: 1 for continuing, 0 for terminal, and -1 for absorbing self-
            loops.
        next_q (array_like): Q estimates at successor states, one per transition.
        next_logp (array_like): Successor-action log probabilities used by SAC entropy
            regularization.
        alpha (float): Entropy-temperature coefficient used by SAC.
        discount (float): Reward discount factor gamma.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        tf.Tensor: Flat Bellman targets under DAC mask semantics, with entropy omitted at absorbing
            successors.
    """
    return ReplaySemantics(absorbing, "dac").target(
        rewards, next_q, masks, discount, entropy=alpha * next_logp
    )


def dac_actor_loss(logp, q, masks, alpha, absorbing):
    """SAC actor objective for callers supplying DAC masks.

    Args:
        logp (array_like): Action log probability for each transition; also stored on synthetic rows
            when supplied.
        q (array_like): Current policy-action Q estimates, one per transition.
        masks (array_like): DAC masks: 1 for continuing, 0 for terminal, and -1 for absorbing self-
            loops.
        alpha (float): Entropy-temperature coefficient used by SAC.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        tf.Tensor: Scalar mean alpha*logp-Q over eligible actor samples.
    """
    return ReplaySemantics(absorbing, "dac").actor_mean(alpha * logp - q, masks)
