import numpy as np
import tensorflow as tf

from tf2rl.algos.rl.ddpg import DDPG
from tf2rl.utils.tensorflow import update_target_variables


class BiResDDPG(DDPG):
    """
    This is an implementation of Bi-Res-DDPG
    https://arxiv.org/abs/1905.01072
    """

    def __init__(self, eta=0.05, name="BiResDDPG", **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters.

        Args:
            eta (float): Weight assigned to the residual-gradient term in BiResDDPG.
            name (str): Name assigned to the model, operation, or registered component.
            **kwargs (dict): Additional keyword arguments forwarded to DDPG construction.

        Returns:
            None: Initializes the instance.
        """
        kwargs["name"] = name
        super().__init__(**kwargs)
        self._eta = eta

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        """Apply actor and critic gradient updates, update target networks, and return
        diagnostics.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.
            rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
                have a trailing singleton axis.
            dones (array_like): True MDP termination flags aligned with the replay batch; time-limit
                truncations should remain zero.
            weights (array_like or None): Per-transition importance-sampling weights, shaped (batch,) or
                (batch, 1); public update methods replace None with unit weights.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Actor loss, critic loss, and sum of absolute
                forward/backward residuals per sample.
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors1, td_errors2 = self._compute_td_error_body(
                    states, actions, next_states, rewards, dones)
                critic_loss = tf.reduce_mean(
                    tf.square(td_errors1) * weights +
                    tf.square(td_errors2) * weights * self.discount * self._eta)

            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic(states, next_action))

            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            update_target_variables(
                self.critic_target.weights, self.critic.weights, self.tau)
            update_target_variables(
                self.actor_target.weights, self.actor.weights, self.tau)

            return actor_loss, critic_loss, tf.abs(td_errors1) + tf.abs(td_errors2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """Return temporal-difference errors for the supplied transition batch.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.
            rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
                have a trailing singleton axis.
            dones (array_like): True MDP termination flags aligned with the replay batch; time-limit
                truncations should remain zero.

        Returns:
            np.ndarray: Sum of absolute residuals from both BiRes objectives, with singleton dimensions
                squeezed.
        """
        td_error1, td_error2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(np.abs(td_error1.numpy()) + np.abs(td_error2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        """Compute batched temporal-difference errors using the target networks.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.
            rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
                have a trailing singleton axis.
            dones (array_like): True MDP termination flags aligned with the replay batch; time-limit
                truncations should remain zero.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The two signed Bellman residuals used by the bidirectional
                residual objective.
        """
        with tf.device(self.device):
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            # Compute standard TD error
            target_Q = self.critic_target(next_states, self.actor_target(next_states))
            target_Q = rewards + (not_dones * self.discount * target_Q)
            target_Q = tf.stop_gradient(target_Q)
            current_Q = self.critic(states, actions)
            td_errors1 = target_Q - current_Q
            # Compute residual TD error
            next_actions = tf.stop_gradient(self.actor(next_states))
            target_Q = self.critic(next_states, next_actions)
            target_Q = rewards + (not_dones * self.discount * target_Q)
            current_Q = tf.stop_gradient(self.critic_target(states, actions))
            td_errors2 = target_Q - current_Q
        return td_errors1, td_errors2

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = DDPG.get_argument(parser)
        parser.add_argument('--eta', type=float, default=0.05)
        return parser
