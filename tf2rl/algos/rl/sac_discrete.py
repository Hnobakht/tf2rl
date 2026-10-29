import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.rl.sac import SAC
from tf2rl.networks.actors import CategoricalActor
from tf2rl.utils.tensorflow import huber_loss, update_target_variables


class CriticQ(tf.keras.Model):
    """
    Compared with original (continuous) version of SAC, the output of Q-function moves
        from Q: S x A -> R
        to   Q: S -> R^|A|
    """

    def __init__(self, state_shape, action_dim, critic_units=(256, 256), name='qf'):
        """Build the hidden state-action layers and scalar Q-value head."""
        super().__init__(name=name)

        self.l1 = Dense(critic_units[0], name="L1", activation='relu')
        self.l2 = Dense(critic_units[1], name="L2", activation='relu')
        self.l3 = Dense(action_dim, name="L2", activation='linear')

        dummy_state = tf.constant(
            np.zeros(shape=(1,) + state_shape, dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        """Return Q-value estimates for a batch of state-action pairs."""
        features = self.l1(states)
        features = self.l2(features)
        values = self.l3(features)

        return values


class SACDiscrete(SAC):
    """Soft actor-critic for categorical action spaces."""
    def __init__(
            self,
            state_shape,
            action_dim,
            *args,
            actor_fn=None,
            critic_fn=None,
            target_update_interval=None,
            **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters."""
        kwargs["name"] = "SAC_discrete"
        self.actor_fn = actor_fn if actor_fn is not None else CategoricalActor
        self.critic_fn = critic_fn if critic_fn is not None else CriticQ
        self.target_hard_update = target_update_interval is not None
        self.target_update_interval = target_update_interval
        self.n_training = tf.Variable(0, dtype=tf.int32)
        super().__init__(state_shape, action_dim, *args, **kwargs)
        if self.auto_alpha:
            # Referring author's implementation of original paper:
            # https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.pyhttps://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py
            self.target_alpha = -np.log((1.0 / action_dim)) * 0.98

    def _setup_actor(self, state_shape, action_dim, actor_units, lr, max_action=1.):
        # The output of actor is categorical distribution
        """Create the action policy and its optimizer."""
        self.actor = self.actor_fn(
            state_shape, action_dim, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_q(self, state_shape, action_dim, critic_units, lr):
        """Create Q-value networks, initialize their targets, and configure optimization."""
        self.qf1 = self.critic_fn(state_shape, action_dim, critic_units, name="qf1")
        self.qf2 = self.critic_fn(state_shape, action_dim, critic_units, name="qf2")
        self.qf1_target = self.critic_fn(state_shape, action_dim, critic_units, name="qf1_target")
        self.qf2_target = self.critic_fn(state_shape, action_dim, critic_units, name="qf2_target")
        update_target_variables(self.qf1_target.weights,
                                self.qf1.weights, tau=1.)
        update_target_variables(self.qf2_target.weights,
                                self.qf2.weights, tau=1.)
        self.qf1_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.qf2_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def _setup_critic_v(self, *args, **kwargs):
        """
        Do not need state-value function because it can be directly computed from Q-function.
        See Eq.(10) in the paper.
        """
        pass

    def train(self, states, actions, next_states, rewards, dones, weights=None):
        """Update the actor and critics from a replay minibatch and return training diagnostics."""
        if weights is None:
            weights = np.ones_like(rewards)

        td_errors, actor_loss, mean_ent, logp_min, logp_max, logp_mean = self._train_body(
            states, actions, next_states, rewards, dones, weights)

        tf.summary.scalar(name=self.policy_name + "/actor_loss", data=actor_loss)
        tf.summary.scalar(name=self.policy_name + "/critic_loss", data=td_errors)
        tf.summary.scalar(name=self.policy_name + "/mean_ent", data=mean_ent)
        tf.summary.scalar(name=self.policy_name + "/logp_meab", data=logp_mean)
        if self.auto_alpha:
            tf.summary.scalar(name=self.policy_name + "/log_ent", data=self.log_alpha)
            tf.summary.scalar(name=self.policy_name+"/logp_mean+target", data=logp_mean+self.target_alpha)
        tf.summary.scalar(name=self.policy_name + "/ent", data=self.alpha)

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        """Apply actor and critic gradient updates, update target networks, and return diagnostics.
        """
        with tf.device(self.device):
            batch_size = states.shape[0]
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            actions = tf.cast(actions, dtype=tf.int32)

            indices = tf.concat(
                values=[tf.expand_dims(tf.range(batch_size), axis=1),
                        actions], axis=1)

            with tf.GradientTape(persistent=True) as tape:
                # Compute critic loss
                next_action_prob = self.actor.compute_prob(next_states)
                next_action_logp = tf.math.log(next_action_prob + 1e-8)
                next_q = tf.minimum(
                    self.qf1_target(next_states), self.qf2_target(next_states))

                # Compute state value function V by directly computes expectation
                target_q = tf.expand_dims(tf.einsum(
                    'ij,ij->i', next_action_prob, next_q - self.alpha * next_action_logp), axis=1)  # Eq.(10)
                target_q = tf.stop_gradient(
                    rewards + not_dones * self.discount * target_q)

                current_q1 = self.qf1(states)
                current_q2 = self.qf2(states)

                td_loss1 = tf.reduce_mean(huber_loss(
                    target_q - tf.expand_dims(tf.gather_nd(current_q1, indices), axis=1),
                    delta=self.max_grad) * weights)
                td_loss2 = tf.reduce_mean(huber_loss(
                    target_q - tf.expand_dims(tf.gather_nd(current_q2, indices), axis=1),
                    delta=self.max_grad) * weights)  # Eq.(7)

                # Compute actor loss
                current_action_prob = self.actor.compute_prob(states)
                current_action_logp = tf.math.log(current_action_prob + 1e-8)

                policy_loss = tf.reduce_mean(
                    tf.einsum('ij,ij->i', current_action_prob,
                              self.alpha * current_action_logp - tf.stop_gradient(
                                  tf.minimum(current_q1, current_q2))) * weights)  # Eq.(12)
                mean_ent = tf.reduce_mean(
                    tf.einsum('ij,ij->i', current_action_prob, current_action_logp)) * (-1)

                if self.auto_alpha:
                    alpha_loss = -tf.reduce_mean(
                        (self.log_alpha * tf.stop_gradient(current_action_logp + self.target_alpha)))

            q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
            self.qf1_optimizer.apply_gradients(
                zip(q1_grad, self.qf1.trainable_variables))
            q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
            self.qf2_optimizer.apply_gradients(
                zip(q2_grad, self.qf2.trainable_variables))

            if self.target_hard_update:
                if self.n_training % self.target_update_interval == 0:
                    update_target_variables(self.qf1_target.weights,
                                            self.qf1.weights, tau=1.)
                    update_target_variables(self.qf2_target.weights,
                                            self.qf2.weights, tau=1.)
            else:
                update_target_variables(self.qf1_target.weights,
                                        self.qf1.weights, tau=self.tau)
                update_target_variables(self.qf2_target.weights,
                                        self.qf2.weights, tau=self.tau)

            actor_grad = tape.gradient(
                policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            if self.auto_alpha:
                alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
                self.alpha_optimizer.apply_gradients(
                    zip(alpha_grad, [self.log_alpha]))
                self.alpha.assign(tf.exp(self.log_alpha))

        return ((td_loss1 + td_loss2) / 2.,
                policy_loss, mean_ent,
                tf.reduce_min(current_action_logp),
                tf.reduce_max(current_action_logp),
                tf.reduce_mean(current_action_logp))

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        """Return temporal-difference errors for the supplied transition batch."""
        td_errors_q1, td_errors_q2 = self._compute_td_error_body(
            states, actions, next_states, rewards, dones)
        return np.squeeze(
            np.abs(td_errors_q1.numpy()) +
            np.abs(td_errors_q2.numpy()))

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        """Compute batched temporal-difference errors using the target networks."""
        with tf.device(self.device):
            batch_size = states.shape[0]
            not_dones = 1. - tf.cast(dones, dtype=tf.float32)
            actions = tf.cast(actions, dtype=tf.int32)

            indices = tf.concat(
                values=[tf.expand_dims(tf.range(batch_size), axis=1),
                        actions], axis=1)

            next_action_prob = self.actor.compute_prob(next_states)
            next_action_logp = tf.math.log(next_action_prob + 1e-8)
            next_q = tf.minimum(
                self.qf1_target(next_states), self.qf2_target(next_states))

            target_q = tf.expand_dims(tf.einsum(
                'ij,ij->i', next_action_prob, next_q - self.alpha * next_action_logp), axis=1)  # Eq.(10)
            target_q = tf.stop_gradient(
                rewards + not_dones * self.discount * target_q)

            current_q1 = self.qf1(states)
            current_q2 = self.qf2(states)

            td_errors_q1 = target_q - tf.expand_dims(
                tf.gather_nd(current_q1, indices), axis=1)
            td_errors_q2 = target_q - tf.expand_dims(
                tf.gather_nd(current_q2, indices), axis=1)  # Eq.(7)

        return td_errors_q1, td_errors_q2

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = SAC.get_argument(parser)
        parser.add_argument('--target-update-interval', type=int, default=None)
        return parser
