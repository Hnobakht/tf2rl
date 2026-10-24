import numpy as np
import tensorflow as tf

from tf2rl.algos.vpg import VPG


class PPO(VPG):
    """PPO with ``L = -E[min(r*A, clip(r,1-eps,1+eps)*A)]``.

    Here ``r=exp(logpi_new-logpi_old)`` and rollout advantages are held fixed.
    Clipping limits the surrogate incentive; it does not enforce a hard KL
    constraint. The sample mean ``logpi_old-logpi_new`` is a noisy estimate of
    forward KL and can be negative on finite batches. See Schulman et al.
    (2017), https://arxiv.org/abs/1707.06347.
    """
    def __init__(
            self,
            clip=True,
            clip_ratio=0.2,
            name="PPO",
            target_kl=0.075,
            **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters."""
        super().__init__(name=name, **kwargs)
        self.clip = clip
        self.clip_ratio = clip_ratio
        self.target_kl=target_kl
        assert self.horizon % self.batch_size == 0 , "The horizon % batch_size should be 0. [__init__ PPO]"

    def should_stop_update(self, diagnostics):
        """PPO stops its rollout update when the estimated KL exceeds its target."""
        return diagnostics[2] > self.target_kl

    @tf.function(reduce_retracing=True)
    def train_tensor(self, states, actions, advantages, logp_olds, returns):
        """Apply PPO losses without transferring minibatch diagnostics to the host."""
        if self.actor_critic is not None:
            actor_loss, critic_loss, _, _, _, kl = self._train_actor_critic_body(
                states, actions, advantages, logp_olds, returns)
            return tf.stack((actor_loss, critic_loss, kl))
        actor_loss, _, _, _, kl = self._train_actor_body(
            states, actions, advantages, logp_olds)
        critic_loss = self._train_critic_body(states, returns)
        return tf.stack([actor_loss, critic_loss, kl])

    def train(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        """Update the PPO policy and value function from a rollout batch."""
        if self.actor_critic is not None:
            actor_loss, critic_loss, logp_news, ratio, ent, kl = self._train_actor_critic_body(
                states, actions, advantages, logp_olds, returns)
        else:
            actor_loss, logp_news, ratio, ent, kl = self._train_actor_body(
                states, actions, advantages, logp_olds)
            critic_loss = self._train_critic_body(states, returns)
        # Visualize results in TensorBoard
        tf.summary.scalar(name=self.policy_name+"/actor_loss",
                          data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max",
                          data=np.max(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_min",
                          data=np.min(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_mean",
                          data=np.mean(logp_news))
        tf.summary.scalar(name=self.policy_name+"/adv_max",
                          data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min",
                          data=np.min(advantages))
        tf.summary.scalar(name=self.policy_name+"/kl",
                          data=kl)
        tf.summary.scalar(name=self.policy_name+"/ent",
                          data=ent)
        tf.summary.scalar(name=self.policy_name+"/ratio",
                          data=tf.reduce_mean(ratio))
        tf.summary.scalar(name=self.policy_name+"/critic_loss",
                          data=critic_loss)
        return actor_loss, critic_loss, kl.numpy()

    def train_actor(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        """Update the PPO policy from observations, actions, advantages, and old log probabilities.
        """
        if self.actor_critic is not None:
            raise NotImplementedError("Shared actor-critic requires a joint train() update")
        else:
            actor_loss, logp_news, ratio, ent, kl = self._train_actor_body(
                states, actions, advantages, logp_olds)
        # Visualize results in TensorBoard
        tf.summary.scalar(name=self.policy_name+"/actor_loss",
                          data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/logp_max",
                          data=np.max(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_min",
                          data=np.min(logp_news))
        tf.summary.scalar(name=self.policy_name+"/logp_mean",
                          data=np.mean(logp_news))
        tf.summary.scalar(name=self.policy_name+"/adv_max",
                          data=np.max(advantages))
        tf.summary.scalar(name=self.policy_name+"/adv_min",
                          data=np.min(advantages))
        tf.summary.scalar(name=self.policy_name+"/kl",
                          data=kl)
        tf.summary.scalar(name=self.policy_name+"/ent",
                          data=ent)
        tf.summary.scalar(name=self.policy_name+"/ratio",
                          data=tf.reduce_mean(ratio))
        return actor_loss, kl.numpy()

    def train_critic(self, states, returns):
        # Train actor and critic
        """Update the value function against the supplied return targets."""
        if self.actor_critic is not None:
            raise NotImplementedError("Shared actor-critic requires a joint train() update")
        else:
            critic_loss = self._train_critic_body(states, returns)
        # Visualize results in TensorBoard
        tf.summary.scalar(name=self.policy_name+"/critic_loss", data=critic_loss)
        return critic_loss

    @tf.function
    def _train_actor_critic_body(self, states, actions, advantages, logp_olds, returns):
        """Update the shared actor-critic model using PPO losses."""
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                _, _, current_V = self.actor_critic(states)
                ent = tf.constant(0., tf.float32)
                if self.entropy_coef > 0.:
                    ent = tf.reduce_mean(
                        self.actor_critic.compute_entropy(states))
                # Train actor
                if self.clip:
                    logp_news = self.actor_critic.compute_log_probs(
                        states, actions)
                    ratio = tf.math.exp(
                        logp_news - tf.squeeze(logp_olds))
                    min_adv = tf.clip_by_value(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio) * tf.squeeze(advantages)
                    actor_loss = -tf.reduce_mean(tf.minimum(
                        ratio * tf.squeeze(advantages),
                        min_adv))
                    if self.entropy_coef > 0.:
                        actor_loss -= self.entropy_coef * ent
                else:
                    raise NotImplementedError
                # Train critic
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(tf.square(td_errors))
                total_loss = actor_loss + self.vfunc_coef * critic_loss

            grads = tape.gradient(
                total_loss, self.actor_critic.trainable_variables)
            self.actor_critic_optimizer.apply_gradients(
                zip(grads, self.actor_critic.trainable_variables))
            kl = tf.reduce_mean(tf.squeeze(logp_olds) - logp_news)

        return actor_loss, critic_loss, logp_news, ratio, ent, kl

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):
        """Apply one policy-gradient update and return the optimization diagnostics."""
        with tf.device(self.device):
            # Update actor
            with tf.GradientTape() as tape:
                if self.clip:
                    logp_news = self.actor.compute_log_probs(states, actions)
                    ratio = tf.math.exp(logp_news - tf.squeeze(logp_olds))
                    min_adv = tf.where(tf.squeeze(advantages) > 0,
                                       (1 + self.clip_ratio) * tf.squeeze(advantages),
                                       (1 - self.clip_ratio) * tf.squeeze(advantages))
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * tf.squeeze(advantages), min_adv))
                    ent = tf.reduce_mean(-logp_news)
                    if self.entropy_coef > 0:
                        actor_loss -= self.entropy_coef * ent
                else:
                    raise NotImplementedError
            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))

            kl = tf.reduce_mean(tf.squeeze(logp_olds) - logp_news)
        return actor_loss, logp_news, ratio, ent, kl
