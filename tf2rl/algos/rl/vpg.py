import numpy as np
import tensorflow as tf

from tf2rl.algos.base import OnPolicyAgent
from tf2rl.envs.wrappers.atari import LazyFrames
from tf2rl.networks.actors import CategoricalActor, GaussianActor
from tf2rl.networks.critics import CriticV


class VPG(OnPolicyAgent):
    """Vanilla policy gradient with a learned state-value baseline."""
    def __init__(
            self,
            state_shape,
            action_dim,
            is_discrete,
            actor=None,
            critic=None,
            actor_critic=None,
            max_action=1.,
            actor_units=(256, 256),
            critic_units=(256, 256),
            lr_actor=1e-3,
            lr_critic=3e-3,
            hidden_activation_actor="relu",
            hidden_activation_critic="relu",
            name="VPG",
            **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            is_discrete (bool): Select a categorical policy instead of a continuous Gaussian policy.
            actor (keras.Model or None): Optional externally constructed actor network.
            critic (keras.Model or None): Optional externally constructed value network.
            actor_critic (keras.Model or None): Optional shared network returning actions, densities,
                and values.
            max_action (float): Symmetric continuous-action bound or output scale.
            actor_units (sequence[int]): Widths of the actor hidden layers.
            critic_units (sequence[int]): Widths of the critic hidden layers.
            lr_actor (float): Actor optimizer learning rate.
            lr_critic (float): Critic optimizer learning rate.
            hidden_activation_actor (str or callable): Activation used by actor hidden layers.
            hidden_activation_critic (str or callable): Activation used by critic hidden layers.
            name (str): Name assigned to the model, operation, or registered component.
            **kwargs (dict): Additional keyword arguments forwarded to OnPolicyAgent construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name, **kwargs)
        self._is_discrete = is_discrete

        # TODO: clean codes
        if actor_critic is not None:
            self.actor_critic = actor_critic
            self.actor_critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_actor)
            self.actor = None
            self.critic = None
        else:
            self.actor_critic = None
            if actor is None:
                with tf.device(self.device):
                    if is_discrete:
                        self.actor = CategoricalActor(
                            state_shape, action_dim, actor_units)
                    else:
                        self.actor = GaussianActor(
                            state_shape, action_dim, max_action, actor_units,
                            hidden_activation=hidden_activation_actor,
                            state_independent_std=True)
            else:
                self.actor = actor
            if critic is None:
                self.critic = CriticV(state_shape, critic_units,
                                      hidden_activation=hidden_activation_critic,
                                      device=self.device)
            else:
                self.critic = critic
            self.actor_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_actor)
            self.critic_optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr_critic)

        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

    def get_action(self, state, test=False):
        """Return actions and their log probabilities; test mode selects deterministic actions.

        Args:
            state (np.ndarray or LazyFrames): One observation or a batch. LazyFrames is materialized
                before the NumPy-array requirement is checked.
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sampled/evaluation actions and log probabilities. One-state
                actions are unbatched; the log-probability result retains its batch dimension.
        """
        if isinstance(state, LazyFrames):
            state = np.array(state)
        msg = "Input instance should be np.ndarray, not {}".format(type(state))
        assert isinstance(state, np.ndarray), msg

        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)
        action, logp = self._get_action_body(state, test)

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_and_val(self, state, test=False):
        """Return a policy action, its log probability, and the estimated state value.

        Args:
            state (np.ndarray or LazyFrames): One observation or batch; the observation rank determines
                whether action/value outputs are unbatched.
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            tuple: NumPy actions, action log probabilities, and values. For a single observation, action
                and value are unbatched while the log probability retains its batch dimension; batched
                input preserves all batch axes.
        """
        if isinstance(state, LazyFrames):
            state = np.array(state)
        is_single_input = state.ndim == self._state_ndim
        if is_single_input:
            state = np.expand_dims(state, axis=0).astype(np.float32)

        action, logp, v = self._get_action_logp_v_body(state, test)

        if is_single_input:
            v = v[0]
            action = action[0]

        return action.numpy(), logp.numpy(), v.numpy()

    def get_action_and_val_tensor(self, states, test=False):
        """Return batched on-policy inference tensors on the configured device.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Batched actions, action log probabilities, and
                values; discrete shared networks retain their own density convention.
        """
        with tf.device(self.device):
            return self._get_action_logp_v_body(states, test)

    @tf.function
    def _get_action_logp_v_body(self, state, test):
        """Return sampled actions, their log probabilities, and state-value estimates.

        Args:
            state (tf.Tensor): Observation batch shaped (batch, *state_shape); callers must add a batch
                axis to individual observations.
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Actions, densities, and state values returned by the
                separate or shared actor/value networks.
        """
        if self.actor_critic is not None:
            return self.actor_critic(state, test=test)
        else:
            action, logp = self.actor(state, test=test)
            v = self.critic(state)
            return action, logp, v

    @tf.function
    def _get_action_body(self, state, test):
        """Return batched policy actions and log probabilities on the configured device.

        Args:
            state (tf.Tensor): Observation batch shaped (batch, *state_shape); callers must add a batch
                axis to individual observations.
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Batched actions and their policy log probabilities, or the
                corresponding shared-network density output.
        """
        if self.actor_critic is not None:
            action, logp, _ = self.actor_critic(state, test=test)
            return action, logp
        else:
            return self.actor(state, test=test)

    def initialize_tensor_training(self, states):
        """Build actor/value networks and optimizer slots on the learner device.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).

        Returns:
            None.
        """
        with tf.device(self.device):
            self.get_action_and_val_tensor(states)
            if self.actor_critic is not None:
                self.actor_critic_optimizer.build(self.actor_critic.trainable_variables)
            else:
                self.actor_optimizer.build(self.actor.trainable_variables)
                self.critic_optimizer.build(self.critic.trainable_variables)

    def get_value_tensor(self, states):
        """Evaluate the baseline for separate or shared actor/value networks.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).

        Returns:
            tf.Tensor: State values of shape (batch,), computed without copying to NumPy.
        """
        with tf.device(self.device):
            return self._get_value_body(states)

    @tf.function(reduce_retracing=True)
    def _get_value_body(self, states):
        """Evaluate the shared or separate state-value model on a tensor batch.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).

        Returns:
            tf.Tensor: State-value estimates from the shared actor-critic or separate critic.
        """
        if self.actor_critic is not None:
            return self.actor_critic(states, test=True)[2]
        return self.critic(states)

    @tf.function(reduce_retracing=True)
    def train_tensor(self, states, actions, advantages, logp_olds, returns):
        """Apply vanilla policy gradients and value regression without host copies.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            advantages (array_like): Fixed policy advantages, one value per sampled transition.
            logp_olds (array_like): Collection-time log probabilities of the actions, held fixed during
                updates.
            returns (array_like): Discounted return targets, one value per transition.

        Returns:
            tf.Tensor: Diagnostics of shape (3,): actor loss, critic loss, and estimated KL, after one
                minibatch update.
        """
        with tf.device(self.device):
            if self.actor_critic is not None:
                return self._train_shared_body(states, actions, advantages, logp_olds, returns)
            actor_loss, logp_news = self._train_actor_body(states, actions, advantages, logp_olds)
            critic_loss = self._train_critic_body(states, returns)
            kl = tf.reduce_mean(tf.reshape(logp_olds, (-1,)) - logp_news)
            return tf.stack((actor_loss, critic_loss, kl))

    def _train_shared_body(self, states, actions, advantages, logp_olds, returns):
        """Apply the VPG and value losses to a shared actor-critic model.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            advantages (array_like): Fixed policy advantages, one value per sampled transition.
            logp_olds (array_like): Collection-time log probabilities of the actions, held fixed during
                updates.
            returns (array_like): Discounted return targets, one value per transition.

        Returns:
            tf.Tensor: Diagnostics [actor_loss, critic_loss, estimated_kl] after updating the shared
                network.
        """
        with tf.GradientTape() as tape:
            logp = self.actor_critic.compute_log_probs(states, actions)
            values = self.actor_critic(states, test=True)[2]
            actor_loss = -tf.reduce_mean(logp * tf.stop_gradient(tf.reshape(advantages, (-1,))))
            critic_loss = tf.reduce_mean(tf.square(tf.reshape(returns, (-1,)) - values))
            loss = actor_loss + self.vfunc_coef * critic_loss
        variables = self.actor_critic.trainable_variables
        self.actor_critic_optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        kl = tf.reduce_mean(tf.reshape(logp_olds, (-1,)) - logp)
        return tf.stack((actor_loss, critic_loss, kl))

    def train(self, states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        """Update the policy and value function from sampled returns and advantages.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            advantages (array_like): Fixed policy advantages, one value per sampled transition.
            logp_olds (array_like): Collection-time log probabilities of the actions, held fixed during
                updates.
            returns (array_like): Discounted return targets, one value per transition.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar actor and critic losses after the legacy public update
                path.
        """
        if self.actor_critic is not None:
            actor_loss, critic_loss, kl = tf.unstack(
                self.train_tensor(states, actions, advantages, logp_olds, returns))
            tf.summary.scalar(self.policy_name + "/actor_loss", actor_loss)
            tf.summary.scalar(self.policy_name + "/critic_loss", critic_loss)
            tf.summary.scalar(self.policy_name + "/kl", kl)
            return actor_loss, critic_loss
        actor_loss, logp_news = self._train_actor_body(
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
                          data=tf.reduce_mean(logp_olds - logp_news))
        tf.summary.scalar(name=self.policy_name +
                          "/critic_loss", data=critic_loss)
        return actor_loss, critic_loss

    @tf.function
    def _train_actor_body(self, states, actions, advantages, logp_olds):
        """Apply one policy-gradient update and return the optimization diagnostics.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            advantages (array_like): Fixed policy advantages, one value per sampled transition.
            logp_olds (array_like): Collection-time log probabilities of the actions, held fixed during
                updates. This implementation does not read this argument.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Scalar policy-gradient loss and per-action log probabilities
                after the actor update.
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                log_probs = self.actor.compute_log_probs(states, actions)
                weights = tf.stop_gradient(tf.squeeze(advantages))
                # + lambda * entropy
                actor_loss = tf.reduce_mean(-log_probs * weights)
            actor_grads = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grads, self.actor.trainable_variables))

        return actor_loss, log_probs

    @tf.function
    def _train_critic_body(self, states, returns):
        """Fit the state-value network to the supplied return targets.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            returns (array_like): Discounted return targets, one value per transition.

        Returns:
            tf.Tensor: Scalar value-regression loss after the critic update.
        """
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                current_V = self.critic(states)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(tf.square(td_errors))
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
