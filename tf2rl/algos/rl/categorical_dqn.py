import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.base import OffPolicyAgent
from tf2rl.envs.wrappers.atari import LazyFrames
from tf2rl.networks.layers import NoisyDense
from tf2rl.utils.tensorflow import huber_loss, update_target_variables


class QFunc(tf.keras.Model):
    """Predict a categorical return distribution for each discrete action."""
    def __init__(self, state_shape, action_dim, units=(32, 32),
                 name="CategoricalQFunc", enable_dueling_dqn=False,
                 enable_noisy_dqn=False, n_atoms=51):
        """Build the discrete-action value network with the requested optional DQN
        enhancements.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            name (str): Name assigned to the model, operation, or registered component.
            enable_dueling_dqn (bool): Separate the value and advantage heads of the Q network.
            enable_noisy_dqn (bool): Use noisy dense layers for value-based exploration.
            n_atoms (int): Number of atoms in each categorical return distribution.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name)
        self._enable_dueling_dqn = enable_dueling_dqn
        self._enable_noisy_dqn = enable_noisy_dqn

        self._action_dim = action_dim
        self._n_atoms = n_atoms
        action_dim = (action_dim + int(enable_dueling_dqn)) * n_atoms

        DenseLayer = NoisyDense if enable_noisy_dqn else Dense

        self.l1 = DenseLayer(units[0], name="L1", activation="relu")
        self.l2 = DenseLayer(units[1], name="L2", activation="relu")
        self.l3 = DenseLayer(action_dim, name="L3", activation="linear")

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1,) + state_shape, dtype=np.float32)))

    def call(self, inputs):
        """Return a categorical return distribution for each action in the batch.

        Args:
            inputs (tf.Tensor): Observation batch shaped (batch, *state_shape); pixel models accept
                image observations.

        Returns:
            tf.Tensor: Clipped return-atom probabilities shaped (batch, action_dim, n_atoms).
        """
        features = self.l1(inputs)
        features = self.l2(features)
        features = self.l3(features)
        if self._enable_dueling_dqn:
            features = tf.reshape(
                features, (-1, self._action_dim + 1, self._n_atoms))  # [batch_size, action_dim, n_atoms]
            v_values = tf.reshape(
                features[:, 0], (-1, 1, self._n_atoms))
            advantages = tf.reshape(
                features[:, 1:], [-1, self._action_dim, self._n_atoms])
            features = v_values + (advantages - tf.expand_dims(
                tf.reduce_mean(advantages, axis=1), axis=1))
        else:
            features = tf.reshape(
                features, (-1, self._action_dim, self._n_atoms))  # [batch_size, action_dim, n_atoms]
        # [batch_size, action_dim, n_atoms]
        q_dist = tf.keras.activations.softmax(features, axis=2)
        return tf.clip_by_value(q_dist, 1e-8, 1.0 - 1e-8)

    @property
    def n_atoms(self):
        """Return the number of support atoms in each action-value distribution.

        Returns:
            int: Number of atoms used by each categorical action-value distribution.
        """
        return self._n_atoms


class CategoricalDQN(OffPolicyAgent):
    """Distributional DQN agent representing action values on a categorical support."""
    def __init__(
            self,
            state_shape,
            action_dim,
            q_func=None,
            name="DQN",
            lr=0.001,
            adam_eps=1e-07,
            units=(32, 32),
            epsilon=0.1,
            epsilon_min=None,
            epsilon_decay_step=int(1e6),
            n_warmup=int(1e4),
            target_replace_interval=int(5e3),
            memory_capacity=int(1e6),
            enable_double_dqn=False,
            enable_dueling_dqn=False,
            enable_noisy_dqn=False,
            **kwargs):
        """Configure the agent's networks, optimizers, and training hyperparameters.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            q_func (keras.Model or None): Optional Q network replacing the default architecture.
            name (str): Name assigned to the model, operation, or registered component.
            lr (float): Optimizer learning rate.
            adam_eps (float): Numerical stabilizer supplied to the Adam optimizer.
            units (sequence[int]): Hidden-layer widths in forward order.
            epsilon (float): Probability of a uniformly random action during epsilon-greedy exploration.
            epsilon_min (float or None): Final epsilon used by the optional linear exploration schedule.
            epsilon_decay_step (int): Number of environment steps over which epsilon decays.
            n_warmup (int): Number of physical transitions collected with random actions before
                learning.
            target_replace_interval (int): Learner-update interval between hard copies of target Q
                parameters.
            memory_capacity (int): Maximum number of transitions retained in replay.
            enable_double_dqn (bool): Select successor actions online and evaluate them using the target
                Q network.
            enable_dueling_dqn (bool): Separate the value and advantage heads of the Q network.
            enable_noisy_dqn (bool): Use noisy dense layers for value-based exploration.
            **kwargs (dict): Additional keyword arguments forwarded to OffPolicyAgent construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        q_func = q_func if q_func is not None else QFunc
        # Define and initialize Q-function network
        kwargs_dqn = {
            "state_shape": state_shape,
            "action_dim": action_dim,
            "units": units,
            "enable_dueling_dqn": enable_dueling_dqn,
            "enable_noisy_dqn": enable_noisy_dqn}
        self.q_func = q_func(**kwargs_dqn)
        self.q_func_target = q_func(**kwargs_dqn)
        self.q_func_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=adam_eps)
        update_target_variables(self.q_func_target.weights,
                                self.q_func.weights, tau=1.)

        self._action_dim = action_dim
        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

        # Distributional DQN
        self._v_max, self._v_min = 10., -10.
        self._delta_z = (self._v_max - self._v_min) / (self.q_func._n_atoms - 1)
        self._z_list = tf.constant(
            [self._v_min + i *
             self._delta_z for i in range(self.q_func._n_atoms)],
            dtype=tf.float32)
        self._z_list_broadcasted = tf.tile(
            tf.reshape(self._z_list, [1, self.q_func._n_atoms]),
            tf.constant([self._action_dim, 1]))

        # Set hyper-parameters
        if epsilon_min is not None and not enable_noisy_dqn:
            assert epsilon > epsilon_min
            self.epsilon_min = epsilon_min
            self.epsilon_decay_rate = (epsilon - epsilon_min) / epsilon_decay_step
            self.epsilon = max(epsilon - self.epsilon_decay_rate * self.n_warmup,
                               self.epsilon_min)
        else:
            epsilon = epsilon if not enable_noisy_dqn else 0.
            self.epsilon = epsilon
            self.epsilon_min = epsilon
            self.epsilon_decay_rate = 0.
        self.target_replace_interval = target_replace_interval
        self.n_update = 0

        # DQN variants
        self._enable_double_dqn = enable_double_dqn
        self._enable_noisy_dqn = enable_noisy_dqn

    def get_action(self, state, test=False, tensor=False):
        """Select an action for one observation or a batch; test mode disables exploration.

        Args:
            state (np.ndarray or tf.Tensor or LazyFrames): One observation or batch. LazyFrames is
                materialized; tensor=False requires NumPy input.
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.
            tensor (bool): Return a TensorFlow tensor instead of converting the action to NumPy, where
                implemented.

        Returns:
            int or np.ndarray or tf.Tensor: Epsilon-greedy action index or index batch; tensor=True
                returns a tensor, retaining the greedy batch axis for a single observation.
        """
        if isinstance(state, LazyFrames):
            state = np.array(state)
        if not tensor:
            assert isinstance(state, np.ndarray)
        is_single_input = state.ndim == self._state_ndim

        if not test and np.random.rand() < self.epsilon:
            if is_single_input:
                action = np.random.randint(self._action_dim)
            else:
                action = np.array([np.random.randint(self._action_dim)
                                   for _ in range(state.shape[0])], dtype=np.int64)
            if tensor:
                return tf.convert_to_tensor(action)
            else:
                return action

        state = np.expand_dims(state, axis=0).astype(
            np.float32) if is_single_input else state
        action = self._get_action_body(tf.constant(state))
        if tensor:
            return action
        else:
            if is_single_input:
                return action.numpy()[0]
            else:
                return action.numpy()

    @tf.function
    def _get_action_body(self, state):
        """Return greedy actions from the batched Q-network predictions.

        Args:
            state (tf.Tensor): Observation batch shaped (batch, *state_shape); callers must add a batch
                axis to individual observations.

        Returns:
            tf.Tensor: Greedy discrete action indices of shape (batch,), maximizing expected Q-values.
        """
        action_probs = self.q_func(state)
        return tf.argmax(
            tf.reduce_sum(action_probs * self._z_list_broadcasted, axis=2),
            axis=1)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        """Update the categorical Q network, scheduled targets, and exploration rate, returning
        the replay diagnostic.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.
            rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
                have a trailing singleton axis.
            done (array_like): True MDP termination flags aligned with the replay batch; time-limit
                truncations should remain zero.
            weights (array_like or None): Per-transition importance-sampling weights, shaped (batch,) or
                (batch, 1); public update methods replace None with unit weights.

        Returns:
            tf.Tensor: Per-transition replay-priority diagnostic produced by the update.
        """
        if weights is None:
            weights = np.ones_like(rewards)
        td_errors, q_func_loss = self._train_body(
            states, actions, next_states, rewards, done, weights)

        tf.summary.scalar(name=self.policy_name + "/q_func_Loss", data=q_func_loss)

        # TODO: Remove following by using tf.global_step
        self.n_update += 1
        # Update target networks
        if self.n_update % self.target_replace_interval == 0:
            update_target_variables(
                self.q_func_target.weights, self.q_func.weights, tau=1.)

        # Update exploration rate
        self.epsilon = max(self.epsilon - self.epsilon_decay_rate * self.update_interval,
                           self.epsilon_min)
        tf.summary.scalar(name=self.policy_name + "/epsilon", data=self.epsilon)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        """Apply a Q-network gradient update using the categorical error diagnostic.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.
            rewards (array_like): Rewards aligned with transitions; scalar per-transition values may
                have a trailing singleton axis.
            done (array_like): True MDP termination flags aligned with the replay batch; time-limit
                truncations should remain zero.
            weights (array_like or None): Per-transition importance-sampling weights, shaped (batch,) or
                (batch, 1); public update methods replace None with unit weights.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Per-transition TD/prioritization errors and scalar Q-network
                loss; parameters and scheduled targets are updated in place.
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, done)
                q_func_loss = tf.reduce_mean(
                    huber_loss(tf.negative(td_errors),
                               delta=self.max_grad) * weights)

            q_func_grad = tape.gradient(
                q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(
                zip(q_func_grad, self.q_func.trainable_variables))

            return td_errors, q_func_loss

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        # TODO: fix this ugly conversion
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
            tf.Tensor: Per-transition error diagnostic for replay priority calculation without applying
                gradients.
        """
        if isinstance(actions, tf.Tensor):
            actions = tf.expand_dims(actions, axis=1)
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)

        return self._compute_td_error_body(
            states, actions, next_states, rewards, dones)

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        """Compute per-sample log-probability errors against the projected categorical Bellman
        distribution.

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
            tf.Tensor: Per-transition error diagnostic computed from online predictions and target-
                network values.
        """
        batch_size = states.shape[0]
        actions = tf.cast(actions, dtype=tf.int32)


        with tf.device(self.device):
            rewards = tf.tile(
                tf.reshape(rewards, [-1, 1]),
                tf.constant([1, self.q_func.n_atoms]))  # [batch_size, n_atoms]
            not_done = 1.0 - tf.tile(
                tf.reshape(dones, [-1, 1]),
                tf.constant([1, self.q_func.n_atoms]))  # [batch_size, n_atoms]
            discounts = tf.cast(
                tf.reshape(self.discount, [-1, 1]), tf.float32)  # [1, 1]
            z = tf.reshape(
                self._z_list, [1, self.q_func.n_atoms])  # [1, n_atoms]
            z = rewards + not_done * discounts * z  # [batch_size, n_atoms]
            z = tf.clip_by_value(z, self._v_min, self._v_max)  # [batch_size, n_atoms]
            b = (z - self._v_min) / self._delta_z  # [batch_size, n_atoms]

            index_help = tf.expand_dims(
                tf.tile(
                    tf.reshape(tf.range(batch_size), [-1, 1]),
                    tf.constant([1, self.q_func.n_atoms])),
                -1)  # [batch_size, n_atoms, 1]
            u, l = tf.math.ceil(b), tf.math.floor(b)  # [batch_size, n_atoms]
            u_id = tf.concat(
                [index_help, tf.expand_dims(tf.cast(u, tf.int32), -1)],
                axis=2)  # [batch_size, n_atoms, n_action]
            l_id = tf.concat(
                [index_help, tf.expand_dims(tf.cast(l, tf.int32), -1)],
                axis=2)  # [batch_size, n_atoms, n_action]

            target_q_next_dist = self.q_func_target(
                next_states)  # [batch_size, n_action, n_atoms]

            if self._enable_double_dqn:
                # TODO: Check this implementation is correct
                target_q_next_dist = tf.gather_nd(
                    target_q_next_dist,
                    tf.concat(
                        [tf.reshape(tf.range(batch_size), [-1, 1]),
                         tf.reshape(actions, [-1, 1])],
                        axis=1))
            else:
                target_q_next_sum = tf.reduce_sum(
                    target_q_next_dist * self._z_list_broadcasted, axis=2)  # [batch_size, n_action]
                actions_by_target_q = tf.cast(
                    tf.argmax(target_q_next_sum, axis=1),
                    tf.int32)  # [batch_size,]
                target_q_next_dist = tf.gather_nd(
                    target_q_next_dist,
                    tf.concat(
                        [tf.reshape(tf.range(batch_size), [-1, 1]),
                         tf.reshape(actions_by_target_q, [-1, 1])],
                        axis=1))  # [batch_size, n_atoms]

            action_indices = tf.concat(
                values=(tf.expand_dims(tf.range(batch_size), axis=1),
                        actions), axis=1)
            current_q_dist = tf.gather_nd(
                self.q_func(states), action_indices)  # [batch_size, n_atoms]

            td_errors = tf.reduce_sum(
                target_q_next_dist * (u - b) * tf.math.log(tf.gather_nd(current_q_dist, l_id)) +
                target_q_next_dist * (b - l) * tf.math.log(tf.gather_nd(current_q_dist, u_id)),
                axis=1)

        return td_errors

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = OffPolicyAgent.get_argument(parser)
        parser.add_argument('--enable-double-dqn', action='store_true')
        parser.add_argument('--enable-dueling-dqn', action='store_true')
        parser.add_argument('--enable-noisy-dqn', action='store_true')
        return parser
