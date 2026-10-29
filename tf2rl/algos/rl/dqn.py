import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from tf2rl.algos.base import OffPolicyAgent
from tf2rl.envs.wrappers.atari import LazyFrames
from tf2rl.networks.layers import NoisyDense
from tf2rl.utils.tensorflow import huber_loss, update_target_variables


class QFunc(tf.keras.Model):
    """Neural network estimating values for each discrete action."""
    def __init__(self, state_shape, action_dim, units=(32, 32),
                 name="QFunc", enable_dueling_dqn=False, enable_noisy_dqn=False):
        """Build the discrete-action value network with the requested optional DQN
        enhancements.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order.
            name (str): Name assigned to the model, operation, or registered component.
            enable_dueling_dqn (bool): Separate the value and advantage heads of the Q network.
            enable_noisy_dqn (bool): Use noisy dense layers for value-based exploration.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(name=name)
        self._enable_dueling_dqn = enable_dueling_dqn
        self._enable_noisy_dqn = enable_noisy_dqn

        DenseLayer = NoisyDense if enable_noisy_dqn else Dense

        self.l1 = DenseLayer(units[0], name="L1", activation="relu")
        self.l2 = DenseLayer(units[1], name="L2", activation="relu")
        self.l3 = DenseLayer(action_dim, name="L3", activation="linear")

        if enable_dueling_dqn:
            self.l4 = DenseLayer(1, name="L3", activation="linear")

        with tf.device("/cpu:0"):
            self(inputs=tf.constant(np.zeros(shape=(1,) + state_shape,
                                             dtype=np.float32)))

    def call(self, inputs):
        """Return one Q-value per discrete action for each observation.

        Args:
            inputs (tf.Tensor): Observation batch shaped (batch, *state_shape); pixel models accept
                image observations.

        Returns:
            tf.Tensor: Discrete-action Q-values shaped (batch, action_dim).
        """
        features = self.l1(inputs)
        features = self.l2(features)
        if self._enable_dueling_dqn:
            advantages = self.l3(features)
            v_values = self.l4(features)
            q_values = (v_values
                        + (advantages
                           - tf.reduce_mean(advantages, axis=1, keepdims=True)))
        else:
            q_values = self.l3(features)
        return q_values


class DQN(OffPolicyAgent):
    """Deep Q-learning agent with epsilon-greedy exploration and a target network."""
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
            optimizer=None,
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
            optimizer (keras.optimizers.Optimizer or None): Optional externally supplied optimizer.
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
        self.q_func_optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=lr, epsilon=adam_eps)
        update_target_variables(self.q_func_target.weights,
                                self.q_func.weights, tau=1.)

        self._action_dim = action_dim
        # This is used to check if input state to `get_action` is multiple (batch) or single
        self._state_ndim = np.array(state_shape).shape[0]

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
        q_values = self.q_func(state)
        return tf.argmax(q_values, axis=1)

    def train(self, states, actions, next_states, rewards, done, weights=None):
        """Update the Q network from a replay minibatch and return temporal-difference errors.

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
    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        """Apply one Q-network gradient update using the importance-weighted Huber loss.

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
            tuple[tf.Tensor, tf.Tensor]: Per-transition TD/prioritization errors and scalar Q-network
                loss; parameters and scheduled targets are updated in place.
        """
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(
                    states, actions, next_states, rewards, dones)
                q_func_loss = tf.reduce_mean(
                    huber_loss(td_errors,
                               delta=self.max_grad) * weights)

            q_func_grad = tape.gradient(
                q_func_loss, self.q_func.trainable_variables)
            self.q_func_optimizer.apply_gradients(
                zip(q_func_grad, self.q_func.trainable_variables))

            return td_errors, q_func_loss

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
            tf.Tensor: Per-transition error diagnostic computed from online predictions and target-
                network values.
        """
        batch_size = states.shape[0]
        not_dones = 1. - tf.cast(dones, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.int32)

        with tf.device(self.device):
            """
            Get indices to update. If the sampled actions and corresponding q-values are:
                actions = [1, 0, 1], q_values = [[-1, 1], [-2, 2], [-3, 3]]
            then the indices will be:
                indices = [[0, 1], [1, 0], [2, 1]],
                current_q = [[1, -2, 3]]
            """
            indices = tf.concat(
                values=(tf.expand_dims(tf.range(batch_size), axis=1),
                        actions), axis=1)  # (batch_size, 1)
            current_q = tf.expand_dims(
                tf.gather_nd(self.q_func(states), indices), axis=1)  # (batch_size, 1)

            if self._enable_double_dqn:
                max_q_indexes = tf.argmax(self.q_func(next_states),
                                          axis=1, output_type=tf.int32)  # (batch_size,)
                indices = tf.concat(
                    values=(tf.expand_dims(tf.range(batch_size), axis=1),
                            tf.expand_dims(max_q_indexes, axis=1)), axis=1)  # (batch_size, 1)
                target_q = tf.expand_dims(
                    tf.gather_nd(self.q_func_target(next_states), indices), axis=1)  # (batch_size, 1)
                target_q = rewards + not_dones * self.discount * target_q
            else:
                next_target_q = tf.reduce_max(self.q_func_target(next_states), keepdims=True, axis=1)  # (batch_size, 1)
                target_q = rewards + not_dones * self.discount * next_target_q  # (batch_size, 1)
            td_errors = current_q - tf.stop_gradient(target_q)
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
