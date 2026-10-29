import tensorflow as tf

from tf2rl.algos.imitation.gail_eqn import GAIL_EQL, Discriminator_EQL
from tf2rl.utils.tensorflow import update_target_variables


class Discriminator_EQL_WAIL(Discriminator_EQL):
    """Equation discriminator with a Wasserstein-style raw reward output."""
    def __init__(self, **kwargs):
        """Configure the equation discriminator for a raw Wasserstein-style output.

        Args:
            **kwargs (dict): Additional keyword arguments forwarded to Discriminator_EQL construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(**kwargs)

    def compute_reward(self, inputs, l1_regularizers = 0):
        """Return the raw equation discriminator output in inference mode.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it.

        Returns:
            tf.Tensor: Raw Wasserstein equation-critic scores shaped (batch, 1).
        """
        print("[DEBUG] initializing [compute_reward WAIL_EQL] shape: {} name: {}".format(inputs.shape, self.name))
        return self(inputs, training=False, l1_regularizers=l1_regularizers)

class WAIL_EQL(GAIL_EQL):
    """Experimental Wasserstein adversarial imitation with an equation discriminator."""
    def __init__(
            self,
            state_shape,
            action_dim,
            name="WAIL_EQL",
            grad_penalty_coeff=10,
            **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            name (str): Name assigned to the model, operation, or registered component.
            grad_penalty_coeff (float): Coefficient multiplying the discriminator gradient penalty.
            **kwargs (dict): Additional keyword arguments forwarded to GAIL_EQL construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(state_shape=state_shape, action_dim=action_dim, name=name, **kwargs)
        self.grad_penalty_coeff = grad_penalty_coeff #if grad_penalty_coeff > 0 else None

    def _init(self):
        """Create the discriminator copies and initialize their parameters.

        Returns:
            None.
        """
        self.disc = Discriminator_EQL_WAIL(state_shape=self.state_shape, action_dim=self.action_dim, v=self.v, drop_out=self.drop_out,
                                          num_layers=self.num_layers, enable_sn=self.enable_sn, constraint=None,
                                          name="Discriminator", lmbda=self.lmbda, exclude = self.exclude,
                                          is_lmbda_dynamic=self._is_reg_dyna, output_activation=tf.nn.sigmoid,)
        if self.atol_starting_step or self.tau > 0:
            self.disc_target = Discriminator_EQL_WAIL(state_shape=self.state_shape, action_dim=self.action_dim, v=self.v,
                                                     drop_out=self.drop_out,num_layers=self.num_layers,
                                                     enable_sn=self.enable_sn, constraint=None,name="Discriminator_target",
                                                     lmbda=self.lmbda, exclude = self.exclude,
                                                     is_lmbda_dynamic=self._is_reg_dyna, output_activation=tf.nn.sigmoid,)
        if self.tau > 0:
            with tf.device(self.device):
                update_target_variables(self.disc_target.weights,
                                        self.disc.weights, tau=1.)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.5)

    @tf.function
    def _compute_js_divergence(self, fake_logits, real_logits):
        """Return the historical Jensen-Shannon diagnostic for agent and expert outputs.

        Args:
            fake_logits (tf.Tensor): Agent discriminator outputs used directly in the probability-based
                divergence calculation.
            real_logits (tf.Tensor): Expert discriminator outputs used directly in the probability-based
                divergence calculation.

        Returns:
            tf.Tensor: Scalar stabilized probability-based Jensen-Shannon diagnostic between agent and
                expert outputs.
        """
        print("[DEBUG] initializing [_compute_js_divergence WAIL_EQL] shape: {TODO}")
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)


    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts, l1_regularizers):
        """Apply discriminator gradients for the expert and agent batches and return training
        metrics.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_acts (array_like): Agent actions aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_acts (array_like): Expert actions aligned with expert_states.
            l1_regularizers (float): Historical per-call L1 coefficient; some modern equation paths
                retain but ignore it.

        Returns:
            tuple: Total critic loss, accuracy, JS diagnostic, and discriminator gradient tensors after
                applying the Wasserstein update.
        """
        print("[DEBUG] initializing [_train_body WAIL_EQL] shape: {TODO}")
        epsilon = 1e-8
        with tf.device(self.device):
            expert_inputs =  tf.concat((expert_states, expert_acts), axis=1)
            agent_inputs = tf.concat((agent_states, agent_acts), axis=1)

            alpha = tf.random.uniform(shape=(agent_inputs.get_shape()[0], 1))
            inter = alpha * agent_inputs + (1 - alpha) * expert_inputs
            if self.grad_penalty_coeff > 0:
                print("[DEBUG] the gradient penalty is enabled with coefficient: {} [_train_body WAIL_EQL]".format(
                    self.grad_penalty_coeff))
                with tf.GradientTape() as tape2:
                    tape2.watch(inter)
                    output = self.disc(inter, l1_regularizers=l1_regularizers)
                grad = tape2.gradient(output, [inter])[0]
                grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))
            else:
                print("[DEBUG] the gradient penalty is disabled. [_train_body WAIL_EQL]")
                grad_penalty = 0.

            with tf.GradientTape() as tape: # watch_accessed_variables=False
                real_logits = self.disc(expert_inputs, l1_regularizers=l1_regularizers)
                fake_logits = self.disc(agent_inputs, l1_regularizers=l1_regularizers)
                classification_loss =  tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

                total_loss = classification_loss + self.grad_penalty_coeff * tf.stop_gradient(grad_penalty)

            grads = tape.gradient(total_loss, self.disc.variables)
            self.optimizer.apply_gradients(zip(grads, self.disc.variables))

            accuracy = (tf.reduce_mean(tf.cast(real_logits >= 0.5, tf.float32)) / 2. +
                        tf.reduce_mean(tf.cast(fake_logits < 0.5, tf.float32)) / 2.)
            js_divergence = self._compute_js_divergence(fake_logits, real_logits)

            if self.tau > 0:
                update_target_variables(self.disc_target.weights,
                                        self.disc.weights, tau=self.tau)
        return total_loss, accuracy, js_divergence, grads
