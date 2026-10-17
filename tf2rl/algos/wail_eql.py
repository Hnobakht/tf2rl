import tensorflow as tf


from tf2rl.misc.target_update_ops import update_target_variables
from tf2rl.algos.gail_eql import Discriminator_EQL, GAIL_EQL

class Discriminator_EQL_WAIL(Discriminator_EQL):
    """Equation discriminator with a Wasserstein-style raw reward output."""
    def __init__(self, **kwargs):
        """Configure the equation discriminator for a raw Wasserstein-style output."""
        super().__init__(**kwargs)

    def compute_reward(self, inputs, l1_regularizers = 0):
        """Return the raw equation discriminator output in inference mode."""
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
        """Configure the imitation objective, discriminator networks, and optimizer settings."""
        super().__init__(state_shape=state_shape, action_dim=action_dim, name=name, **kwargs)
        self.grad_penalty_coeff = grad_penalty_coeff #if grad_penalty_coeff > 0 else None

    def _init(self):
        """Create the discriminator copies and initialize their parameters."""
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
        """Return the historical Jensen-Shannon diagnostic for agent and expert outputs."""
        print("[DEBUG] initializing [_compute_js_divergence WAIL_EQL] shape: {TODO}")
        m = (fake_logits + real_logits) / 2.
        return tf.reduce_mean((
            fake_logits * tf.math.log(fake_logits / m + 1e-8) + real_logits * tf.math.log(real_logits / m + 1e-8)) / 2.)


    @tf.function
    def _train_body(self, agent_states, agent_acts, expert_states, expert_acts, l1_regularizers):
        """
        Apply discriminator gradients for the expert and agent batches and return training metrics.
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
