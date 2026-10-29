import numpy as np
import tensorflow as tf

from tf2rl.algos.imitation.vail import VAIL


class VAILfO(VAIL):
    """Variational adversarial imitation from observation transitions."""
    def __init__(
            self,
            state_shape,
            action_dim,
            name="VAILfO",
            **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            name (str): Name assigned to the model, operation, or registered component.
            **kwargs (dict): Additional keyword arguments forwarded to VAIL construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(state_shape=state_shape ,action_dim=action_dim,  name = name, **kwargs)


    def train(self, agent_states, agent_next_states, expert_states, expert_next_states, **kwargs):
        """Update the discriminator from agent/expert minibatches and write training
        diagnostics to TensorBoard.

        Args:
            agent_states (array_like): Agent observation batch shaped (agent_batch, *state_shape).
            agent_next_states (array_like): Agent successor observations aligned with agent_states.
            expert_states (array_like): Expert observation batch shaped (expert_batch, *state_shape).
            expert_next_states (array_like): Expert successor observations aligned with expert_states.
            **kwargs (dict): Additional compatibility keyword options, currently unused.

        Returns:
            None.
        """
        with tf.device(self.device):
            agent_states, agent_next_states, expert_states, expert_next_states = \
                tuple([tf.constant(i) for i in (agent_states, agent_next_states,
                                                expert_states, expert_next_states)])

            loss, accuracy, real_kl, fake_kl, js_divergence = self._train_body_graph(agent_states, agent_next_states,
                                                                                     expert_states, expert_next_states,
                                                                                     tf.constant(True))

            tf.summary.scalar(name=self.policy_name+"/DiscriminatorLoss", data=loss)
            tf.summary.scalar(name=self.policy_name+"/Accuracy", data=accuracy)
            tf.summary.scalar(name=self.policy_name+"/RegParam", data=self._reg_param)
            tf.summary.scalar(name=self.policy_name+"/RealLatentKL", data=real_kl)
            tf.summary.scalar(name=self.policy_name+"/FakeLatentKL", data=fake_kl)
            tf.summary.scalar(name=self.policy_name+"/JSdivergence", data=js_divergence)


    def inference(self, states, actions, next_states):
        """Return learned imitation rewards for the supplied observations and actions or
        transitions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis. This implementation does not read this argument.
            next_states (array_like): Physical successor observations aligned with the current-state
                batch.

        Returns:
            tf.Tensor: Batched variational discriminator reward predictions; single-state inputs are
                promoted to a batch before inference.
        """
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        with tf.device(self.device):
            out = self._inference_body_graph(tf.constant(inputs), tf.constant(False))
        return out

    def _init_static_graph(self, state_shape, action_dim, n_latent_unit):
        """Set up TensorFlow signatures and traced model, training, or inference computations.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector. This
                implementation does not read this argument.
            n_latent_unit (int): Number of latent coordinates in the variational discriminator.

        Returns:
            None.
        """
        print("[DEBUG] initializing {_init_static_graph VAILfO}")
        # Compiling the static graphs
        assert len(state_shape) == 1, "[ERROR] This SAC only supports RAM env {_init_static_graph SAC}"
        with tf.device(self.device):
            # TODO: the @tf.function on top is better!
            self.disc.call = tf.function(self.disc.call,
                                         input_signature=[
                                             tf.TensorSpec(shape=(None, state_shape[-1]*2), dtype=tf.float32),
                                             tf.TensorSpec(shape=(), dtype=tf.bool)]
                                         )

            self._train_body_graph = tf.function(self._train_body,
                                                 input_signature=[
                                                     tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None, state_shape[-1]), dtype=tf.float32),
                                                     tf.TensorSpec(shape=(None), dtype=tf.bool)],
                                                 )

            self._inference_body_graph = tf.function(self._inference_body,
                                                     input_signature=[tf.TensorSpec(shape=(None, state_shape[-1]*2),dtype=tf.float32),
                                                                      tf.TensorSpec(shape=(None), dtype=tf.bool)],
                                                     )

            self._compute_kl_latent_graph = tf.function(self._compute_kl_latent,
                                                        input_signature=[tf.TensorSpec(shape=(None, n_latent_unit), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, n_latent_unit), dtype=tf.float32)],
                                                        )

            self._compute_js_divergence_graph = tf.function(self._compute_js_divergence,
                                                            input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(None, 1), dtype=tf.float32)],
                                                            )
