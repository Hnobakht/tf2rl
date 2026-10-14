import numpy as np
import tensorflow as tf

from tf2rl.algos.vail import VAIL, Discriminator
# from tf2rl.algos.policy_base import IRLPolicy

class VAILfO(VAIL):
    def __init__(
            self,
            state_shape,
            action_dim,
            # units=(32, 32),
            # decoder_units = (1),
            # n_latent_unit=32,
            # lr=5e-5,
            # kl_target=0.5,
            # reg_param=0.,
            # enable_sn=False,
            # grad_penalty_coeff = 0.,
            # n_training = 1,
            name="VAILfO",
            **kwargs):
        super().__init__(state_shape=state_shape ,action_dim=action_dim,  name = name, **kwargs)


    def train(self, agent_states, agent_next_states, expert_states, expert_next_states, **kwargs):
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
        assert states.shape == next_states.shape
        if states.ndim == 1:
            states = np.expand_dims(states, axis=0)
            next_states = np.expand_dims(next_states, axis=0)
        inputs = np.concatenate((states, next_states), axis=1)
        with tf.device(self.device):
            out = self._inference_body_graph(tf.constant(inputs), tf.constant(False))
        return out

    def _init_static_graph(self, state_shape, action_dim, n_latent_unit):
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
                                                 # experimental_compile=True,
                                                 # autograph=False
                                                 )

            self._inference_body_graph = tf.function(self._inference_body,
                                                     input_signature=[tf.TensorSpec(shape=(None, state_shape[-1]*2),dtype=tf.float32),
                                                                      tf.TensorSpec(shape=(None), dtype=tf.bool)],
                                                     # experimental_compile=True
                                                     # autograph=False
                                                     )

            self._compute_kl_latent_graph = tf.function(self._compute_kl_latent,
                                                        input_signature=[tf.TensorSpec(shape=(None, n_latent_unit), dtype=tf.float32),
                                                                         tf.TensorSpec(shape=(None, n_latent_unit), dtype=tf.float32)],
                                                        # experimental_compile=True
                                                        # autograph=False
                                                        )

            self._compute_js_divergence_graph = tf.function(self._compute_js_divergence,
                                                            input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                                                                             tf.TensorSpec(shape=(None, 1), dtype=tf.float32)],
                                                            # experimental_compile=True
                                                            # autograph=False
                                                            )
