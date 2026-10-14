"""EQN-GAIL with equation penalties, delayed updates and log-odds rewards."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.gail import GAIL
from tf2rl.networks.eqn import EquationNetwork
from tf2rl.misc.target_update_ops import update_target_variables

Discriminator_EQL = EquationNetwork


def equation_arguments(parser):
    parser.add_argument('--update-delay', type=int, default=3)
    parser.add_argument('--dropout', default='.5,0.')
    parser.add_argument('-v', '--v', default='2,1')
    parser.add_argument('--exclude', default=None)
    return parser


class GAIL_EQL(GAIL):
    equation_discriminator = True

    def __init__(self, state_shape, action_dim, training_stages=None, num_layers=2,
                 lr=0.001, enable_sn=False, name="GAIL_EQL", is_debug=False,
                 update_delay=1, atol=0, atol_starting_step=0, v=(2, 1), mask_rate=0,
                 masking_starting_step=0, lmbda=0, regularizer_starting_step=0,
                 drop_out=None, tau=0., exclude=None, grad_penalty_coeff=0, **kwargs):
        if mask_rate or regularizer_starting_step or training_stages:
            raise NotImplementedError("EQN masking and staged L1 training were never implemented")
        self.atol = atol
        self.atol_starting_step = atol_starting_step if atol > 0 else None
        self.tau = tau
        self._equation_config = dict(state_shape=state_shape, action_dim=action_dim,
                                     num_layers=num_layers, v=v, drop_out=drop_out,
                                     exclude=exclude, lmbda=lmbda, output_activation=tf.nn.sigmoid)
        super().__init__(state_shape, action_dim, lr=lr, enable_sn=enable_sn, name=name,
                         is_debug=is_debug, update_delay=update_delay,
                         grad_penalty_coeff=grad_penalty_coeff, **kwargs)
        if self.atol_starting_step is not None or tau > 0:
            self.disc_target = EquationNetwork(name="Discriminator_target", **self._equation_config)
            if tau > 0:
                update_target_variables(self.disc_target.weights, self.disc.weights, tau=1.)

    def _make_discriminator(self, *args):
        return EquationNetwork(**self._equation_config)

    def _after_update(self):
        if self.tau > 0:
            update_target_variables(self.disc_target.weights, self.disc.weights, tau=self.tau)

    def inference(self, states, actions, next_states=None, apply_atol=False):
        states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if states.ndim == 1:
            states, actions = states[None], actions[None]
        inputs = np.concatenate((states, actions), axis=1)
        return self._inference_body_atol(inputs) if apply_atol else self._inference_body(inputs)

    @tf.function
    def _inference_body_atol(self, inputs):
        update_target_variables(self.disc_target.weights,
                                [tf.where(tf.abs(w) > self.atol, w, 0.) for w in self.disc.weights])
        return self.disc_target.compute_reward(inputs)

    @tf.function
    def _inference_body(self, inputs):
        with tf.device(self.device):
            return (self.disc_target if self.tau > 0 else self.disc).compute_reward(inputs)

    @staticmethod
    def get_argument(parser=None):
        return equation_arguments(GAIL.get_argument(parser))


EQNGAIL = GAIL_EQL
