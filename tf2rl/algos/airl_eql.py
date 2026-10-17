"""EQN-AIRL: equation reward network and the original dense shaping potential."""
from tf2rl.algos.airl import AIRL
from tf2rl.networks.eqn import EquationNetwork


class StateModel_EQL(EquationNetwork):
    """Equation reward model over state observations."""
    def __init__(self, *args, **kwargs):
        """Configure the state-only equation reward model and its regularized output."""
        kwargs.setdefault("regularize_output", True)
        super().__init__(*args, **kwargs)

    def compute_reward(self, inputs):
        """Reject this reward conversion because it is undefined for this AIRL model."""
        raise NotImplementedError("AIRL uses the raw equation reward, not discriminator odds")


class StateActionModel_EQL(EquationNetwork):
    """Equation reward model over concatenated state-action inputs."""
    def compute_reward(self, inputs):
        """Reject this reward conversion because it is undefined for this AIRL model."""
        raise NotImplementedError("AIRL uses the raw equation reward, not discriminator odds")


class AIRL_EQL(AIRL):
    """AIRL with an equation reward network and a dense shaping potential."""
    equation_reward = True

    def __init__(self, state_shape, action_dim, lr=0.001, num_layers=2, name="AIRL_EQL",
                 enable_sn=False, is_debug=False, is_state_only=False, update_delay=1,
                 atol=0, atol_starting_step=0, v=(2, 1), units=(32, 32), mask_rate=0,
                 masking_starting_step=0, lmbda=0, regularizer_starting_step=0,
                 drop_out=None, tau=None, exclude=None, training_stages=None,
                 prune_threshold=.001, **kwargs):
        """Configure the imitation objective, discriminator networks, and optimizer settings."""
        if atol or mask_rate or masking_starting_step or tau or regularizer_starting_step:
            raise NotImplementedError("Use training_stages; legacy AIRL mask/target options are unsupported")
        self.atol_starting_step = None
        self._equation_config = {"num_layers": num_layers, "v": v, "drop_out": drop_out,
                                     "exclude": exclude, "lmbda": lmbda, "training_stages": training_stages,
                                     "prune_threshold": prune_threshold}
        super().__init__(state_shape, action_dim, state_only=is_state_only, units=units,
                         lr=lr, enable_sn=enable_sn, name=name, is_debug=is_debug,
                         update_delay=update_delay, **kwargs)

    def _make_reward(self, state_shape, action_dim, units, enable_sn):
        """Build the state-only or state-action reward network."""
        cls = StateModel_EQL if self._state_only else StateActionModel_EQL
        return cls(state_shape=state_shape, action_dim=0 if self._state_only else action_dim,
                   name="reward_net", **self._equation_config)

    @property
    def equation_model(self):
        """Return the equation network used for reward inference and export."""
        return self.rew_net

    def _before_update(self):
        """Prepare the equation sparsity stage before the discriminator update."""
        self.rew_net.prepare_update()

    def _after_update(self):
        """Finalize sparsity masks and update any target discriminator after training."""
        self.rew_net.finish_update()

    def train(self, *args, **kwargs):
        """
        Update the imitation discriminator from agent and expert minibatches and report diagnostics.
        """
        result = super().train(*args, **kwargs)
        if result is not None:
            self.rew_net.write_sparsity_summaries()
        return result

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        from tf2rl.algos.gail_eql import equation_arguments
        parser = AIRL.get_argument(parser)
        parser.add_argument('--units-ail', default="64,64")
        return equation_arguments(parser)


EQNAIRL = AIRL_EQL
