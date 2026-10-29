"""EQN-AIRL: equation reward network and the original dense shaping potential."""
from tf2rl.algos.imitation.airl import AIRL
from tf2rl.networks.equation import EquationNetwork


class StateModel_EQL(EquationNetwork):
    """Equation reward model over state observations."""
    def __init__(self, *args, **kwargs):
        """Configure the state-only equation reward model and its regularized output.

        Args:
            *args (tuple): Positional arguments forwarded to EquationNetwork construction.
            **kwargs (dict): Additional keyword arguments forwarded to EquationNetwork construction.

        Returns:
            None: Initializes the instance.
        """
        kwargs.setdefault("regularize_output", True)
        super().__init__(*args, **kwargs)

    def compute_reward(self, inputs):
        """Reject this reward conversion because it is undefined for this AIRL model.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: AIRL uses the raw equation reward, not discriminator odds.
        """
        raise NotImplementedError("AIRL uses the raw equation reward, not discriminator odds")


class StateActionModel_EQL(EquationNetwork):
    """Equation reward model over concatenated state-action inputs."""
    def compute_reward(self, inputs):
        """Reject this reward conversion because it is undefined for this AIRL model.

        Args:
            inputs (tf.Tensor): Batched discriminator inputs with concatenated state/action or
                state/successor coordinates as required by the model.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: AIRL uses the raw equation reward, not discriminator odds.
        """
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
        """Configure the imitation objective, discriminator networks, and optimizer settings.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            lr (float): Optimizer learning rate.
            num_layers (int): Number of equation layers.
            name (str): Name assigned to the model, operation, or registered component.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it.
            is_debug (bool): Enable additional discriminator/parameter diagnostics where implemented.
            is_state_only (bool): Use a state-only EQN reward parameterization for AIRL.
            update_delay (int): Run discriminator training only when its supplied iteration is divisible
                by this value.
            atol (float): Absolute weight threshold used by legacy tolerance-based inference copies.
            atol_starting_step (int): Environment-step threshold after which tolerance-based inference
                is selected. This implementation does not read this argument.
            v (sequence[int] or int): Operator repetition counts, one per equation layer; an individual
                layer takes one count.
            units (sequence[int]): Hidden-layer widths in forward order.
            mask_rate (float): Historical random-masking option; the active staged-sparsity
                implementations reject nonzero values.
            masking_starting_step (int): Historical masking start step; unsupported active
                configurations are rejected.
            lmbda (float): L1 regularization coefficient for equation parameters.
            regularizer_starting_step (int): Historical regularizer start step; use training_stages in
                the active EQN implementations.
            drop_out (float or sequence[float] or None): Dropout probability, or one probability per
                equation layer; None disables dropout.
            tau (float or None): Historical target-network option; nonzero values are unsupported by the
                current AIRL equation implementation.
            exclude (sequence or None): Equation operators to exclude; network constructors use one
                exclusion list per layer.
            training_stages (sequence[int] or None): Counts of warmup and sparse optimizer updates
                before fixed-mask refitting.
            prune_threshold (float): Prune parameters whose absolute values are below this threshold at
                the refitting boundary.
            **kwargs (dict): Additional keyword arguments forwarded to AIRL construction.

        Returns:
            None: Initializes the instance.

        Raises:
            NotImplementedError: Use training_stages; legacy AIRL mask/target options are unsupported.
        """
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
        """Build the state-only or state-action reward network.

        Args:
            state_shape (tuple[int, ...]): Shape of one observation, excluding the batch axis.
            action_dim (int): Number of discrete choices or width of a continuous action vector.
            units (sequence[int]): Hidden-layer widths in forward order. This implementation does not
                read this argument.
            enable_sn (bool): Request spectral normalization where the selected implementation supports
                it. This implementation does not read this argument.

        Returns:
            EquationNetwork: State-only or state-action equation reward network, without AIRL potential
                shaping.
        """
        cls = StateModel_EQL if self._state_only else StateActionModel_EQL
        return cls(state_shape=state_shape, action_dim=0 if self._state_only else action_dim,
                   name="reward_net", **self._equation_config)

    @property
    def equation_model(self):
        """Return the equation network used for reward inference and export.

        Returns:
            EquationNetwork: The equation reward network used for inference and export.
        """
        return self.rew_net

    def _before_update(self):
        """Prepare the equation sparsity stage before the discriminator update.

        Returns:
            None.
        """
        self.rew_net.prepare_update()

    def _after_update(self):
        """Project permanent masks and advance the equation reward optimizer-update counter.

        Returns:
            None.
        """
        self.rew_net.finish_update()

    def train(self, *args, **kwargs):
        """Update the imitation discriminator from agent and expert minibatches and report
        diagnostics.

        Args:
            *args (tuple): Agent and expert minibatches forwarded to the inherited discriminator
                training method in its declared order.
            **kwargs (dict): Additional discriminator-training options forwarded to the inherited
                implementation.

        Returns:
            tuple[tf.Tensor, tf.Tensor, tf.Tensor] or None: Discriminator loss, mean agent reward, and
                mean agent potential; None when update_delay skips this iteration.
        """
        result = super().train(*args, **kwargs)
        if result is not None:
            self.rew_net.write_sparsity_summaries()
        return result

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        from tf2rl.algos.imitation.gail_eqn import equation_arguments
        parser = AIRL.get_argument(parser)
        parser.add_argument('--units-ail', default="64,64")
        return equation_arguments(parser)


EQNAIRL = AIRL_EQL
