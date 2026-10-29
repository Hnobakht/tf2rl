import tensorflow as tf


class Policy(tf.keras.Model):
    """Base policy interface and shared training configuration."""

    def __init__(
        self,
        name,
        memory_capacity,
        update_interval=1,
        batch_size=256,
        discount=0.99,
        n_warmup=0,
        max_grad=10.0,
        n_epoch=1,
        gpu=0,
    ):
        """Store replay, update, discount, batch, and device settings shared by all policies.

        Args:
            name (str): Name assigned to the model, operation, or registered component.
            memory_capacity (int): Maximum number of transitions retained in replay.
            update_interval (int): Environment-step interval between scheduled learner updates.
            batch_size (int): Number of transitions in each learning or sampling minibatch.
            discount (float): Reward discount factor gamma.
            n_warmup (int): Number of physical transitions collected with random actions before
                learning.
            max_grad (float): Huber-loss transition threshold used by value-based losses; this field
                does not clip gradient norms.
            n_epoch (int): Number of learning epochs or dataset repetitions used by this operation.
            gpu (int): TensorFlow GPU index; a negative index selects the CPU.

        Returns:
            None: Initializes the instance.
        """
        super().__init__()
        self.policy_name = name
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.discount = discount
        self.n_warmup = n_warmup
        self.n_epoch = n_epoch
        self.max_grad = max_grad
        self.memory_capacity = memory_capacity
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    def get_action(self, observation, test=False):
        """Define action selection for subclasses; the base implementation is abstract.

        Subclasses define whether the result is an action alone or a tuple carrying action
        densities. Trainers accept either form.

        Args:
            observation (array_like): Observation returned by the wrapped environment.
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: The requested operation or argument combination is unsupported by this
                component.
        """
        raise NotImplementedError

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        import argparse

        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler="resolve")
        parser.add_argument(
            "--n-warmup", type=int, default=int(1e4), help=" <default: %(default)s>"
        )
        parser.add_argument("--batch-size", type=int, default=128, help=" <default: %(default)s>")
        parser.add_argument("--gpu", type=int, default=-1, help="GPU id  <default: %(default)s>")
        parser.add_argument("--debug", action="store_true", help=" <default: %(default)s>")
        parser.add_argument(
            "--pixels", action="store_true", help=" Training from raw Pixels <default: %(default)s>"
        )
        return parser


class OnPolicyAgent(Policy):
    """
    Base class for on-policy agents
    """

    def get_action_and_val_tensor(self, states, test=False):
        """Return batched action, action log probability and state value tensors.

        Subclasses must return (actions, log_probabilities, values) as tensors with a leading
        batch axis; log probabilities and values have one scalar per state.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: On-policy vector collection requires tensor inference.
        """
        raise NotImplementedError("On-policy vector collection requires tensor inference")

    def get_value_tensor(self, states):
        """Return state values without assuming a particular network architecture.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).

        Returns:
            tf.Tensor: Batched state values of shape (batch,), taken from the action/log-density/value
                inference result.
        """
        return self.get_action_and_val_tensor(states, test=True)[2]

    def initialize_tensor_training(self, states):
        """Build inference state before tracing the rollout update loop.

        Policies with lazy optimizer slots should also build those in this hook.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).

        Returns:
            None.
        """
        self.get_action_and_val_tensor(states)

    def train_tensor(self, states, actions, advantages, logp_olds, returns):
        """Update one minibatch; return [actor loss, value loss, estimated KL].

        Subclasses must return a tensor of shape (3,) containing actor loss, critic loss, and
        estimated KL after one minibatch update.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            advantages (array_like): Fixed policy advantages, one value per sampled transition.
            logp_olds (array_like): Collection-time log probabilities of the actions, held fixed during
                updates.
            returns (array_like): Discounted return targets, one value per transition.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: On-policy vector training requires tensor minibatch updates.
        """
        raise NotImplementedError("On-policy vector training requires tensor minibatch updates")

    def should_stop_update(self, diagnostics):
        """Define an algorithm's rollout stopping rule; the default runs all epochs.

        Args:
            diagnostics (array_like): Update diagnostics; on-policy agents place estimated KL at index
                two. This implementation does not read this argument.

        Returns:
            tf.Tensor: Scalar False; base on-policy agents do not request early stopping.
        """
        return tf.constant(False)

    def __init__(
        self,
        horizon=2048,
        lam=0.95,
        enable_gae=True,
        normalize_adv=True,
        entropy_coef=0.01,
        vfunc_coef=1.0,
        **kwargs,
    ):
        """Configure rollout length, advantage estimation, and policy/value loss coefficients.

        Args:
            horizon (int): Number of transitions in one on-policy rollout batch; must be divisible by
                batch_size.
            lam (float): GAE trace-decay factor lambda.
            enable_gae (bool): Use generalized advantage estimation; false selects one-step TD
                advantages.
            normalize_adv (bool): Standardize rollout advantages before policy updates.
            entropy_coef (float): Coefficient multiplying the policy entropy bonus.
            vfunc_coef (float): Coefficient multiplying the value-regression loss.
            **kwargs (dict): Additional keyword arguments forwarded to Policy construction.

        Returns:
            None: Initializes the instance.
        """
        self.horizon = horizon
        self.lam = lam
        self.enable_gae = enable_gae
        self.normalize_adv = normalize_adv
        self.entropy_coef = entropy_coef
        self.vfunc_coef = vfunc_coef
        kwargs["n_warmup"] = 0
        kwargs["memory_capacity"] = self.horizon
        super().__init__(**kwargs)
        msg = "Horizon should be divisible by batch size"
        assert self.horizon % self.batch_size == 0, msg

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = Policy.get_argument(parser)
        parser.add_argument("--horizon", type=int, default=2048)
        parser.add_argument("--normalize-adv", action="store_true")
        parser.add_argument("--enable-gae", action="store_true")
        return parser


class OffPolicyAgent(Policy):
    """
    Base class for off-policy agents
    """

    replay_mask_encoding = "done"

    def get_action_tensor(self, states, test=False):
        """Compatibility path for custom agents; agents can override to avoid host copies.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            test (bool or tf.Tensor): Select the policy evaluation action rule, such as Gaussian mean,
                categorical mode, or greedy Q action.

        Returns:
            tf.Tensor or tuple[tf.Tensor, ...]: Tensor-converted action result, preserving any auxiliary
                values returned by get_action.
        """
        result = self.get_action(states.numpy(), test=test)
        with tf.device(self.device):
            if isinstance(result, tuple):
                return tuple(tf.convert_to_tensor(value) for value in result)
            return tf.convert_to_tensor(result)

    def __init__(self, memory_capacity, **kwargs):
        """Set replay memory capacity and initialize the shared policy settings.

        Args:
            memory_capacity (int): Maximum number of transitions retained in replay.
            **kwargs (dict): Additional keyword arguments forwarded to Policy construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(memory_capacity=memory_capacity, **kwargs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = Policy.get_argument(parser)
        parser.add_argument("--memory-capacity", type=int, default=int(1e6))
        return parser


class RewardLearner(tf.keras.Model):
    """Base configuration and interface for inverse reinforcement learning models."""

    requires_log_probs = False

    def __init__(
        self, name, n_training=1, memory_capacity=0, batch_size=256, discount=0.99, gpu=0, **kwargs
    ):
        """Own reward models and optimizers without an action-selection interface.

        Args:
            name (str): Name assigned to the model, operation, or registered component.
            n_training (int): Number of discriminator training attempts per scheduled generator update.
            memory_capacity (int): Maximum number of transitions retained in replay.
            batch_size (int): Number of transitions in each learning or sampling minibatch.
            discount (float): Reward discount factor gamma.
            gpu (int): TensorFlow GPU index; a negative index selects the CPU.
            **kwargs (dict): Additional keyword arguments forwarded to tf.keras.Model construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__()
        self.policy_name = name
        self.n_training = n_training
        self.batch_size = batch_size
        self.discount = discount
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        # Retain constructor compatibility for experimental reward learners.
        self.memory_capacity = memory_capacity
        self.update_interval = kwargs.pop("update_interval", 1)
        self.n_warmup = kwargs.pop("n_warmup", 0)
        self.max_grad = kwargs.pop("max_grad", 10.0)
        self.n_epoch = kwargs.pop("n_epoch", 1)
        if kwargs:
            raise TypeError(f"Unknown reward learner options: {sorted(kwargs)}")

    def inference(self, states, actions, next_states=None):
        """Return learned rewards; subclasses define the scientific convention.

        Subclasses must return one learned reward per transition, conventionally a tensor of
        shape (batch, 1). Reward learners do not select environment actions.

        Args:
            states (array_like): Observation batch shaped (batch, *state_shape).
            actions (array_like): Actions aligned with the observation batch; continuous actions have a
                final action_dim axis.
            next_states (array_like or None): Physical successor observations aligned with the current-
                state batch.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: The requested operation or argument combination is unsupported by this
                component.
        """
        raise NotImplementedError

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = Policy.get_argument(parser)
        parser.add_argument(
            "--lr-ail",
            type=float,
            default=1e-4,
            help="Learning rate for GAIL, GAIfO, AIRL. <default: %(default)s>",
        )
        parser.add_argument(
            "--batch-size-ail", type=int, default=32, help=" <default: %(default)s>"
        )
        parser.add_argument(
            "--units-ail",
            type=str,
            default="128,128",
            help="Discriminator architecture. use case: 128,128 or 400,300 <default: %(default)s>",
        )
        parser.add_argument(
            "--n-training",
            type=int,
            default=1,
            help="Number of trainings for AIL <default: %(default)s>",
        )
        parser.add_argument(
            "--gp-coef",
            type=float,
            default=0.0,
            help="Gradient Penalty coefficient. <default: %(default)s>",
        )
        return parser


# Historical spelling retained for user code.
IRLPolicy = RewardLearner
