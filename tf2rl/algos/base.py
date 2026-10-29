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
        """Store replay, update, discount, batch, and device settings shared by all policies."""
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
        """Define action selection for subclasses; the base implementation is abstract."""
        raise NotImplementedError

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
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
        """Return batched action, action log probability and state value tensors."""
        raise NotImplementedError("On-policy vector collection requires tensor inference")

    def get_value_tensor(self, states):
        """Return state values without assuming a particular network architecture."""
        return self.get_action_and_val_tensor(states, test=True)[2]

    def initialize_tensor_training(self, states):
        """Build inference state before tracing the rollout update loop.

        Policies with lazy optimizer slots should also build those in this hook.
        """
        self.get_action_and_val_tensor(states)

    def train_tensor(self, states, actions, advantages, logp_olds, returns):
        """Update one minibatch; return [actor loss, value loss, estimated KL]."""
        raise NotImplementedError("On-policy vector training requires tensor minibatch updates")

    def should_stop_update(self, diagnostics):
        """Define an algorithm's rollout stopping rule; the default runs all epochs."""
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
        """Configure rollout length, advantage estimation, and policy/value loss coefficients."""
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
        """Add this component's command-line options and return the argument parser."""
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
        """Compatibility path for custom agents; agents can override to avoid host copies."""
        result = self.get_action(states.numpy(), test=test)
        with tf.device(self.device):
            if isinstance(result, tuple):
                return tuple(tf.convert_to_tensor(value) for value in result)
            return tf.convert_to_tensor(result)

    def __init__(self, memory_capacity, **kwargs):
        """Set replay memory capacity and initialize the shared policy settings."""
        super().__init__(memory_capacity=memory_capacity, **kwargs)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = Policy.get_argument(parser)
        parser.add_argument("--memory-capacity", type=int, default=int(1e6))
        return parser


class RewardLearner(tf.keras.Model):
    """Base configuration and interface for inverse reinforcement learning models."""

    requires_log_probs = False

    def __init__(
        self, name, n_training=1, memory_capacity=0, batch_size=256, discount=0.99, gpu=0, **kwargs
    ):
        """Own reward models and optimizers without an action-selection interface."""
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
        """Return learned rewards; subclasses define the scientific convention."""
        raise NotImplementedError

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
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
