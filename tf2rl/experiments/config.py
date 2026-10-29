"""CLI defaults and validation for the three research training recipes."""

import argparse
import json
from pathlib import Path

import numpy as np


class ConfigurationError(ValueError):
    """Invalid experiment settings, reported by the CLI without a traceback."""


def off_policy_parser(algorithm=None):
    """Build the compatible CLI without starting an experiment.

    Args:
        algorithm (str or None): Algorithm identifier or default selection for the experiment
            recipe.

    Returns:
        argparse.ArgumentParser: Parser with the corresponding experiment or trainer options added.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("sac", "td3", "ddpg"), default=algorithm or "sac")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--env-gpu", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--cpu-inter-threads",
        type=int,
        default=1,
        help="CPU workers for independent TensorFlow graph operations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument(
        "--collection-steps",
        type=int,
        default=16,
        help="Vector steps per progress report; updates occur after each step",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-capacity", type=int, default=1000000)
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10000,
        help="Random warmup physical transitions, aggregated over all worlds",
    )
    parser.add_argument(
        "--updates-per-transition",
        type=float,
        default=1.0,
        help="Gradient minibatch updates per new physical transition after warmup",
    )
    parser.add_argument("--units", nargs=2, type=int, default=[64, 64])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--fixed-alpha", action="store_true")
    parser.add_argument(
        "--xla", action="store_true", help="Compile SAC with XLA; adds startup cost"
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument(
        "--absorbing",
        action="store_true",
        help="Enable DAC absorbing observations and targets for SAC, TD3 or DDPG",
    )
    parser.add_argument("--absorbing-indicator", type=float, default=0.1)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--eval-env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--save-expert", action="store_true")
    parser.add_argument("--expert-episodes", type=int, default=20)
    parser.add_argument("--expert-dir", type=Path)
    parser.add_argument("--stochastic-expert", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Restore a policy and evaluate/export without training",
    )
    parser.add_argument("--output", type=Path, default=Path("results/off_policy"))
    return parser


def validate_off_policy(args):
    """Normalize and validate off policy options before constructing environments or models.

    Args:
        args (argparse.Namespace): Parsed experiment options used to configure this operation.

    Returns:
        argparse.Namespace: Validated argument object, with any normalized/defaulted fields written
            back in place.

    Raises:
        ConfigurationError: Counts must be positive. Warmup and evaluation counts must be
            nonnegative. updates-per-transition must be finite and nonnegative. memory-capacity must
            be at least batch-size. --xla currently supports SAC. --evaluate requires --model-dir.
            --gpu must name a visible GPU or be -1 for CPU. MJWarp requires a visible --env-gpu.
    """
    import tensorflow as tf

    if (
        min(
            args.num_envs,
            args.max_steps,
            args.collection_steps,
            args.batch_size,
            args.memory_capacity,
            args.cpu_threads,
            args.cpu_inter_threads,
            args.checkpoint_interval,
            args.eval_interval,
            args.eval_num_envs,
            args.expert_episodes,
            *args.units,
        )
        < 1
    ):
        raise ConfigurationError("Counts must be positive")
    if args.n_warmup < 0 or args.eval_episodes < 0:
        raise ConfigurationError("Warmup and evaluation counts must be nonnegative")
    if not np.isfinite(args.updates_per_transition) or args.updates_per_transition < 0:
        raise ConfigurationError("updates-per-transition must be finite and nonnegative")
    if args.memory_capacity < args.batch_size:
        raise ConfigurationError("memory-capacity must be at least batch-size")
    if args.xla and args.algorithm != "sac":
        raise ConfigurationError("--xla currently supports SAC")
    if args.evaluate and args.model_dir is None:
        raise ConfigurationError("--evaluate requires --model-dir")
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpu >= len(gpus):
        raise ConfigurationError("--gpu must name a visible GPU or be -1 for CPU")
    if "mjwarp" in (args.env_backend, args.eval_env_backend) and not 0 <= args.env_gpu < len(gpus):
        raise ConfigurationError("MJWarp requires a visible --env-gpu")
    return args


def on_policy_parser(algorithm=None):
    """Build the compatible CLI without starting an experiment.

    Args:
        algorithm (str or None): Algorithm identifier or default selection for the experiment
            recipe.

    Returns:
        argparse.ArgumentParser: Parser with the corresponding experiment or trainer options added.
    """
    from tf2rl.experiments.build import _ALGORITHMS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(_ALGORITHMS), default=algorithm or "ppo")
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--env-backend", choices=("gymnasium", "mjwarp"), default="gymnasium")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--units", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float, default=None, help="PPO KL stopping threshold")
    parser.add_argument(
        "--policy-kwargs",
        type=json.loads,
        default={},
        help="JSON keywords for the selected algorithm",
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument(
        "--env-gpu",
        type=int,
        default=None,
        help="MJWarp GPU index; defaults to --gpu (or 0 for a CPU policy)",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save a checkpoint every N measured iterations",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--env-kwargs", type=json.loads, default={})
    parser.add_argument("--nconmax", type=int, default=64)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def validate_on_policy(args):
    """Normalize and validate on policy options before constructing environments or models.

    Args:
        args (argparse.Namespace): Parsed experiment options used to configure this operation.

    Returns:
        argparse.Namespace: Validated argument object, with any normalized/defaulted fields written
            back in place.

    Raises:
        ConfigurationError: --policy-kwargs must be a JSON object. --target-kl is a PPO option; use
            --policy-kwargs for other algorithms. Counts must be positive. Evaluation and warmup
            counts must be nonnegative. cpu-threads and checkpoint-interval must be positive.
            --units requires two positive hidden widths. num-envs * rollout-steps must be divisible
            by batch-size. Select a visible --gpu for GPU policy training, or -1 for CPU. Select a
            visible --env-gpu for MJWarp simulation.
    """
    import tensorflow as tf

    from tf2rl.experiments.build import _ALGORITHMS

    if args.epochs is None:
        args.epochs = _ALGORITHMS[args.algorithm]["default_epochs"]
    if args.output is None:
        args.output = Path("results") / args.algorithm
    if not isinstance(args.policy_kwargs, dict):
        raise ConfigurationError("--policy-kwargs must be a JSON object")
    if args.target_kl is not None and args.algorithm != "ppo":
        raise ConfigurationError(
            "--target-kl is a PPO option; use --policy-kwargs for other algorithms"
        )
    if min(args.num_envs, args.rollout_steps, args.iterations, args.batch_size, args.epochs) < 1:
        raise ConfigurationError("Counts must be positive")
    if args.eval_episodes < 0 or args.warmup_iterations < 0:
        raise ConfigurationError("Evaluation and warmup counts must be nonnegative")
    if args.cpu_threads < 1 or args.checkpoint_interval < 1:
        raise ConfigurationError("cpu-threads and checkpoint-interval must be positive")
    if len(args.units) != 2 or min(args.units) < 1:
        raise ConfigurationError("--units requires two positive hidden widths")
    count = args.num_envs * args.rollout_steps
    if count % args.batch_size:
        raise ConfigurationError("num-envs * rollout-steps must be divisible by batch-size")
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpu >= len(gpus):
        raise ConfigurationError("Select a visible --gpu for GPU policy training, or -1 for CPU")
    if args.env_gpu is None:
        args.env_gpu = max(args.gpu, 0)
    if args.env_backend == "mjwarp" and not 0 <= args.env_gpu < len(gpus):
        raise ConfigurationError("Select a visible --env-gpu for MJWarp simulation")
    return args


def imitation_parser(algorithm=None):
    """Build the compatible imitation/expert CLI without starting training.

    Args:
        algorithm (str or None): Algorithm identifier or default selection for the experiment
            recipe.

    Returns:
        argparse.ArgumentParser: Parser with the corresponding experiment or trainer options added.
    """
    from tf2rl.algos.imitation.airl import AIRL
    from tf2rl.algos.imitation.gail_eqn import equation_arguments
    from tf2rl.algos.rl.sac import SAC
    from tf2rl.training.imitation import ImitationTrainer

    parser = equation_arguments(
        AIRL.get_argument(SAC.get_argument(ImitationTrainer.get_argument()))
    )
    parser.add_argument(
        "--algorithm",
        choices=("airl", "gail", "eqn-airl", "eqn-gail", "sac"),
        default=algorithm or "eqn-airl",
    )
    parser.add_argument("--env-name", required=True)
    parser.add_argument(
        "--generator",
        choices=("sac", "td3", "ddpg"),
        default="sac",
        help="Off-policy generator for imitation learning; AIRL requires SAC",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--absorbing-indicator", type=float, default=0.1)
    parser.add_argument("--irl-model-dir")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run TensorFlow functions eagerly for Python breakpoints (slower)",
    )
    parser.set_defaults(units_ail=None, memory_capacity=2_000_000, update_delay=None)
    return parser


def validate_imitation(args):
    """Normalize and validate imitation options before constructing environments or models.

    Args:
        args (argparse.Namespace): Parsed experiment options used to configure this operation.

    Returns:
        argparse.Namespace: Validated argument object, with any normalized/defaulted fields written
            back in place.

    Raises:
        ConfigurationError: Pixel training and reward plotting remain separate experimental
            interfaces. --reward-dir requires an explicitly constructed external-reward wrapper; see
            docs/history/modernization.md. --debug requires eager execution and cannot be combined
            with --xla. Equation sparsity/export options require eqn-gail or eqn-airl.
    """
    if args.units_ail is None:
        args.units_ail = "64,64" if args.algorithm == "eqn-airl" else "128,128"
    if args.pixels or args.draw_reward_function:
        raise ConfigurationError(
            "Pixel training and reward plotting remain separate experimental interfaces"
        )
    if args.reward_dir:
        raise ConfigurationError(
            "--reward-dir requires an explicitly constructed external-reward wrapper; see docs/history/modernization.md"
        )
    if args.debug and args.xla:
        raise ConfigurationError(
            "--debug requires eager execution and cannot be combined with --xla"
        )
    if (args.eqn_l1 or args.eqn_stages or args.export_equation) and not args.algorithm.startswith(
        "eqn-"
    ):
        raise ConfigurationError("Equation sparsity/export options require eqn-gail or eqn-airl")
    return args


def add_trainer_arguments(parser=None):
    """Add this component's command-line options and return the argument parser.

    Args:
        parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

    Returns:
        argparse.ArgumentParser: Parser with the corresponding experiment or trainer options added.
    """
    if parser is None:
        parser = argparse.ArgumentParser(conflict_handler="resolve")
    # experiment settings
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(1e6),
        help="Maximum number steps to interact with env. <default: %(default)s>",
    )
    parser.add_argument(
        "--episode-max-steps",
        type=int,
        default=int(1e3),
        help="Maximum steps in an episode <default: %(default)s>",
    )
    parser.add_argument(
        "--n-experiments", type=int, default=1, help="Number of experiments. <default: %(default)s>"
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Call `render` in training process. <default: %(default)s>",
    )
    parser.add_argument(
        "--show-progress-interval",
        type=int,
        default=int(1e4),
        help="Interval to save model. <default: %(default)s>",
    )
    parser.add_argument(
        "--save-model-interval",
        type=int,
        default=int(5e4),
        help="Interval to save model. <default: %(default)s>",
    )
    parser.add_argument(
        "--save-summary-interval",
        type=int,
        default=int(1e3),
        help="Interval to save summary. <default: %(default)s>",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to restore model. <default: %(default)s>",
    )
    parser.add_argument(
        "--reward-dir",
        type=str,
        default=None,
        help="Directory to restore rewad function. <default: %(default)s>",
    )
    parser.add_argument(
        "--dir-suffix",
        type=str,
        default="",
        help="Suffix for directory that contains results. <default: %(default)s>",
    )
    parser.add_argument(
        "--normalize-obs", action="store_true", help="Normalize observation. <default: %(default)s>"
    )
    parser.add_argument(
        "--logdir", type=str, default="results", help="Output directory. <default: %(default)s>"
    )
    # test settings
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate trained model. <default: %(default)s>"
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=int(1e4),
        help="Interval to evaluate trained model. <default: %(default)s>",
    )
    parser.add_argument(
        "--show-test-progress",
        action="store_true",
        help="Call `render` in evaluation process. <default: %(default)s>",
    )
    parser.add_argument(
        "--test-episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate at once. <default: %(default)s>",
    )
    parser.add_argument(
        "--save-test-path",
        action="store_true",
        help="Save trajectories of evaluation. <default: %(default)s>",
    )
    parser.add_argument(
        "--show-test-images",
        action="store_true",
        help="Show input images to neural networks when an episode finishes. <default: %(default)s>",
    )
    parser.add_argument(
        "--save-test-movie",
        action="store_true",
        help="Save rendering results. <default: %(default)s>",
    )
    # replay buffer
    parser.add_argument(
        "--use-prioritized-rb",
        action="store_true",
        help="Flag to use prioritized experience replay. <default: %(default)s>",
    )
    parser.add_argument(
        "--use-nstep-rb",
        action="store_true",
        help="Flag to use nstep experience replay. <default: %(default)s>",
    )
    parser.add_argument(
        "--n-step", type=int, default=4, help="Number of steps to look over. <default: %(default)s>"
    )
    # others
    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING"],
        default="INFO",
        help="Logging level. <default: %(default)s>",
    )
    parser.add_argument(
        "--is-debug",
        action="store_true",
        help="Plot the weights and/or gradient of the networks (only available for EQL methods). <default: %(default)s>",
    )
    return parser
