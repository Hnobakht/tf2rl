"""One experiment entry point with compatible AIRL/GAIL/EQN example scripts."""

import random
from contextlib import ExitStack

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.airl import AIRL
from tf2rl.algos.gail_eql import equation_arguments
from tf2rl.algos.sac import SAC
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.experiments.imitation import (
    make_discriminator,
    make_imitation_trainer,
    make_policy,
    restore_discriminator,
)
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.runtime import ConfigurationError
from tf2rl.experiments.trainer import Trainer


def build_parser(algorithm=None):
    """Build the compatible imitation/expert CLI without starting training."""
    parser = equation_arguments(AIRL.get_argument(SAC.get_argument(IRLTrainer.get_argument())))
    parser.add_argument(
        "--algorithm",
        choices=("airl", "gail", "eqn-airl", "eqn-gail", "sac"),
        default=algorithm or "eqn-airl",
    )
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--generator", choices=("sac", "td3", "ddpg"), default="sac",
                        help="Off-policy generator for imitation learning; AIRL requires SAC")
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


def run(args):
    """Run one imitation or expert experiment using the existing trainers."""
    if args.units_ail is None:
        args.units_ail = "64,64" if args.algorithm == "eqn-airl" else "128,128"
    if args.pixels or args.draw_reward_function:
        raise ConfigurationError(
            "Pixel training and reward plotting remain separate experimental interfaces"
        )
    if args.reward_dir:
        raise ConfigurationError(
            "--reward-dir requires an explicitly constructed external-reward wrapper; see docs/modernization.md"
        )
    if args.debug and args.xla:
        raise ConfigurationError(
            "--debug requires eager execution and cannot be combined with --xla"
        )
    if (args.eqn_l1 or args.eqn_stages or args.export_equation) and not args.algorithm.startswith(
        "eqn-"
    ):
        raise ConfigurationError("Equation sparsity/export options require eqn-gail or eqn-airl")
    if args.debug:
        tf.config.run_functions_eagerly(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    imitation = args.algorithm != "sac"
    absorbing = imitation and args.absorbed_state_disable
    with ExitStack() as resources:
        # Human-mode reset can create a viewer before the trainer/checkpoint exists.
        # Register each environment immediately so setup errors also close it.
        env = gym.make(args.env_name, render_mode="human" if args.show_progress else None)
        resources.callback(env.close)
        test_mode = (
            "rgb_array" if args.save_test_movie else ("human" if args.show_test_progress else None)
        )
        test_env = gym.make(args.env_name, render_mode=test_mode)
        resources.callback(test_env.close)
        if imitation:
            env = AbsorbingWrapper(env, args.absorbing_indicator)
            test_env = AbsorbingWrapper(test_env, args.absorbing_indicator)
        if not isinstance(env.action_space, gym.spaces.Box) or len(env.action_space.shape) != 1:
            raise ConfigurationError("These research experiments require a vector Box action space")
        if not np.allclose(env.action_space.high, -env.action_space.low) or not np.allclose(
            env.action_space.high, env.action_space.high[0]
        ):
            raise ConfigurationError("The original actor requires uniform symmetric action bounds")
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        test_env.reset(seed=args.seed + 1)
        policy = make_policy(args, env, imitation=imitation, absorbing=absorbing)
        if not imitation or args.evaluate:
            trainer = Trainer(policy, env, args, test_env)
        else:
            if args.expert_path_dir is None:
                raise ConfigurationError(
                    "Imitation training requires --expert-path-dir; use --algorithm sac --save-test-path to export demonstrations"
                )
            irl = make_discriminator(args, env)
            trainer = make_imitation_trainer(args, policy, irl, env, test_env, absorbing)
        # The initialized trainer now owns both environments and its summary writer.
        resources.pop_all()
        resources.callback(trainer.close)
        if imitation and not args.evaluate and args.irl_model_dir:
            restore_discriminator(trainer, irl, args.irl_model_dir)
        trainer()


def main(argv=None, algorithm=None):
    """Parse options and preserve existing example/console entry points."""
    parser = build_parser(algorithm)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ConfigurationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
