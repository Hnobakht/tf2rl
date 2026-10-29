"""Assemble an experiment, call its trainer, and close resources.

Training loops live in tf2rl.training. Use --runner to select an existing recipe.
"""

import argparse
import random
from contextlib import ExitStack

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.envs.factory import make_environment, make_single_environment
from tf2rl.experiments import config
from tf2rl.experiments.build import (
    make_discriminator,
    make_imitation_agent,
    make_imitation_trainer,
    make_off_policy_agent,
    make_on_policy_agent,
    preload_imitation_algorithms,
)
from tf2rl.training.checkpoints import checkpoint_prefix, restore_discriminator
from tf2rl.training.off_policy import OffPolicyTrainer
from tf2rl.training.on_policy import OnPolicyTrainer
from tf2rl.utils.tensorflow import configure_tensorflow


def run_off_policy(args):
    config.validate_off_policy(args)
    configure_tensorflow(args.seed, args.cpu_threads, args.cpu_inter_threads)
    with ExitStack() as resources:
        env = make_environment(args, args.env_backend, args.num_envs, absorbing=args.absorbing)
        resources.callback(env.close)
        policy = make_off_policy_agent(
            args.algorithm,
            env.single_observation_space,
            env.single_action_space,
            absorbing=args.absorbing,
            gpu=args.gpu,
            units=args.units,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            memory_capacity=args.memory_capacity,
            n_warmup=args.n_warmup,
            auto_alpha=not args.fixed_alpha,
            alpha=args.alpha,
            sigma=args.sigma,
            xla=args.xla,
        )
        restored = None
        if args.model_dir is not None:
            restored = checkpoint_prefix(args.model_dir)
            tf.train.Checkpoint(policy=policy).read(
                restored
            ).assert_existing_objects_matched().expect_partial()
        trainer = OffPolicyTrainer(
            policy,
            env,
            args,
            absorbing=args.absorbing,
            seed=args.seed,
            updates_per_transition=args.updates_per_transition,
        )
        evaluation_env = None
        if args.eval_episodes or args.save_expert:
            evaluation_env = make_environment(
                args,
                args.eval_env_backend,
                min(
                    args.eval_num_envs,
                    max(args.eval_episodes, args.expert_episodes if args.save_expert else 1),
                ),
                absorbing=args.absorbing,
            )
            resources.callback(evaluation_env.close)
        return trainer.run(evaluation_env=evaluation_env, restored=restored)


def run_on_policy(args):
    config.validate_on_policy(args)
    configure_tensorflow(args.seed, args.cpu_threads)
    with ExitStack() as resources:
        env = make_environment(args, args.env_backend, args.num_envs)
        resources.callback(env.close)
        policy_kwargs = dict(args.policy_kwargs)
        if args.target_kl is not None:
            policy_kwargs["target_kl"] = args.target_kl
        with tf.device(f"/GPU:{args.gpu}" if args.gpu >= 0 else "/CPU:0"):
            policy = make_on_policy_agent(
                args.algorithm,
                env.single_observation_space,
                env.single_action_space,
                horizon=args.num_envs * args.rollout_steps,
                batch_size=args.batch_size,
                epochs=args.epochs,
                units=args.units,
                learning_rate=args.learning_rate,
                gpu=args.gpu,
                policy_kwargs=policy_kwargs,
            )
            trainer = OnPolicyTrainer(policy, env, args.rollout_steps, args.seed)
            return trainer.run(args)


def run_imitation(args):
    preload_imitation_algorithms()
    config.validate_imitation(args)
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
        env = make_single_environment(args, imitation=imitation)
        resources.callback(env.close)
        test_env = make_single_environment(args, evaluation=True, imitation=imitation)
        resources.callback(test_env.close)
        if not isinstance(env.action_space, gym.spaces.Box) or len(env.action_space.shape) != 1:
            raise config.ConfigurationError(
                "These research experiments require a vector Box action space"
            )
        if not np.allclose(env.action_space.high, -env.action_space.low) or not np.allclose(
            env.action_space.high, env.action_space.high[0]
        ):
            raise config.ConfigurationError(
                "The original actor requires uniform symmetric action bounds"
            )
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        test_env.reset(seed=args.seed + 1)
        policy = make_imitation_agent(args, env, imitation=imitation, absorbing=absorbing)
        if not imitation or args.evaluate:
            trainer = OffPolicyTrainer(policy, env, args, test_env)
        else:
            if args.expert_path_dir is None:
                raise config.ConfigurationError("Imitation training requires --expert-path-dir")
            irl = make_discriminator(args, env)
            trainer = make_imitation_trainer(args, policy, irl, env, test_env, absorbing)
        # The initialized trainer now owns both environments and its summary writer.
        resources.pop_all()
        resources.callback(trainer.close)
        if imitation and not args.evaluate and args.irl_model_dir:
            restore_discriminator(trainer, irl, args.irl_model_dir)
        return trainer.run()


_RUNNERS = {"off_policy": run_off_policy, "on_policy": run_on_policy, "imitation": run_imitation}
_PARSERS = {
    "off_policy": config.off_policy_parser,
    "on_policy": config.on_policy_parser,
    "imitation": config.imitation_parser,
}


def main(argv=None, algorithm=None, runner=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--runner", choices=tuple(_RUNNERS), default=runner or "off_policy")
    selected, remaining = selector.parse_known_args(argv)
    parser = _PARSERS[selected.runner](algorithm)
    parser.add_argument(
        "--runner",
        choices=tuple(_RUNNERS),
        default=selected.runner,
        help="Training recipe: off_policy, on_policy, or imitation",
    )
    args = parser.parse_args(remaining)
    try:
        return _RUNNERS[selected.runner](args)
    except config.ConfigurationError as error:
        parser.error(str(error))


def imitation_main(argv=None):
    return main(argv, runner="imitation")


def off_policy_main(argv=None):
    return main(argv, runner="off_policy")


def on_policy_main(argv=None):
    return main(argv, runner="on_policy")


def ppo_main(argv=None):
    return main(argv, runner="on_policy", algorithm="ppo")


if __name__ == "__main__":
    main()
