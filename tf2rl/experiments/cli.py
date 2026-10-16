"""One experiment entry point with compatible AIRL/GAIL/EQN example scripts."""
import random
from contextlib import ExitStack

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.airl import AIRL
from tf2rl.algos.airl_eql import AIRL_EQL
from tf2rl.algos.gail import GAIL
from tf2rl.algos.gail_eql import GAIL_EQL, equation_arguments
from tf2rl.algos.sac_irl import SAC_IRL
from tf2rl.algos.sac import SAC
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.experiments.airl_trainer import AIRLTrainer
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.demonstrations import restore_latest_n_traj


def _csv(value, cast):
    return [cast(item.strip()) for item in value.split(',')]


def main(argv=None, algorithm=None):
    parser = equation_arguments(AIRL.get_argument(SAC_IRL.get_argument(IRLTrainer.get_argument())))
    parser.add_argument('--algorithm', choices=('airl', 'gail', 'eqn-airl', 'eqn-gail', 'sac'),
                        default=algorithm or 'eqn-airl')
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--absorbing-indicator', type=float, default=.1)
    parser.add_argument('--irl-model-dir')
    parser.add_argument('--debug', action='store_true',
                        help='Run TensorFlow functions eagerly for Python breakpoints (slower)')
    parser.set_defaults(units_ail=None, memory_capacity=2_000_000)
    args = parser.parse_args(argv)
    if args.units_ail is None:
        args.units_ail = '64,64' if args.algorithm == 'eqn-airl' else '128,128'
    if args.pixels or args.draw_reward_function:
        parser.error('Pixel training and reward plotting remain separate experimental interfaces')
    if args.reward_dir:
        parser.error('--reward-dir requires an explicitly constructed external-reward wrapper; see docs/modernization.md')
    if args.debug and args.xla:
        parser.error('--debug requires eager execution and cannot be combined with --xla')
    if args.debug:
        tf.config.run_functions_eagerly(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    imitation = args.algorithm != 'sac'
    absorbing = imitation and args.absorbed_state_disable
    with ExitStack() as resources:
        # Human-mode reset can create a viewer before the trainer/checkpoint exists.
        # Register each environment immediately so setup errors also close it.
        env = gym.make(args.env_name, render_mode='human' if args.show_progress else None)
        resources.callback(env.close)
        test_mode = 'rgb_array' if args.save_test_movie else ('human' if args.show_test_progress else None)
        test_env = gym.make(args.env_name, render_mode=test_mode)
        resources.callback(test_env.close)
        if imitation:
            env = AbsorbingWrapper(env, args.absorbing_indicator)
            test_env = AbsorbingWrapper(test_env, args.absorbing_indicator)
        if not isinstance(env.action_space, gym.spaces.Box) or len(env.action_space.shape) != 1:
            parser.error('These research experiments require a vector Box action space')
        if not np.allclose(env.action_space.high, -env.action_space.low) or not np.allclose(env.action_space.high, env.action_space.high[0]):
            parser.error('The original actor requires uniform symmetric action bounds')
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        test_env.reset(seed=args.seed + 1)
        units = _csv(args.units, int)
        policy_class = SAC_IRL if imitation else SAC
        absorbing_kwargs = dict(is_absorbed_state=absorbing) if imitation else {}
        policy = policy_class(state_shape=env.observation_space.shape, action_dim=env.action_space.shape[0],
                          max_action=float(env.action_space.high[0]), actor_units=units, critic_units=units,
                          gpu=args.gpu, n_warmup=args.n_warmup, batch_size=args.batch_size,
                          memory_capacity=args.memory_capacity, auto_alpha=args.auto_alpha_disable,
                          alpha=args.alpha, lr=args.lr, lr_alpha=args.lr_alpha,
                          is_xla=args.xla, **absorbing_kwargs)
        if not imitation or args.evaluate:
            trainer = Trainer(policy, env, args, test_env)
        else:
            if args.expert_path_dir is None:
                parser.error('Imitation training requires --expert-path-dir; use --algorithm sac --save-test-path to export demonstrations')
            common = dict(state_shape=env.observation_space.shape, action_dim=env.action_space.shape[0],
                          gpu=args.gpu, batch_size=args.batch_size_ail, lr=args.lr_ail,
                          n_training=args.n_training, is_debug=args.is_debug)
            eql = dict(v=_csv(args.v, int), num_layers=len(args.v.split(',')),
                       drop_out=_csv(args.dropout, float), update_delay=args.update_delay,
                       exclude=[item.split('-') if item else [] for item in args.exclude.split(',')]
                       if args.exclude else None)
            if args.algorithm == 'eqn-airl':
                irl = AIRL_EQL(**common, **eql, is_state_only=args.state_only, units=_csv(args.units_ail, int))
            elif args.algorithm == 'airl':
                irl = AIRL(**common, state_only=args.state_only, units=_csv(args.units_ail, int))
            elif args.algorithm == 'eqn-gail':
                irl = GAIL_EQL(**common, **eql, grad_penalty_coeff=args.gp_coef)
            else:
                irl = GAIL(**common, units=_csv(args.units_ail, int), grad_penalty_coeff=args.gp_coef)
            data = restore_latest_n_traj(args.expert_path_dir, n_path=args.n_expert_trajectories,
                                         is_logps=irl.requires_log_probs, env=env, absorbing=absorbing)
            kwargs = dict(test_env=test_env, is_episodic=absorbing,
                          discriminator_noise=args.discriminator_noise, random_epoch=args.random_epoch_n)
            if irl.requires_log_probs:
                trainer = AIRLTrainer(policy, env, args, irl, data['obses'], data['next_obses'],
                                        data['acts'], data['logps'], **kwargs)
            else:
                trainer = IRLTrainer(policy, env, args, irl, data['obses'], data['next_obses'],
                                       data['acts'], **kwargs)
        # The initialized trainer now owns both environments and its summary writer.
        resources.pop_all()
        resources.callback(trainer.close)
        if imitation and not args.evaluate and args.irl_model_dir:
            checkpoint = tf.train.latest_checkpoint(args.irl_model_dir)
            if checkpoint is None:
                parser.error('No discriminator checkpoint in --irl-model-dir')
            trainer._checkpoint_irl.restore(checkpoint)
        trainer()


if __name__ == '__main__':
    main()
