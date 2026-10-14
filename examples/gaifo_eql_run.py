import gymnasium as gym
# import d4rl
import tf2rl

from tf2rl.algos.gaifo_eql import GAIfO_EQL
# from tf2rl.algos.sac import SAC
from tf2rl.algos.sac_irl import SAC_IRL as SAC
# from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.irl_trainer_v2 import IRLTrainerV2 as IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj
from tf2rl.misc.prepare_output_dir import prep_log_tag
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper

if __name__ == '__main__':
    parser = IRLTrainer.get_argument()
    parser = GAIfO_EQL.get_argument(parser)
    parser = SAC.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.add_argument('--env-name', type=str, required=True)
    args = parser.parse_args()

    v = [int(item) for item in args.v.split(',')]
    units = [int(item) for item in args.units.split(',')]
    excludes = [item.split('-') for item in args.exclude.split(',')] if args.exclude else None
    dropouts = [float(item) for item in args.dropout.split(',')]

    args.logdir += prep_log_tag(args, "gaifo-eql", "sac", v, units, "DEBUG_STAGE")
    print("[DEBUG] Excluded activations are " + str(excludes))

    if args.expert_path_dir is None:
        print("Plaese generate demonstrations first")
        print("python examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000")
        exit()

    env = gym.make(args.env_name)
    env = AbsorbingWrapper(env)
    test_env = AbsorbingWrapper(gym.make(args.env_name), indicator=0.1)

    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        max_action=env.action_space.high[0],
        gpu=args.gpu,
        actor_units=units,
        critic_units=units,
        n_warmup=args.n_warmup,
        batch_size=args.batch_size,
        auto_alpha=args.auto_alpha_disable,
        alpha=args.alpha,
        lr=args.lr,
        lr_alpha=args.lr_alpha,
        is_absorbed_state=args.absorbed_state_disable
        )

    irl = GAIfO_EQL(
        state_shape=env.observation_space.shape,
        action_dim=None,
        num_layers=len(v),
        v=v,
        batch_size=args.batch_size_ail,
        gpu=args.gpu,
        lr=args.lr_ail,
        is_debug=args.is_debug,
        update_delay=args.update_delay,
        drop_out=dropouts,
        exclude=excludes,
        # grad_penalty_coeff= 0,
        )

    # sample_size = int(20*1000)
    # expert_dataset = d4rl.qlearning_dataset(gym.make('walker2d-expert-v2'), terminate_on_end=True)
    # trainer = IRLTrainer(policy, env, args, irl, expert_dataset["observations"][-sample_size:, :],
    #                      expert_dataset["next_observations"][-sample_size:, :], expert_dataset["actions"][-sample_size:, :],
    #                      test_env, is_episodic=args.absorbed_state, discriminator_noise=args.discriminator_noise)

    expert_trajs = restore_latest_n_traj(args.expert_path_dir,
                                         n_path=args.n_expert_trajectories,
                                         max_steps=1000, env=env)
    trainer = IRLTrainer(policy, env, args, irl, expert_trajs["obses"],
                         expert_trajs["next_obses"], expert_trajs["acts"],
                         test_env, is_episodic=args.absorbed_state_disable,
                         discriminator_noise=args.discriminator_noise,
                         random_epoch=args.random_epoch_n)

    trainer()