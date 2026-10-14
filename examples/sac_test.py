import gym
import numpy as np
# from gym.wrappers.flatten_observation import FlattenObservation
# import pybullet_envs
import tf2rl

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
from tf2rl.misc.prepare_output_dir import prep_log_tag
from tf2rl.envs.reward_wrapper import *
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SAC.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="HalfCheetah-v2")
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(max_steps=2e6)
    args = parser.parse_args()

    units = [int(item) for item in args.units.split(',')]

    args.logdir += prep_log_tag(args, "sac", units)

    env = gym.make(args.env_name)
    # env = AbsorbingWrapper(env)

    # env = gym.make("HalfCheetah-v2")
    # env = gym.make("reacher-simple-obs-v0")
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make("Pendulum-v0")
    # env = gym.make("Reacher-v2")
    # env = gym.make("Reacher-long-v2")
    # env = gym.make("reacher-simple-long-v0")
    # env = FlattenObservation(gym.make("HandReach-v0"))
    # env = FlattenObservation(gym.make("FetchReach-v1"))
    # env = gym.make("Hopper-v2")

    if args.reward_dir:
        # env = GailExternalReward(env, args.reward_dir, [128,128])
        # env = GailEQLExternalReward(env, args.reward_dir, 2, [2,1], [['div'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']], gpu=0)
        # env = GailEQLExternalReward(env, args.reward_dir, 2, [2,1], None, gpu=-1)
        # env = AirlEQLExternalReward(env, args.reward_dir, 2, [2, 1], [['div'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']], units=[64,64] , gpu=0, state_only=True)
        # env = AirlExternalReward(env, args.reward_dir, [128,128], state_only=True)
        # env = GaifoExternalReward(env, args.reward_dir, [128,128])
        # env = GaifoEQLExternalReward(env, args.reward_dir, 2, [1,1], [['div'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']], gpu=0)
        # env = WeightAtolApply(env, atol=None, drop=0.5, is_show=True)
        # env_visual = MazeRewardVisual(env)
        # [['div','log'],['div','log']]
        #[['div'], ['id', 'sin', 'cos', 'sig', 'log', 'mult']]
        draw_rew_map = True
        if draw_rew_map:
            not_maze = True
            from mpl_toolkits import mplot3d
            import matplotlib.pyplot as plt
            import matplotlib
            # import plotly.graph_objects as go
            from scipy.interpolate import griddata
            import scipy.ndimage as ndimage
            from scipy import interpolate as inter

            observation = env.reset()
            env.render()
            map = []
            act = []
            joi = []
            dis = []
            obs = []
            # map.append([env.get_body_com("target")[0], env.get_body_com("target")[1], 0])
            joints = np.array(np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-np.pi, np.pi, 100))).T.reshape(-1, 2)
            for _ in range(10000):
                action = env.action_space.sample()
                # observation, reward, done, info = env.step(action, observation)
                observation, reward, done, info = env.step(action)
                observation = np.squeeze(observation)
                # reward = reward - 20
                # env.render()
                if not_maze:
                    map.append([env.get_body_com("fingertip")[0],
                                env.get_body_com("fingertip")[1],
                                reward.tolist()])
                    act.append([action[0],
                                action[1],
                                reward.tolist()])
                    joi.append([np.arctan2(np.sin(env.sim.data.qpos.flat[:2][0]), np.cos(env.sim.data.qpos.flat[:2][0])),
                                env.sim.data.qpos.flat[:2][1],
                                reward.tolist()])
                    dis.append([np.linalg.norm(env.get_body_com("target")[0] - env.get_body_com("fingertip")[0]),
                                np.linalg.norm(env.get_body_com("target")[1] - env.get_body_com("fingertip")[1]),
                                reward.tolist()])
                    obs.append([observation[7],
                                observation[6],
                                reward.tolist()])
                else:
                    # map.append([observation[0][0], observation[0][1], reward.tolist()])
                    map.append([observation[0], observation[1], reward.tolist()])
                print(_) if _%1000==0 else None
            if not_maze:
                # # methode = 'nearest'
                # methode = 'linear'
                # # methode = 'cubic'
                # hight= 1000
                # wide = 1000
                # sigma = 5

                map = np.array(map)
                # lx = np.linspace(map[:,0].min(), map[:,0].max(), wide)
                # ly = np.linspace(map[:,1].min(), map[:,1].max(), hight)
                # [mapx, mapy] = np.meshgrid(lx, ly)
                # z = np.interp(map[:, 2], (map[:, 2].min(), map[:, 2].max()), (0, 1))
                # # z = map[:,2]
                # # mapz = griddata((map[:,0], map[:,1]), z, (mapx[None,:], mapy[:,None]), method='cubic', fill_value=0)
                # mapz = griddata((map[:, 0], map[:, 1]), z, (mapx, mapy), method=methode, fill_value=0, rescale=True)
                # # mapz = ndimage.gaussian_filter(mapz, sigma=sigma, order=1)
                #
                act = np.array(act)
                # lx = np.linspace(act[:,0].min(), act[:,0].max(), wide)
                # ly = np.linspace(act[:,1].min(), act[:,1].max(), hight)
                # [actx, acty] = np.meshgrid(lx, ly)
                # z = np.interp(act[:, 2], (act[:, 2].min(), act[:, 2].max()), (0, 1))
                # # z = map[:,2]
                # # mapz = griddata((map[:,0], map[:,1]), z, (mapx[None,:], mapy[:,None]), method='cubic', fill_value=0)
                # actz = griddata((act[:, 0], act[:, 1]), z, (actx, acty), method=methode, fill_value=0)
                # # actz = ndimage.gaussian_filter(actz, sigma=sigma, order=0)
                #
                joi = np.array(joi)
                # lx = np.linspace(joi[:, 0].min(), joi[:, 0].max(), wide)
                # ly = np.linspace(joi[:, 1].min(), joi[:, 1].max(), hight)
                # [joix, joiy] = np.meshgrid(lx, ly)
                # z = np.interp(joi[:, 2], (joi[:, 2].min(), joi[:, 2].max()), (0, 1))
                # # z = map[:,2]
                # # mapz = griddata((map[:,0], map[:,1]), z, (mapx[None,:], mapy[:,None]), method='cubic', fill_value=0)
                # joiz = griddata((joi[:, 0], joi[:, 1]), z, (joix, joiy), method=methode, fill_value=0)
                # # joiz = ndimage.gaussian_filter(joiz, sigma=sigma, order=0)
                #
                dis = np.array(dis)
                # lx = np.linspace(dis[:, 0].min(), dis[:, 0].max(), wide)
                # ly = np.linspace(dis[:, 1].min(), dis[:, 1].max(), hight)
                # [disx, disy] = np.meshgrid(lx, ly)
                # z = np.interp(dis[:, 2], (dis[:, 2].min(), dis[:, 2].max()), (0, 1))
                # # z = map[:,2]
                # # mapz = griddata((map[:,0], map[:,1]), z, (mapx[None,:], mapy[:,None]), method='cubic', fill_value=0)
                # disz = griddata((dis[:, 0], dis[:, 1]), z, (disx, disy), method=methode, fill_value=0)
                # # disz = ndimage.gaussian_filter(disz, sigma=sigma, order=0)
                #
                obs = np.array(obs)
                # lx = np.linspace(obs[:, 0].min(), obs[:, 0].max(), wide)
                # ly = np.linspace(obs[:, 1].min(), obs[:, 1].max(), hight)
                # [obsx, obsy] = np.meshgrid(lx, ly)
                # z = np.interp(obs[:, 2], (obs[:, 2].min(), obs[:, 2].max()), (0, 1))
                # # z = map[:,2]
                # # mapz = griddata((map[:,0], map[:,1]), z, (mapx[None,:], mapy[:,None]), method='cubic', fill_value=0)
                # obsz = griddata((obs[:, 0], obs[:, 1]), z, (obsx, obsy), method=methode, fill_value=0)
                # # obsz = ndimage.gaussian_filter(obsz, sigma=sigma, order=0)

                # ax = plt.axes(projection='3d')
                # fig1, ((axi, axj, axk, axl, axm)) = plt.subplots(ncols=5, nrows=1)
                fig1, ((axi, axk, axl, axm)) = plt.subplots(ncols=4, nrows=1)

                axi.set_title("Cartesian space")
                # axi.set_ylim(-0.3, 0.3)
                # axi.set_xlim(-.3, .3)
                axi.set_box_aspect(1)

                # axj.set_title("act")
                # # axj.set_ylim(-1, 1)
                # # axj.set_xlim(-1, 1)
                # axj.set_box_aspect(1)

                axk.set_title("Joint space")
                # axi.set_ylim(-1, 6000)
                # axi.set_xlim(0, int(2e6))
                axk.set_box_aspect(1)

                axl.set_title("Distance to the targer")
                # axl.set_ylim(-100, 1)
                # axi.set_xlim(0, int(2e6))
                axl.set_box_aspect(1)

                axm.set_title("Joint velocity space")
                axm.set_box_aspect(1)

                mapoo = map[:, 2] ** 3
                rew_map = np.interp(mapoo, (mapoo.min(), mapoo.max()), (0, 1))
                # rew_map = map[:, 2]
                s=8
                vmin=0
                vmax=1
                cmap = 'viridis'

                axi.scatter(map[:, 0], map[:, 1], c=rew_map, cmap=cmap, linewidth=0.5, vmin=vmin, vmax=vmax, s=s)
                # axi.scatter(map[:, 0], map[:, 1], c=rew_map, cmap=cmap, linewidth=0.5, s=s)
                # axi.contourf(mapx, mapy, mapz, 15, cmap='viridis')
                # axi.scatter(mapx, mapy, 10, mapz, cmap='viridis')
                # axi.scatter(mapx, mapy, c=mapz, cmap='viridis', linewidth=0.5)
                # axi.imshow(mapz, extent=(map[:,0].min(), map[:,0].max(), map[:,1].min(), map[:,1].max()), origin='lower', cmap='RdBu_r')
                # axi.pcolormesh(mapx, mapy, mapz, shading='flat', cmap='RdBu_r', vmin=0, vmax=1)
                # axi.plot(env.get_body_com("target")[0], env.get_body_com("target")[1], marker="o", color="red", markersize=1)
                axi.scatter(env.get_body_com("target")[0], env.get_body_com("target")[1], marker="o", color="red")

                # # axj.scatter(act[:, 0], act[:, 1], c=rew_map, cmap='viridis', linewidth=0.5)
                # axj.imshow(actz, extent=(0, 1, 0, 1), origin='lower', cmap='RdBu_r')

                axk.scatter(joi[:, 0], joi[:, 1], c=rew_map, cmap=cmap, linewidth=0.5, vmin=vmin, vmax=vmax, s=s)
                # axk.imshow(joiz, extent=(0, 1, 0, 1), origin='lower', cmap='RdBu_r')
                # axk.pcolormesh(joix, joiy, joiz, shading='flat', cmap='RdBu_r', vmin=0, vmax=1)

                axl.scatter(dis[:, 0], dis[:, 1], c=rew_map, cmap=cmap, linewidth=0.5, vmin=vmin, vmax=vmax, s=s)
                # axl.imshow(disz, extent=(0, 1, 0, 1), origin='lower', cmap='RdBu_r')
                # axl.pcolormesh(disx, disy, disz, shading='flat', cmap='RdBu_r', vmin=0, vmax=1)

                axm.scatter(obs[:, 0], obs[:, 1], c=rew_map, cmap=cmap, linewidth=0.5, vmin=vmin, vmax=vmax, s=s)
                # axm.imshow(obsz, extent=(0, 1, 0, 1), origin='lower', cmap='RdBu_r')
                # axm.pcolormesh(obsx, obsy, obsz, shading='flat', cmap='RdBu_r', vmin=0, vmax=1)
                # axm.contourf([obs[:, 0], obs[:, 1]], rew_map, cmap='viridis', linewidth=0.5)
            else:
                map = np.array(map)
                ax = plt.axes()
                # ax.scatter(map[:,0], map[:,1], map[:,2], c= map[:,2], cmap='viridis', linewidth=0.5)
                z = np.interp(map[:, 2], (map[:, 2].min(), map[:, 2].max()), (0, 1))
                # z = map[:,2]
                ax.scatter(map[:, 0], map[:, 1], c=z, cmap='viridis', linewidth=0.5, vmin=0, vmax=1)
                ax.scatter(env.get_body_com("target")[0], env.get_body_com("target")[1], marker="o", color="red")
                plt.colormaps()
            plt.show()
            exit(-1)


    test_env = env

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
        lr_alpha=args.lr_alpha
        )

    trainer = Trainer(policy, env, args, test_env=test_env, is_episodic=False)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()