import time
import gym
import numpy as np
# import pybullet_envs
# import d4rl
import tf2rl

from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper

# ans = [i for i in d4rl.sequence_dataset(gym.make("halfcheetah-expert-v2"), max_episodes=10)]
# import njustQuadEnv.envs
# from pyquaternion import Quaternion as Q

# np.set_printoptions(suppress=True)

# env = gym.make('MountainCarContinuous-v0')
# env = gym.make("Pendulum-v0")
# env = gym.make("Reacher-v2")
# env = gym.make("Humanoid-v2")
# env = gym.make("HumanoidStandup-v2")
# env = gym.make("HandReach-v0")
# env = gym.make("FetchReach-v1")
# env = gym.make("Hopper-v2")

# env = gym.make('InvertedPendulumBulletEnv-v0')
# env = gym.make('Ant-v2')
# env = wrap_car_racing(env)
# env = wrap_deepmind(env)
# env = gym.make('antrllab-v0')
# env = gym.make("walker2d-fat-v0")
# env = gym.make("Walker2d-v2")

# env = gym.make("pointmaze-right-v0")
# env = gym.make("antrllab-rotation-disabled-v0")
# env = gym.make("reacher-simple-obs-v0")
# env = gym.make("ant-disabled2-v0")
# env = gym.make("ant-full-disabled-v0")
env = gym.make("ant-plot-v0")

env = AbsorbingWrapper(env, indicator=0.1)
# env.render()
obs = env.reset()
# cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
# time_list = []
# for i in range(5000):
# for i in range(env._max_episode_steps):
action = 1
while True:
    # env.render('human')
    action = env.action_space.sample()
    # action = 1 if action == -1 else -1
    # action = np.zeros(env.action_space.shape)
    # print(action)

    # time_befor_loop = time.process_time()
    time_befor_loop = time.time()
    # joint_pos = env.sim.data.qpos[env.model.jnt_qposadr[1:]]
    observation, reward, done, info = env.step([action[0], 1 ,1, 1, 1, 1, 1, 1])
    # print(np.abs(env.sim.data.qpos[env.model.jnt_qposadr[1:]] - joint_pos))
    # print(joint_pos)
    env.render()
    # time.sleep(1)
    # observation, reward, done, info = env.step(np.array([1,1,1,1,1,1,1,1]))
    # observation, reward, done, info = env.step(np.array([-1,-1,-1,-1,-1,-1,-1,-1]))
    # observation, reward, done, info = env.step(np.zeros(8))
    # if done:
    #     observation = env.reset()
    # time.sleep(0.04)
    # time_diff = time.process_time() - time_befor_loop
    # time_diff = time.time() - time_befor_loop
    # print(observation)
    # print(np.rad2deg(Q(observation[4:8]).yaw_pitch_roll))
    # print(reward)
    # if done:
    #     print(done)
    # print("itr <<" + str(i) + ">>: " + str(time_diff))
