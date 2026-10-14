import numpy as np
from gymnasium import utils
from tf2rl.envs import research_mujoco as mujoco_env
from pathlib import Path
import csv

class HalfCheetahDisabledEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_disabled.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, bool(done), False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HalfCheetahLongBodyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_long_body.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, bool(done), False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HalfCheetahLongLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_long_leg.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, bool(done), False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HalfCheetahPlot(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        self.joints_list = []
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)


    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, bool(done), False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        joint_pos = self.data.qpos.flat[3:9]
        joints_dic = {"j1": joint_pos[0], "j2": joint_pos[1], "j3": joint_pos[2], "j4": joint_pos[3],
                      "j5": joint_pos[4], "j6": joint_pos[5]}
        self.joints_list.append(joints_dic)
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if len(self.joints_list) > 0:
            output = Path("results") / "halfcheetah_joints.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.joints_list[0])
                writer.writeheader()
                writer.writerows(self.joints_list)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
