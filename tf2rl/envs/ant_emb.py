import numpy as np
import csv
from pathlib import Path
from gymnasium import utils
from tf2rl.envs import research_mujoco as mujoco_env

class AntEnvDisabled(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'ant_disabled.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, bool(done), False, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class AntEnvLonglegs(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'ant_longlegs.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, bool(done), False, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class AntEnvDisabled2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'ant_disabled_2.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, bool(done), False, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class AntEnvFullDisabled(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,  ploting=False, render_mode=None):
        self.ploting = ploting
        if self.ploting:
            self.joints_list = []
        mujoco_env.MujocoEnv.__init__(self, 'ant_full_disabled.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)


    def step(self, a):
        if self.ploting:
            joints_dic = {"j1": a[0], "j2": a[1], "j3": a[2], "j4": a[3],
                          "j5": a[4], "j6": a[5], "j7": a[6], "j8": a[7]}
            self.joints_list.append(joints_dic)
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, bool(done), False, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # if self.ploting:
        #     joint_pos = self.data.qpos[self.model.jnt_qposadr][1:]
        #     joints_dic = {"j1": joint_pos[0], "j2": joint_pos[1], "j3": joint_pos[2], "j4": joint_pos[3],
        #                   "j5": joint_pos[4], "j6": joint_pos[5], "j7": joint_pos[6], "j8": joint_pos[7]}
        #     self.joints_list.append(joints_dic)
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.ploting and len(self.joints_list) > 0:
            output = Path("results") / f"{type(self).__name__}_joints.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.joints_list[0])
                writer.writeheader()
                writer.writerows(self.joints_list)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class AntEnvPlot(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,  ploting=False, render_mode=None):
        self.ploting = ploting
        if self.ploting:
            self.joints_list = []
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        if self.ploting:
            joints_dic = {"j1": a[0], "j2": a[1], "j3": a[2], "j4": a[3],
                          "j5": a[4], "j6": a[5], "j7": a[6], "j8": a[7]}
            self.joints_list.append(joints_dic)
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, bool(done), False, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        # if self.ploting:
        #     joint_pos = self.data.qpos[self.model.jnt_qposadr][1:]
        #     # joint_pos = self.data.qpos[self.model.jnt_dofadr][1:]
        #     joints_dic = {"j1": joint_pos[0], "j2": joint_pos[1], "j3": joint_pos[2], "j4": joint_pos[3],
        #                   "j5": joint_pos[4], "j6": joint_pos[5], "j7": joint_pos[6], "j8": joint_pos[7]}
        #     self.joints_list.append(joints_dic)
        return np.concatenate([
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
            np.clip(self.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.ploting and len(self.joints_list) > 0:
            output = Path("results") / f"{type(self).__name__}_joints.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.joints_list[0])
                writer.writeheader()
                writer.writerows(self.joints_list)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
