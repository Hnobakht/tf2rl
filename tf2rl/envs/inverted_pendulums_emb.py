import numpy as np
from gymnasium import utils
from tf2rl.envs import research_mujoco as mujoco_env

class InvertedDoublePendulumEnvLonglink(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_longlink.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, bool(done), False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

class InvertedDoublePendulumEnvLimitjoint(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_limitjoint.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, bool(done), False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]


class InvertedDoublePendulumEnvFat(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, render_mode=None):
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_fat.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, bool(done), False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]



class InvertedTriplePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, render_mode=None):
        raise NotImplemented
        mujoco_env.MujocoEnv.__init__(self, 'inverted_triple_pendulum.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        x, _, y = self.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = bool(y <= 1)
        return ob, r, bool(done), False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
