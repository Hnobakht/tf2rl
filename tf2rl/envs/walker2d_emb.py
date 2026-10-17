import numpy as np
from gymnasium import utils
from tf2rl.envs import research_mujoco as mujoco_env

class Walker2dLongLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    """Walker2d task using a longer-leg model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action."""
        mujoco_env.MujocoEnv.__init__(self, "walker2d_long_leg.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.
        """
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, bool(done), False, {}

    def _get_obs(self):
        """Assemble the task observation from the current simulator state."""
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation."""
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task."""
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


class Walker2dDisabledEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    """Walker2d task using the model with restricted joint motion."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action."""
        mujoco_env.MujocoEnv.__init__(self, "walker2d_disabled.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.
        """
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, bool(done), False, {}

    def _get_obs(self):
        """Assemble the task observation from the current simulator state."""
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation."""
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task."""
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

class Walker2dFatEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    """Walker2d task using the thicker-body model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action."""
        mujoco_env.MujocoEnv.__init__(self, "walker2d_fat.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.
        """
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, bool(done), False, {}

    def _get_obs(self):
        """Assemble the task observation from the current simulator state."""
        qpos = self.data.qpos
        qvel = self.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation."""
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task."""
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
