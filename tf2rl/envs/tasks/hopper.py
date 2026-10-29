import numpy as np
from gymnasium import utils
from tf2rl.envs.tasks import research_mujoco as mujoco_env

class HopperEnvLongleg(mujoco_env.MujocoEnv, utils.EzPickle):
    """Hopper task using the longer-leg model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'hopper_longleg.xml', 4, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            a (np.ndarray): Continuous control vector in the task action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, bool(done), False, {}

    def _get_obs(self):
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation and clipped
                joint velocities.
        """
        return np.concatenate([
            self.data.qpos.flat[1:],
            np.clip(self.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation and clipped
                joint velocities.
        """
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

class HopperEnvFat(mujoco_env.MujocoEnv, utils.EzPickle):
    """Hopper task using the thicker-body model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'hopper_fat.xml', 4, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            a (np.ndarray): Continuous control vector in the task action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, bool(done), False, {}

    def _get_obs(self):
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation and clipped
                joint velocities.
        """
        return np.concatenate([
            self.data.qpos.flat[1:],
            np.clip(self.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation and clipped
                joint velocities.
        """
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

class HopperEnvFraction(mujoco_env.MujocoEnv, utils.EzPickle):
    """Hopper task using the alternate friction model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'hopper_frac.xml', 4, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            a (np.ndarray): Continuous control vector in the task action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, bool(done), False, {}

    def _get_obs(self):
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation and clipped
                joint velocities.
        """
        return np.concatenate([
            self.data.qpos.flat[1:],
            np.clip(self.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation and clipped
                joint velocities.
        """
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
