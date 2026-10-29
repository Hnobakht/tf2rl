import numpy as np
from gymnasium import utils
from tf2rl.envs.tasks import research_mujoco as mujoco_env

class InvertedDoublePendulumEnvLonglink(mujoco_env.MujocoEnv, utils.EzPickle):

    """Double-pendulum balance task using longer links."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_longlink.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            action (array_like): Single action in the environment action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

class InvertedDoublePendulumEnvLimitjoint(mujoco_env.MujocoEnv, utils.EzPickle):

    """Double-pendulum balance task with restricted joint motion."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_limitjoint.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            action (array_like): Single action in the environment action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]


class InvertedDoublePendulumEnvFat(mujoco_env.MujocoEnv, utils.EzPickle):

    """Double-pendulum balance task using the thicker-link model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'inverted_double_pendulum_fat.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            action (array_like): Single action in the environment action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]



class InvertedTriplePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    """Cart-based balance task with three pendulum links."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            NoReturn: The base or unsupported implementation always raises NotImplementedError.
        """
        raise NotImplemented
        mujoco_env.MujocoEnv.__init__(self, 'inverted_triple_pendulum.xml', 5, render_mode=render_mode)
        utils.EzPickle.__init__(self)

    def step(self, action):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.

        Args:
            action (array_like): Single action in the environment action space.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: Observation, reward, terminated, truncated, and
                reward diagnostics. True termination is reported separately; truncation is False and
                external TimeLimit wrappers enforce the horizon.
        """
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        return np.concatenate([
            self.data.qpos[:1],  # cart x pos
            np.sin(self.data.qpos[1:]),  # link angles
            np.cos(self.data.qpos[1:]),
            np.clip(self.data.qvel, -10, 10),
            np.clip(self.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Cart position, joint-angle sine/cosine features,
                clipped velocities, and clipped constraint forces.
        """
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
