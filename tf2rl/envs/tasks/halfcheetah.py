import numpy as np
from gymnasium import utils
from tf2rl.envs.tasks import research_mujoco as mujoco_env
from pathlib import Path
import csv

class HalfCheetahDisabledEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """HalfCheetah task using the model with restricted joint motion."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_disabled.xml', 5, render_mode=render_mode)
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HalfCheetahLongBodyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """HalfCheetah task using an elongated body model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_long_body.xml', 5, render_mode=render_mode)
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HalfCheetahLongLegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """HalfCheetah task using a longer-leg model."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_long_leg.xml', 5, render_mode=render_mode)
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HalfCheetahPlot(mujoco_env.MujocoEnv, utils.EzPickle):
    """HalfCheetah task retaining position coordinates for trajectory plots."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        self.joints_list = []
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5, render_mode=render_mode)
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
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
        joint_pos = self.data.qpos.flat[3:9]
        joints_dic = {"j1": joint_pos[0], "j2": joint_pos[1], "j3": joint_pos[2], "j4": joint_pos[3],
                      "j5": joint_pos[4], "j6": joint_pos[5]}
        self.joints_list.append(joints_dic)
        return np.concatenate([
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Positions excluding horizontal translation, followed by
                velocities.
        """
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
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.distance = self.model.stat.extent * 0.5
