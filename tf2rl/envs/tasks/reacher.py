import numpy as np
from gymnasium import utils
from tf2rl.envs.tasks import research_mujoco as mujoco_env

class ReacherSimpleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Reacher task exposing a simplified observation vector."""
    def __init__(self, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action.

        Args:
            render_mode (str or None): Gymnasium rendering mode such as human or rgb_array.

        Returns:
            None: Initializes the instance.
        """
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2, render_mode=render_mode)

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
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward = reward_dist
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, bool(done), False, dict(reward_dist=reward_dist)

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task.

        Returns:
            None.
        """
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation.

        Returns:
            np.ndarray: Flat observation vector. Joint-angle cosine/sine features, remaining positions,
                joint velocities, and fingertip-to-target displacement.
        """
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel  + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """Assemble the task observation from the current simulator state.

        Returns:
            np.ndarray: Flat observation vector. Joint-angle cosine/sine features, remaining positions,
                joint velocities, and fingertip-to-target displacement.
        """
        theta = self.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.data.qpos.flat[2:],
            self.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
