import numpy as np
from gymnasium import utils
from tf2rl.envs.tasks import research_mujoco as mujoco_env
from tf2rl.envs.dynamic_mjc.mjc_models import point_mass_maze


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Point-mass navigation task in a configurable MuJoCo maze."""
    def __init__(self, direction=1, maze_length=0.6, sparse_reward=False, no_reward=False, episode_length=100, render_mode=None):
        """Load the task's MuJoCo model and set the number of simulation frames per action."""
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length

        self.episode_length = 0

        model = point_mass_maze(direction=self.direction, length=self.length)
        mujoco_env.MujocoEnv.__init__(self, model, 5, render_mode=render_mode)

    # def _step(self, a):
    def step(self, a):
        """Advance the task with the given action and return observation, reward, episode flags, and
        info.
        """
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")

        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, False, bool(done), dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        """Configure the camera for this MuJoCo task."""
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 1.0

    def reset_model(self):
        """Randomize the simulator's initial state and return the resulting observation."""
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        """Assemble the task observation from the current simulator state."""
        return np.concatenate([self.get_body_com("particle")[:2],
                               [np.linalg.norm(self.get_body_com("particle")[:2] - self.get_body_com("target")[:2])]
                               ])

    def plot_trajs(self, *args, **kwargs):
        """Plot two-dimensional trajectory paths in the maze."""
        pass
