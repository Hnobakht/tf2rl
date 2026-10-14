"""Native MuJoCo runtime for the original embodiment-mismatch task definitions.

Only the runtime/API is adapted. XML geometry, rewards, reset distributions and
observation ordering stay in the original task modules.
"""
from pathlib import Path
import io
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv as GymnasiumMujocoEnv
import mujoco
import numpy as np


class MujocoEnv(GymnasiumMujocoEnv):
    def __init__(self, model_path, frame_skip, render_mode=None, **kwargs):
        if hasattr(model_path, "root"):
            stream = io.StringIO()
            model_path.root.write(stream)
            self._research_xml = stream.getvalue()
            path = Path(__file__).with_name("assets") / "ant.xml"
        else:
            path = Path(model_path)
            if not path.is_absolute():
                path = Path(__file__).with_name("assets") / path
            self._research_xml = path.read_text()
        xml = ET.fromstring(self._research_xml)
        compiler = xml.find("compiler")
        if compiler is not None:
            # 'local' is now the default; global coordinates would need a conversion.
            coordinate = compiler.attrib.pop("coordinate", "local")
            if coordinate != "local":
                raise ValueError("Research MuJoCo assets must use local coordinates")
        self._research_xml = ET.tostring(xml, encoding="unicode")
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
        super().__init__(str(path), frame_skip,
                         observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float64),
                         render_mode=render_mode, **kwargs)
        # Gym's old base inferred observation width using a random step. Derive it
        # directly instead, without consuming RNG or changing the initial state.
        obs = self._get_obs()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=obs.dtype)

    def _initialize_simulation(self):
        model = mujoco.MjModel.from_xml_string(self._research_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def do_simulation(self, ctrl, n_frames):
        super().do_simulation(ctrl, n_frames)
        if self.render_mode == "human":
            self.render()
