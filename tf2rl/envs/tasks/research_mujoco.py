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


def _vector(value):
    """Parse the simple numeric expressions used by the legacy MuJoCo assets."""
    values = []
    for token in value.split():
        factors = token.split("/")
        number = float(factors[0])
        for denominator in factors[1:]:
            number /= float(denominator)
        values.append(number)
    return np.asarray(values, dtype=np.float64)


def _format_vector(value):
    """Serialize numeric vector entries as space-separated MJCF values."""
    return " ".join(f"{number:.17g}" for number in value)


def _convert_global_coordinates(xml):
    """Convert translation-only legacy global coordinates to local MJCF.

    The Hopper and Walker2d research assets predate MuJoCo 2.3 and use the
    removed ``coordinate=global`` compiler mode.  Their body frames have no
    rotations, so the old compiler conversion is exactly a parent-origin
    subtraction for positions and ``fromto`` endpoints.
    """
    compiler = xml.find("compiler")
    if compiler is None or compiler.attrib.get("coordinate", "local") != "global":
        return
    compiler.attrib.pop("coordinate")

    def convert_element(element, origin):
        """Convert an element's position and segment endpoints relative to the body origin."""
        if "pos" in element.attrib:
            element.attrib["pos"] = _format_vector(_vector(element.attrib["pos"]) - origin)
        if "fromto" in element.attrib:
            endpoints = _vector(element.attrib["fromto"]).reshape(2, 3) - origin
            element.attrib["fromto"] = _format_vector(endpoints.reshape(-1))

    def convert_body(body, parent_origin):
        """Convert an unrotated body and its descendants from global to local coordinates."""
        if {'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'} & body.attrib.keys():
            raise ValueError('Global-coordinate conversion only supports unrotated body frames')
        body_origin = _vector(body.attrib.get("pos", "0 0 0"))
        body.attrib["pos"] = _format_vector(body_origin - parent_origin)
        for child in body:
            if child.tag == "body":
                convert_body(child, body_origin)
            else:
                convert_element(child, body_origin)

    worldbody = xml.find("worldbody")
    if worldbody is not None:
        world_origin = np.zeros(3, dtype=np.float64)
        for child in worldbody:
            if child.tag == "body":
                convert_body(child, world_origin)
            else:
                convert_element(child, world_origin)


class MujocoEnv(GymnasiumMujocoEnv):
    """Run historical research task definitions on the native Gymnasium MuJoCo backend."""
    def __init__(self, model_path, frame_skip, render_mode=None, **kwargs):
        """Convert research XML and initialize the simulator, rendering mode, and observation space.
        """
        if hasattr(model_path, "root"):
            stream = io.StringIO()
            model_path.root.write(stream)
            self._research_xml = stream.getvalue()
            path = (Path(__file__).resolve().parents[1] / "assets") / "ant.xml"
        else:
            path = Path(model_path)
            if not path.is_absolute():
                path = (Path(__file__).resolve().parents[1] / "assets") / path
            self._research_xml = path.read_text()
        xml = ET.fromstring(self._research_xml)
        compiler = xml.find("compiler")
        if compiler is not None:
            _convert_global_coordinates(xml)
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
        """Build the MuJoCo model and data from the converted research XML."""
        model = mujoco.MjModel.from_xml_string(self._research_xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data

    def do_simulation(self, ctrl, n_frames):
        """Advance MuJoCo by the requested frames and render in human mode."""
        super().do_simulation(ctrl, n_frames)
        if self.render_mode == "human":
            self.render()
