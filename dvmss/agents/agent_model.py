from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyvista as pv
from metalpy.scab.modelling.transform import Rotation
from metalpy.utils.bounds import Bounds

AGENT_RES = Path(__file__).resolve().parent / Path("resources")
AEROCRAFT_RES = AGENT_RES / Path("aerocraft")


@dataclass
class Agent:
    model: Any
    scale: float
    northward: Rotation

    def plot(self):
        pass


def cessna_172():
    northward = Rotation(180, 0, 0, degrees=True, seq="zyx")
    model_path = AEROCRAFT_RES / Path("cessna_172.stl")
    model = pv.read(model_path)
    model_bounds = Bounds(model.bounds)  # x:wing, y:body
    wing_true = 11  # unit: m
    scale = wing_true / model_bounds.extent[0]
    return Agent(model, scale, northward)


if __name__ == "__main__":
    model = cessna_172()
    print(model)
