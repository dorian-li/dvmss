from dataclasses import astuple, dataclass
from enum import Enum, auto
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .utils import CartesianCoord


class MagSensorType(Enum):
    """磁传感器类型"""

    SCALAR = auto()
    VECTOR = auto()


class MagSensor(Enum):
    """磁传感器分量，以及详细干扰情况"""

    B_X = auto()
    B_Y = auto()
    B_Z = auto()
    TMI = auto()
    PREM_X = auto()
    PREM_Y = auto()
    PREM_Z = auto()
    PREM_TMI = auto()
    INDUCED_X = auto()
    INDUCED_Y = auto()
    INDUCED_Z = auto()
    INDUCED_TMI = auto()
    GEO_X = auto()
    GEO_Y = auto()
    GEO_Z = auto()
    GEO_T = auto()


@dataclass
class Detector:
    sensor_type: MagSensorType
    id: int = None
    location: Optional[CartesianCoord] = None
    sensor_data: pd.DataFrame = pd.DataFrame(columns=[e for e in MagSensor])
    loc_interactive: bool = False

    def assign_sensor_data(self, component: MagSensor, data: ArrayLike):
        self.sensor_data[component] = data


@dataclass
class DetectorCollection:
    items: List[Detector]

    @classmethod
    def of(cls, *detectors: Detector):
        return cls(list(detectors))

    def get_locations_numpy(self):
        return np.array([astuple(detector.location) for detector in self.items])

    def __iter__(self):
        return iter(self.items)
