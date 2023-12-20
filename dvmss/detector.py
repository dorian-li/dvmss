from dataclasses import astuple, dataclass
from enum import Enum, auto
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

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


# class Detector:
#     def __init__(
#         self,
#         id: int = None,
#         location: Optional[CartesianCoord] = None,
#         sensor_type: MagSensor = None,
#         sensor_data=pd.DataFrame(columns=[e for e in MagSensor]),
#         interactive=False,
#     ) -> None:
#         self.id = id
#         self.location = location
#         self.sensor_type = sensor_type
#         self.sensor_data = sensor_data
#         self.interactive = interactive
# use dataclass instead of class
@dataclass
class Detector:
    sensor_type: MagSensorType
    id: int = None
    location: Optional[CartesianCoord] = None
    sensor_data: pd.DataFrame = pd.DataFrame(columns=[e for e in MagSensor])
    loc_interactive: bool = False


@dataclass
class DetectorCollection:
    detectors: List[Detector]

    @classmethod
    def of(cls, *detectors: Detector):
        return cls(list(detectors))

    def get_locations_numpy(self):
        return np.array([astuple(detector.location) for detector in self.detectors])
