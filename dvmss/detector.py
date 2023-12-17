from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np

from .utils import CartesianCoord

MagSensor = Literal["vector", "scalar"]


@dataclass
class MagScalar:
    timestamp: float
    bt: float


@dataclass
class MagScalarSensor:
    data: List[MagScalar]


@dataclass
class MagVector:
    timestamp: float
    bx: float
    by: float
    bz: float


@dataclass
class MagVectorSensor:
    data: List[MagVector]


@dataclass
class Detector:
    id: Optional[str] = None
    location: Optional[CartesianCoord] = None
    sensor_type: MagSensor
    noise_level: float
    sensor: Optional[Union[MagScalarSensor, MagVectorSensor]] = None

    @classmethod
    def setup(
        cls,
        location: Optional[CartesianCoord],
        sensor_type: MagSensor,
        noise_level: float,
    ):
        return cls(location, sensor_type, noise_level)

    @classmethod
    def setup_with_interactive(cls, sensor_type: MagSensor, noise_level: float):
        return cls.setup(None, sensor_type, noise_level)


@dataclass
class DetectorCollection:
    detectors: List[Detector]

    @classmethod
    def of(cls, *detectors: Detector):
        return cls(list(detectors))
