from dataclasses import astuple, dataclass
from datetime import datetime
from typing import List

import numpy as np
from numpy.typing import ArrayLike

from .utils import flatten_tuple


@dataclass
class Attitude:
    roll: float  # 以向右舷为正(degree)
    pitch: float  # 以向上为正(degree)
    yaw: float  # 以从北顺时针为正(degree)


@dataclass
class VehicleState:
    timestamp: float  # 时间戳(us)
    longitude: float  # 经度(degree)
    latitude: float  # 纬度(degree)
    elevation: float  # 海拔高度(m)
    attitude: Attitude


@dataclass
class Flight:
    date: datetime
    states: np.ndarray

    @classmethod
    def setup(cls, date: datetime, states: List[VehicleState]):
        return cls(date, np.array([flatten_tuple(astuple(s)) for s in states]))

    @classmethod
    def setup_from_series(
        cls,
        date: datetime,
        timestamp: ArrayLike,
        longitude: ArrayLike,
        latitude: ArrayLike,
        elevation: ArrayLike,
        attitude: ArrayLike,
    ):
        # transform ArrayLike to np.ndarray
        timestamp = np.array(timestamp)
        longitude = np.array(longitude)
        latitude = np.array(latitude)
        elevation = np.array(elevation)
        attitude = np.array(attitude)

        assert (
            timestamp.shape == (len(timestamp),)
            and longitude.shape == (len(longitude),)
            and latitude.shape == (len(latitude),)
            and elevation.shape == (len(elevation),)
            and attitude.shape == (len(attitude), 3)
        )
        assert (
            len(timestamp)
            == len(longitude)
            == len(latitude)
            == len(elevation)
            == len(attitude)
        )

        states = [
            VehicleState(
                timestamp=timestamp[i],
                longitude=longitude[i],
                latitude=latitude[i],
                elevation=elevation[i],
                attitude=attitude[i],
            )
            for i in range(len(timestamp))
        ]
        return cls.setup(date, states)
