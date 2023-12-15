from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | pd.DataFrame | list


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
    states: List[VehicleState]

    @classmethod
    def setup(cls, states: List[VehicleState]):
        return cls(states=states)

    @classmethod
    def setup_from_series(
        cls,
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
        return cls.setup(states)
