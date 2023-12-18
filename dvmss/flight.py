from dataclasses import astuple, dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from dvmss.utils import flatten_tuple


class VehicleState(Enum):
    TIMESTAMP = auto()  # 时间戳(us)

    # 地理坐标系
    LONGITUDE = auto()  # 经度(degree)
    LATITUDE = auto()  # 纬度(degree)
    ELEVATION = auto()  # 海拔高度(m)

    # 飞行姿态
    ROLL = auto()  # 以向右舷为正(degree)
    PITCH = auto()  # 以向上为正(degree)
    YAW = auto()  # 以从北顺时针为正(degree)


class Flight:
    def __init__(self) -> None:
        self._date: datetime = None
        self._states = pd.DataFrame(columns=[s for s in VehicleState])

    @property
    def date(self):
        return self._date

    @classmethod
    def setup_from_pandas(cls, date: datetime, states: pd.DataFrame):
        """由pandas.DataFrame经验证后构造Flight"""
        if not isinstance(date, datetime):
            raise ValueError(f"date must be datetime: {date}")

        if states.shape[1] != len(VehicleState):
            raise ValueError(
                f"states shape must be (n, {len(VehicleState)}): {states.shape}"
            )

        if set(states.columns) != set([s for s in VehicleState]):
            raise ValueError(
                f"states columns must be {set([s for s in VehicleState])}: {set(states.columns)}"
            )

        instance = cls()
        instance._date = date
        instance._states = states
        return instance

    def query(self, *states: VehicleState):
        """查询一组指定的载体状态（某一个或多个载体状态)"""
        if len(states) == 1:
            return self._states[states[0]]
        return self._states[[s for s in states]]

    @classmethod
    def setup_from_series(
        cls,
        date: datetime,
        timestamp: ArrayLike,
        longitude: ArrayLike,
        latitude: ArrayLike,
        elevation: ArrayLike,
        roll: ArrayLike,
        pitch: ArrayLike,
        yaw: ArrayLike,
    ):
        timestamp = np.array(timestamp)
        longitude = np.array(longitude)
        latitude = np.array(latitude)
        elevation = np.array(elevation)
        roll = np.array(roll)
        pitch = np.array(pitch)
        yaw = np.array(yaw)

        if not isinstance(date, datetime):
            raise ValueError(f"date must be datetime: {date}")

        if not (
            timestamp.shape
            == longitude.shape
            == latitude.shape
            == elevation.shape
            == roll.shape
            == pitch.shape
            == yaw.shape
        ):
            raise ValueError(
                f"timestamp, longitude, latitude, elevation, roll, pitch, yaw must be the same shape"
            )

        states = pd.DataFrame(
            {
                VehicleState.TIMESTAMP: timestamp,
                VehicleState.LONGITUDE: longitude,
                VehicleState.LATITUDE: latitude,
                VehicleState.ELEVATION: elevation,
                VehicleState.ROLL: roll,
                VehicleState.PITCH: pitch,
                VehicleState.YAW: yaw,
            }
        )
        return cls.setup_from_pandas(date, states)
