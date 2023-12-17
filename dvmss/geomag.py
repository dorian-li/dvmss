from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, make_dataclass
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import List, Union

import numpy as np
import pandas as pd
import ppigrf
from numpy.typing import ArrayLike
from SimPEG.utils.mat_utils import dip_azimuth2cartesian


class GeomagElem(Enum):
    """地磁七要素，与对应标记数组名"""

    NORTH = "north"  # 北向分量强度
    EAST = "east"  # 东向分量强度
    VERTICAL = "vertical"  # 垂直向分量强度
    HORIZONTAL = "horizontal"  # 水平向分量强度
    DECLINATION = "declination"  # 磁偏角
    INCLINATION = "inclination"  # 磁倾角
    TOTAL = "total"  # 总强度


class GeomagData:
    """一组地磁场数据，每个数据点包含地磁七要素，存储底层为pandas.DataFrame"""

    def __init__(self):
        self._data = pd.DataFrame(columns=[e.value for e in GeomagElem])

    @classmethod
    def setup_from_pandas(cls, data: pd.DataFrame):
        """由pandas.DataFrame经验证后构造GeomagData"""

        try:
            assert data.shape[1] == len(GeomagElem)
        except AssertionError as e:
            raise ValueError(
                f"data shape must be (n, {len(GeomagElem)}): {data.shape}"
            ) from e

        try:
            assert set(data.columns) == set([e.value for e in GeomagElem])
        except AssertionError as e:
            raise ValueError(
                f"data columns must be {set([e.value for e in GeomagElem])}: {set(data.columns)}"
            ) from e
        instance = cls()
        instance._data = data
        return instance

    def query(self, *elements: GeomagElem):
        """查询一组指定的地磁场要素（某一个或多个要素)"""
        return self._data[[e.value for e in elements]]

    def get_orientations(self):
        """获取地磁场方向向量"""
        return dip_azimuth2cartesian(
            self._data[GeomagElem.INCLINATION],
            self._data[GeomagElem.DECLINATION],
        )

    @classmethod
    def make_from_NED(cls, north: ArrayLike, east: ArrayLike, vertical: ArrayLike):
        """由北向分量、东向分量、垂直向分量经验证后构造完整GeomagData"""

        north = np.array(north)
        east = np.array(east)
        vertical = np.array(vertical)
        try:
            assert north.shape == east.shape == vertical.shape
        except AssertionError as e:
            raise ValueError("north, east and vertical must be the same shape") from e
        try:
            assert np.ndim(north) == 1
        except AssertionError as e:
            raise ValueError("north, east and vertical must be 1D") from e

        horizontal = np.sqrt(north**2 + east**2)
        declination = np.rad2deg(np.arctan2(east, north))
        inclination = np.rad2deg(np.arctan2(vertical, horizontal))
        total = np.sqrt(north**2 + east**2 + vertical**2)
        return cls.setup_from_pandas(
            pd.DataFrame(
                {
                    GeomagElem.NORTH.value: north,
                    GeomagElem.EAST.value: east,
                    GeomagElem.VERTICAL.value: vertical,
                    GeomagElem.HORIZONTAL.value: horizontal,
                    GeomagElem.DECLINATION.value: declination,
                    GeomagElem.INCLINATION.value: inclination,
                    GeomagElem.TOTAL.value: total,
                }
            )
        )


class GeomagRefField(ABC):
    @classmethod
    @abstractmethod
    def query(cls, date, lon, lat, alt) -> GeomagData:
        """查询某一日期、某一地理位置的参考地磁场数据"""
        if cls.__class__ == GeomagRefField:
            raise NotImplementedError(
                "GeomagneticReferenceField is an abstract class, "
                "please use one of its subclasses"
            )
        pass


class IGRF(GeomagRefField):
    @classmethod
    def query(
        cls,
        date: datetime,
        longitude: ArrayLike,
        latitude: ArrayLike,
        elevation: ArrayLike,
    ) -> GeomagData:
        """通过IGRF模型，查询某一日期、某一地理位置的参考地磁场数据"""
        longitude = np.array(longitude)
        latitude = np.array(latitude)
        elevation = np.array(elevation)

        b_e, b_n, b_u = ppigrf.igrf(
            longitude,
            latitude,
            elevation,
            date,
        )  # 大地坐标系，z轴向上为正
        # 默认输出shape为(1, n)，将其转换为(n,)
        b_e = np.array(b_e).flatten()
        b_n = np.array(b_n).flatten()
        b_u = np.array(b_u).flatten()
        return GeomagData.make_from_NED(b_n, b_e, -b_u)  # 地磁场坐标系，z轴向下为正
