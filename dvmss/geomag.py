from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, make_dataclass
from datetime import datetime
from enum import Enum, auto
from typing import List, Union

import numpy as np
import pandas as pd
import ppigrf
from numpy.typing import ArrayLike
from SimPEG.utils.mat_utils import dip_azimuth2cartesian


class GeomagElem(Enum):
    """地磁七要素"""

    NORTH = auto()  # 北向分量强度(nanoTesla)
    EAST = auto()  # 东向分量强度(nanoTesla)
    VERTICAL = auto()  # 垂直向分量强度(nanoTesla)，向下为正
    HORIZONTAL = auto()  # 水平向分量强度(nanoTesla)
    DECLINATION = auto()  # 磁偏角(degree)
    INCLINATION = auto()  # 磁倾角(degree)
    TOTAL = auto()  # 总强度(nanoTesla)


class GeomagData:
    """一组地磁场数据，每个数据点包含地磁七要素，存储底层为pandas.DataFrame"""

    def __init__(self):
        self._data = pd.DataFrame(columns=[e for e in GeomagElem])

    @classmethod
    def setup_from_pandas(cls, data: pd.DataFrame):
        """由pandas.DataFrame经验证后构造GeomagData"""

        if data.shape[1] != len(GeomagElem):
            raise ValueError(f"data shape must be (n, {len(GeomagElem)}): {data.shape}")

        if set(data.columns) != set([e for e in GeomagElem]):
            raise ValueError(
                f"data columns must be {set([e.value for e in GeomagElem])}: {set(data.columns)}"
            )

        instance = cls()
        instance._data = data
        return instance

    def query(self, *elements: GeomagElem):
        """查询一组指定的地磁场要素（某一个或多个要素)"""
        if len(elements) == 1:
            return self._data[elements[0]]
        return self._data[[e for e in elements]]

    def get_orientations(self):
        """获取地磁场方向向量"""
        return dip_azimuth2cartesian(
            self._data[GeomagElem.INCLINATION],
            self._data[GeomagElem.DECLINATION],
        )  # ENU

    @classmethod
    def make_from_ENU(cls, easting: ArrayLike, northing: ArrayLike, upward: ArrayLike):
        """由东向分量、北向分量、垂直向上分量经验证后构造完整GeomagData"""

        easting = np.array(easting)
        northing = np.array(northing)
        downward = np.array(-upward)

        if northing.shape != easting.shape or northing.shape != downward.shape:
            raise ValueError("north, east and vertical must be the same shape")

        if np.ndim(northing) != 1:
            raise ValueError("north, east and vertical must be 1D")

        horizontal = np.sqrt(northing**2 + easting**2)
        declination = np.rad2deg(np.arctan2(easting, northing))
        inclination = np.rad2deg(np.arctan2(downward, horizontal))
        total = np.sqrt(northing**2 + easting**2 + downward**2)
        return cls.setup_from_pandas(
            pd.DataFrame(
                {
                    GeomagElem.NORTH: northing,
                    GeomagElem.EAST: easting,
                    GeomagElem.VERTICAL: downward,
                    GeomagElem.HORIZONTAL: horizontal,
                    GeomagElem.DECLINATION: declination,
                    GeomagElem.INCLINATION: inclination,
                    GeomagElem.TOTAL: total,
                }  # NED
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
        elevation: ArrayLike,  # 米
    ) -> GeomagData:
        """通过IGRF模型，查询某一日期、某一地理位置的参考地磁场数据"""
        longitude = np.array(longitude)
        latitude = np.array(latitude)
        elevation = np.array(elevation) / 1000  # 单位转换为千米

        b_e, b_n, b_u = ppigrf.igrf(
            longitude,
            latitude,
            elevation,
            date,
        )  # 东北天坐标系
        # 默认输出shape为(1, n)，将其转换为(n,)
        b_e = np.array(b_e).flatten()
        b_n = np.array(b_n).flatten()
        b_u = np.array(b_u).flatten()
        return GeomagData.make_from_ENU(b_e, b_n, b_u)
