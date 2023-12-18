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
    VERTICAL = auto()  # 垂直向分量强度(nanoTesla)
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
        )

    @classmethod
    def make_from_NED(cls, north: ArrayLike, east: ArrayLike, vertical: ArrayLike):
        """由北向分量、东向分量、垂直向分量经验证后构造完整GeomagData"""

        north = np.array(north)
        east = np.array(east)
        vertical = np.array(vertical)

        if north.shape != east.shape or north.shape != vertical.shape:
            raise ValueError("north, east and vertical must be the same shape")

        if np.ndim(north) != 1:
            raise ValueError("north, east and vertical must be 1D")

        horizontal = np.sqrt(north**2 + east**2)
        declination = np.rad2deg(np.arctan2(east, north))
        inclination = np.rad2deg(np.arctan2(vertical, horizontal))
        total = np.sqrt(north**2 + east**2 + vertical**2)
        return cls.setup_from_pandas(
            pd.DataFrame(
                {
                    GeomagElem.NORTH: north,
                    GeomagElem.EAST: east,
                    GeomagElem.VERTICAL: vertical,
                    GeomagElem.HORIZONTAL: horizontal,
                    GeomagElem.DECLINATION: declination,
                    GeomagElem.INCLINATION: inclination,
                    GeomagElem.TOTAL: total,
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
