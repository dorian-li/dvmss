from dataclasses import astuple, dataclass, make_dataclass
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import List, Union

import numpy as np
import ppigrf
from numpy.typing import ArrayLike


class GeomagneticElement(IntEnum):
    X = 0  # 北向分量强度
    Y = auto()  # 东向分量强度
    Z = auto()  # 垂直向分量强度
    H = auto()  # 水平向分量强度
    D = auto()  # 磁偏角
    I = auto()  # 磁倾角
    F = auto()  # 总强度


@dataclass
class GeomagneticField:
    X: float  # 北向分量强度(nT)
    Y: float  # 东向分量强度(nT)
    Z: float  # 垂直向分量强度(nT)
    H: float  # 水平向分量强度(nT)
    D: float  # 磁偏角(degree)
    I: float  # 磁倾角(degree)
    F: float  # 总强度(nT)

    @classmethod
    def make_from_XYZ(cls, X: float, Y: float, Z: float):
        H = (X**2 + Y**2) ** 0.5
        D = np.arctan2(Y, X) * 180 / np.pi
        I = np.arctan2(Z, H) * 180 / np.pi
        F = (X**2 + Y**2 + Z**2) ** 0.5
        return cls(X, Y, Z, H, D, I, F)


class GeomagneticFieldCollection:
    def __init__(self, fields: List[GeomagneticField]):
        # fields transform to numpy.ndarray, shape=(len(fields), len(GeomagneticElement))
        self.fields = np.array([astuple(f) for f in fields])

    def get_element_array(self, *element: GeomagneticElement):
        return make_dataclass(
            "GeomagneticFieldSubset",
            [(f"{e.name}", np.ndarray) for e in element],
        )(*[self.fields[:, e] for e in element])

    @classmethod
    def make_from_XYZ(cls, X: ArrayLike, Y: ArrayLike, Z: ArrayLike):
        fields = [GeomagneticField.make_from_XYZ(x, y, z) for x, y, z in zip(X, Y, Z)]
        return cls(fields)


class GeomagneticReferenceField:
    def get_magnetic_field(self, date, lon, lat, alt) -> GeomagneticElement:
        if self.__class__ is GeomagneticReferenceField:
            raise NotImplementedError

        pass


class IGRF(GeomagneticReferenceField):
    def get_magnetic_field(
        self, date: datetime, lon: float, lat: float, alt: float
    ) -> GeomagneticFieldCollection:
        b_e, b_n, b_u = ppigrf.igrf(lon, lat, alt, date)  # 大地坐标系，z轴向上为正
        return GeomagneticElement.create_by_XYZ(b_n, b_e, -b_u)  # 地磁场坐标系，z轴向下为正
