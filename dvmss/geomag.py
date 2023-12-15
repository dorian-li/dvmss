from dataclasses import dataclass
from datetime import datetime

import numpy as np
import ppigrf


@dataclass
class GeomagneticElement:
    X: float  # 北向分量强度(nT)
    Y: float  # 东向分量强度(nT)
    Z: float  # 垂直向分量强度(nT)
    H: float  # 水平向分量强度(nT)
    D: float  # 磁偏角(degree)
    I: float  # 磁倾角(degree)
    F: float  # 总强度(nT)

    @classmethod
    def create_by_XYZ(cls, X, Y, Z):
        H = (X**2 + Y**2) ** 0.5
        D = np.arctan2(Y, X) * 180 / np.pi
        I = np.arctan2(Z, H) * 180 / np.pi
        F = (X**2 + Y**2 + Z**2) ** 0.5
        return cls(X, Y, Z, H, D, I, F)


class GeomagneticReferenceField:
    def get_magnetic_field(self, date, lon, lat, alt):
        if self.__class__ is GeomagneticReferenceField:
            raise NotImplementedError

        pass


class IGRF(GeomagneticReferenceField):
    def get_magnetic_field(self, date: datetime, lon: float, lat: float, alt: float):
        b_e, b_n, b_u = ppigrf.igrf(lon, lat, alt, date)  # 大地坐标系，z轴向上为正
        return GeomagneticElement.create_by_XYZ(b_n, b_e, -b_u)  # 地磁场坐标系，z轴向下为正
