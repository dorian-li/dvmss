from dataclasses import dataclass

import numpy as np
import ppigrf
from SimPEG.utils.mat_utils import dip_azimuth2cartesian


@dataclass
class GeomagneticField:
    total: float
    inclination: float
    declination: float
    horizontal_intensity: float
    northward_intensity: float
    eastward_intensity: float
    vertical_intensity: float

    @property
    def orientation(self):
        return dip_azimuth2cartesian(self.inclination, self.declination)

    @classmethod
    def make(cls, lon, lat, altitude, date, total):
        total = np.asarray(total)

        be, bn, bu = ppigrf.igrf(lon, lat, altitude, date)
        be = be.squeeze()
        bn = bn.squeeze()
        bu = bu.squeeze()
        bt = np.linalg.norm([be, bn, bu], axis=0)

        cos_e = be / bt
        cos_n = bn / bt
        cos_d = -bu / bt

        declination = np.rad2deg(np.arctan2(be, bn))
        bh = np.sqrt(be**2 + bn**2)
        inclination = np.rad2deg(np.arctan2(-bu, bh))

        eastward_intensity = total * cos_e
        northward_intensity = total * cos_n
        vertical_intensity = total * cos_d
        horizontal_intensity = np.sqrt(
            eastward_intensity**2 + northward_intensity**2
        )
        return [
            cls(
                total[i],
                inclination[i],
                declination[i],
                horizontal_intensity[i],
                northward_intensity[i],
                eastward_intensity[i],
                vertical_intensity[i],
            )
            for i in range(len(lon))
        ]


if __name__ == "__main__":
    from datetime import datetime

    lon = [-85.3232, -85.323]
    lat = [51.2538, 51.253]
    above_sea_level = np.ones_like(lon) * 4
    date = datetime(2023, 8, 26)
    total = [56451.7, 53145]
    geo = GeomagneticField.make(lon, lat, above_sea_level, date, total)
    print(geo)
