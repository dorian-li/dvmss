import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from dvmss.geomag import IGRF, GeomagData, GeomagElem
from dvmss.utils import count_decimal_places

# 期望的地磁场数据计算自 https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml?useFullSite=true#igrfwmm
# location:
#  - longitude(deg): -113.64250
#  - latitude(deg): 60.10861
#  - elevation(km): 0.00 Mean sea level
# date: 2023.12.17
desired = {
    GeomagElem.NORTH: ["10542.1", "10542.1", "10542.1"],
    GeomagElem.EAST: ["2857.6", "2857.6", "2857.6"],
    GeomagElem.VERTICAL: ["56694.7", "56694.7", "56694.7"],
    GeomagElem.HORIZONTAL: ["10922.6", "10922.6", "10922.6"],
    GeomagElem.DECLINATION: ["15.16640", "15.16640", "15.16640"],
    GeomagElem.INCLINATION: ["79.09523", "79.09523", "79.09523"],
    GeomagElem.TOTAL: ["57737.3", "57737.3", "57737.3"],
}
decimals = {k: count_decimal_places(v[0]) - 1 for k, v in desired.items()}
desired = {k: np.array([float(vv) for vv in v]) for k, v in desired.items()}


def test_geomag_data_make_from_NED():
    data = GeomagData.make_from_NED(
        desired[GeomagElem.NORTH],
        desired[GeomagElem.EAST],
        desired[GeomagElem.VERTICAL],
    )
    assert isinstance(data, GeomagData)
    assert_allclose(data.query(GeomagElem.NORTH), desired[GeomagElem.NORTH])
