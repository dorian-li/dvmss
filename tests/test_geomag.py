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


def test_geomag_data_setup_from_pandas_valid_data():
    """GeomagData.setup_from_pandas应该能够正确设置GeomagData实例的数据"""
    desired_df = pd.DataFrame(desired)
    geomags = GeomagData.setup_from_pandas(desired_df)
    assert_allclose(geomags._data, desired_df)


def test_geomag_data_setup_from_pandas_invalid_shape():
    """GeomagData.setup_from_pandas应该能够正确设置GeomagData实例的数据"""
    desired_df = pd.DataFrame(desired)
    invalid_df = desired_df.drop(desired_df.columns[-1], axis=1)
    with pytest.raises(ValueError):
        geomags = GeomagData.setup_from_pandas(invalid_df)


def test_geomag_data_setup_from_pandas_invalid_element():
    """GeomagData.setup_from_pandas应该能够正确设置GeomagData实例的数据"""
    desired_df = pd.DataFrame(desired)
    invalid_df = desired_df.rename(
        columns={GeomagElem.NORTH: "north", GeomagElem.EAST: "east"}
    )
    with pytest.raises(ValueError):
        geomags = GeomagData.setup_from_pandas(invalid_df)


def test_geomag_data_make_from_NEU():
    """GeomagData.make_from_XYZ应该能够根据XYZ分量数组，正确构造拥有完整地磁七要素的数组"""
    geomags = GeomagData.make_from_ENU(
        desired[GeomagElem.EAST],
        desired[GeomagElem.NORTH],
        -1 * desired[GeomagElem.VERTICAL],
    )
    assert isinstance(geomags, GeomagData)
    assert_almost_equal(
        geomags._data[GeomagElem.NORTH],
        desired[GeomagElem.NORTH],
        decimals[GeomagElem.NORTH],
    )
    assert_almost_equal(
        geomags._data[GeomagElem.EAST],
        desired[GeomagElem.EAST],
        decimals[GeomagElem.EAST],
    )
    assert_almost_equal(
        geomags._data[GeomagElem.VERTICAL],
        desired[GeomagElem.VERTICAL],
        decimals[GeomagElem.VERTICAL],
    )
    assert_almost_equal(
        geomags._data[GeomagElem.HORIZONTAL],
        desired[GeomagElem.HORIZONTAL],
        decimals[GeomagElem.HORIZONTAL],
    )
    assert_almost_equal(
        geomags._data[GeomagElem.DECLINATION],
        desired[GeomagElem.DECLINATION],
        decimals[GeomagElem.DECLINATION],
    )
    assert_almost_equal(
        geomags._data[GeomagElem.INCLINATION],
        desired[GeomagElem.INCLINATION],
        decimals[GeomagElem.INCLINATION],
    )
    assert_almost_equal(
        geomags._data[GeomagElem.TOTAL],
        desired[GeomagElem.TOTAL],
        decimals[GeomagElem.TOTAL],
    )


def test_geomag_data_query():
    """GeomagData.query应该能够根据地磁七要素的名称，查询对应的数据"""
    geomags = GeomagData.make_from_ENU(
        desired[GeomagElem.EAST],
        desired[GeomagElem.NORTH],
        -1 * desired[GeomagElem.VERTICAL],
    )
    assert_allclose(
        geomags.query(GeomagElem.NORTH),
        desired[GeomagElem.NORTH],
    )
    assert_almost_equal(
        geomags.query(GeomagElem.DECLINATION, GeomagElem.INCLINATION),
        np.column_stack(
            (
                desired[GeomagElem.DECLINATION],
                desired[GeomagElem.INCLINATION],
            )
        ),
        min(decimals[GeomagElem.DECLINATION], decimals[GeomagElem.INCLINATION]),
    )
