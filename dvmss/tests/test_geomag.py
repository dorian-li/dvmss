import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import is_dataclass

import numpy as np
from numpy.testing import assert_allclose

from dvmss.geomag import (
    GeomagneticElement,
    GeomagneticField,
    GeomagneticFieldCollection,
)


def test_geomagnetic_element():
    """GeomagneticElement应该是枚举类型，对应底层存储数组的磁场元素索引"""
    assert GeomagneticElement.X == 0
    assert GeomagneticElement.Y == 1
    assert GeomagneticElement.Z == 2
    assert GeomagneticElement.H == 3
    assert GeomagneticElement.D == 4
    assert GeomagneticElement.I == 5
    assert GeomagneticElement.F == 6


def test_geomagnetic_element_and_geomagnetic_field_member_consistency():
    """GeomagneticElement和GeomagneticField的成员应该一致"""
    geomagnetic_element_members = [
        member.name for member in GeomagneticElement.__members__.values()
    ]
    geomagnetic_field_members = [
        member_name for member_name in GeomagneticField.__dataclass_fields__.keys()
    ]

    assert geomagnetic_element_members == geomagnetic_field_members


def test_geomagnetic_field_make_from_XYZ():
    """GeomagneticField.make_from_XYZ应该能够根据XYZ分量，正确计算剩余磁场元素"""
    X = 1.0
    Y = 2.0
    Z = 3.0

    field = GeomagneticField.make_from_XYZ(X, Y, Z)

    assert field.X == X
    assert field.Y == Y
    assert field.Z == Z
    assert field.H == np.sqrt(X**2 + Y**2)
    assert field.D == np.arctan2(Y, X) * 180 / np.pi
    assert field.I == np.arctan2(Z, np.sqrt(X**2 + Y**2)) * 180 / np.pi
    assert field.F == np.sqrt(X**2 + Y**2 + Z**2)


def test_geomagnetic_field_collection_make_from_XYZ():
    """GeomagneticFieldCollection.make_from_XYZ应该能够根据XYZ分量数组，正确构造拥有完整地磁七要素的数组"""
    X = np.array([1.0, 2.0, 3.0])
    Y = np.array([2.0, 3.0, 4.0])
    Z = np.array([3.0, 4.0, 5.0])

    collection = GeomagneticFieldCollection.make_from_XYZ(X, Y, Z)

    assert collection.fields.shape == (3, 7)
    assert_allclose(collection.fields[:, 0], X)
    assert_allclose(collection.fields[:, 1], Y)
    assert_allclose(collection.fields[:, 2], Z)
    assert_allclose(collection.fields[:, 3], np.sqrt(X**2 + Y**2))
    assert_allclose(collection.fields[:, 4], np.arctan2(Y, X) * 180 / np.pi)
    assert_allclose(
        collection.fields[:, 5], np.arctan2(Z, np.sqrt(X**2 + Y**2)) * 180 / np.pi
    )
    assert_allclose(collection.fields[:, 6], np.sqrt(X**2 + Y**2 + Z**2))


def test_geomagnetic_field_collection_get_element_array():
    '''GeomagneticFieldCollection.get_element_array应该能够根据需要的地磁场要素，返回对应的地磁场要素数组'''
    X = np.array([1.0, 2.0, 3.0])
    Y = np.array([2.0, 3.0, 4.0])
    Z = np.array([3.0, 4.0, 5.0])
    collection = GeomagneticFieldCollection.make_from_XYZ(X, Y, Z)
    subset = collection.get_element_array(
        GeomagneticElement.X, GeomagneticElement.Y, GeomagneticElement.Z
    )
    assert is_dataclass(subset)
    assert_allclose(subset.X, X)
    assert_allclose(subset.Y, Y)
    assert_allclose(subset.Z, Z)
