import numpy as np
import pytest

from dvmss.agent import CartesianOrientation, FieldSourceConfig, NormalizedUnitVector
from dvmss.utils import CartesianCoord


def test_cartesian_orientation_initialization():
    co = CartesianOrientation(1.0, 2.0, 3.0)
    assert co.x == 1.0
    assert co.y == 2.0
    assert co.z == 3.0


def test_cartesian_orientation_to_numpy():
    co = CartesianOrientation(1.0, 2.0, 3.0)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(co.to_numpy(), expected)


def test_field_source_config_initialization_default_moment_vector():
    orientation = NormalizedUnitVector(1, 2, 3)
    moment = 5.0
    location = CartesianCoord(0, 0, 0)
    fsc = FieldSourceConfig(orientation, moment, location)
    expected_moment_vector = moment * orientation.to_numpy()
    np.testing.assert_array_almost_equal(
        fsc.moment_vector.to_numpy(), expected_moment_vector
    )


def test_field_source_config_error_on_incorrect_moment_vector():
    orientation = NormalizedUnitVector(1, 2, 3)
    moment = 5.0
    location = CartesianCoord(0, 0, 0)
    wrong_moment_vector = NormalizedUnitVector(4, 5, 6)
    with pytest.raises(ValueError):
        FieldSourceConfig(orientation, moment, location, wrong_moment_vector)
