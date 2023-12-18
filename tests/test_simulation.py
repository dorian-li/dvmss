from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from dvmss.simulation import Simulation


@pytest.fixture
def mock_mag_agent():
    return MagicMock()


@pytest.fixture
def mock_geomag():
    return MagicMock()


@pytest.fixture
def mock_flight():
    return MagicMock()


@pytest.fixture
def simulation(mock_mag_agent, mock_geomag, mock_flight):
    return Simulation(mock_mag_agent, mock_geomag, mock_flight)


def test_simulation_initialization(
    simulation, mock_mag_agent, mock_geomag, mock_flight
):
    assert simulation.mag_agent == mock_mag_agent
    assert simulation.geomag == mock_geomag
    assert simulation.flight == mock_flight
    assert simulation._cached_background_field is None


def test_simulation_background_field_cached(simulation, mock_geomag, mock_flight):
    mock_geomag.query.return_value = np.array([1, 2, 3, 4])
    # assert simulation.background_field == np.array([1, 2, 3, 4])
    assert_allclose(simulation.background_field, np.array([1, 2, 3, 4]))
    assert mock_geomag.query.call_count == 1

    # Call again, should return the cached value without calling query again
    # assert simulation.background_field == np.array([1, 2, 3, 4])
    assert_allclose(simulation.background_field, np.array([1, 2, 3, 4]))
    assert mock_geomag.query.call_count == 2


def test_simulation_perm_interf(simulation):
    detectors = MagicMock()
    result = simulation.perm_interf(detectors)
    assert result == detectors


def test_simulation_sample(simulation):
    detectors = MagicMock()
    result = simulation.sample(detectors)
    assert result == detectors
