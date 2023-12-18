from dataclasses import astuple
from datetime import datetime
from typing import List

import numpy as np
from numpy.testing import assert_allclose

from dvmss.flight import Attitude, Flight, VehicleState


def test_flight_setup():
    """Flight.setup应该能够正确设置Flight实例的日期和状态数组"""
    date = datetime(2022, 1, 1)
    states = [
        VehicleState(1, 2, 3, 10, Attitude(1, 1, 1)),
        VehicleState(4, 5, 6, 20, Attitude(2, 2, 2)),
        VehicleState(7, 8, 9, 30, Attitude(3, 3, 3)),
    ]

    flight = Flight.setup(date, states)

    assert flight.date == date
    desired_states = np.array(
        [[1, 2, 3, 10, 1, 1, 1], [4, 5, 6, 20, 2, 2, 2], [7, 8, 9, 30, 3, 3, 3]]
    )
    assert_allclose(flight.states, desired_states)
