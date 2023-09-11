from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, reduce

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


@dataclass
class FlightMotion:
    date: datetime
    lon: pd.Series
    lat: pd.Series
    altitude: pd.Series
    pitch: pd.Series
    roll: pd.Series
    yaw: pd.Series

    @property
    def attitude_rotation(self):
        return R.from_euler(
            "xyz",
            angles=np.column_stack((self.pitch, self.roll, self.yaw)),
            degrees=True,
        )

    def iteratts(self):
        for i in range(len(self.lon)):
            yield self.pitch.iloc[i], self.roll.iloc[i], self.yaw.iloc[i]

    def __len__(self):
        return len(self.lon)
