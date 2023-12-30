import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from sgl2020 import SGL2020

from dvmss.agent import MagAgent
from dvmss.detector import Detector, DetectorCollection, MagSensor, MagSensorType
from dvmss.flight import Flight, VehicleState
from dvmss.geomag import IGRF
from dvmss.simulation import Simulation
from dvmss.utils import CartesianCoord, compute_noise_level


def make_mag_agent():
    cessna_172_1_config = Path(__file__).resolve().parent / "cessna_172_1.yaml"
    return MagAgent(cessna_172_1_config)


def make_flights():
    surv = SGL2020()
    flt_d = (
        surv.line(1002.02)
        .source(
            [
                "tt",
                "lon",
                "lat",
                "utm_x",
                "utm_y",
                "utm_z",
                "ins_pitch",
                "ins_roll",
                "ins_yaw",
                "mag_4_uc",
            ]
        )
        .take(include_line=True)
    )
    # plt.plot(flt_d["ins_pitch"], label="pitch")
    # plt.plot(flt_d["ins_roll"], label="roll")
    # plt.plot(flt_d["ins_yaw"], label="yaw")
    # plt.legend()
    # plt.show()

    date = datetime(2020, 6, 20)
    return Flight.setup_from_series(
        date=date,
        timestamp=flt_d["tt"],
        longitude=flt_d["lon"],
        latitude=flt_d["lat"],
        elevation=flt_d["utm_z"],
        roll=flt_d["ins_roll"],
        pitch=flt_d["ins_pitch"],
        yaw=flt_d["ins_yaw"],
    )


def make_detectors():
    return DetectorCollection.of(
        Detector(
            location=CartesianCoord(-2.8, 1.1, 0.32),
            sensor_type=MagSensorType.SCALAR,
            # loc_interactive=True,
        ),
        Detector(
            location=CartesianCoord(2.8, 1.1, 0.32),
            sensor_type=MagSensorType.SCALAR,
            # loc_interactive=True,
        ),
        Detector(
            location=CartesianCoord(-5.5, 1.1, 0.32),
            sensor_type=MagSensorType.SCALAR,
            # loc_interactive=True,
        ),
        Detector(
            location=CartesianCoord(5.5, 1.1, 0.32),
            sensor_type=MagSensorType.SCALAR,
            # loc_interactive=True,
        ),
        Detector(
            location=CartesianCoord(0, -8, -0.7),
            sensor_type=MagSensorType.SCALAR,
            # loc_interactive=True,
        ),
    )


if __name__ == "__main__":
    # 各世界坐标系到笛卡尔坐标系xyz的关系：
    # 1. pyvista笛卡尔坐标系：飞机机头朝向y轴正方向，右翼x轴正方向，z轴向上为正方向
    # 2. 姿态坐标系：roll旋转轴->y，pitch旋转轴->x，yaw旋转轴-> -z
    # (x, y, z) => (roll, pitch, yaw), NED
    # 3. 地磁场坐标系 (x,y,z)=>(northing, east, downward), NED
    # 4. simpeg: (x,y,z)=>(easting, northing, upward), ENU
    mag_agent = make_mag_agent()

    flights = make_flights()

    detectors = make_detectors()

    simluation = Simulation(mag_agent, IGRF, flights)
    simluation.preview("test.wmv")
    exit()

    sampled_detectors = simluation.sample(detectors, plot=False)

    # subplot all MagSensor
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(sampled_detectors[0].sensor_data.iloc[:, i])
        ax.set_title(sampled_detectors[0].sensor_data.columns[i])
        ax.set_xticks([])
    plt.show()
