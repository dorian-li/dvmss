import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from pathlib import Path

import pyvista as pv
from sgl2020 import SGL2020

from dvmss.agent import InducedParam, MagAgent, PermParam, VehicleParam
from dvmss.detector import Detector, DetectorCollection, MagSensorType
from dvmss.flight import Flight
from dvmss.geomag import IGRF
from dvmss.simulation import Simulation
from dvmss.utils import CartesianCoord

if __name__ == "__main__":
    # 各世界坐标系到笛卡尔坐标系xyz的关系：
    # 1. pyvista笛卡尔坐标系：飞机机头朝向y轴正方向，右翼x轴正方向，z轴向上为正方向
    # 2. 姿态坐标系：roll旋转轴->y，pitch旋转轴->x，yaw旋转轴-> -z
    # 3. 地磁场坐标系：东北地NED，北->y，东->x，地->-z
    cessna_172_1_config = Path(__file__).resolve().parent / "cessna_172_1.yaml"
    mag_agent = MagAgent.setup_from_config(cessna_172_1_config)
    print(mag_agent)

    surv = SGL2020()
    flt_d = (
        surv.line(1002.02)
        .source(
            [
                "tt",
                "lon",
                "lat",
                "utm_z",
                "ins_pitch",
                "ins_roll",
                "ins_yaw",
            ]
        )
        .take(include_line=True)
    )
    print(flt_d)
    date = datetime(2020, 6, 20)
    flight = Flight.setup_from_series(
        date=date,
        timestamp=flt_d["tt"],
        longitude=flt_d["lon"],
        latitude=flt_d["lat"],
        elevation=flt_d["utm_z"],
        roll=flt_d["ins_roll"],
        pitch=flt_d["ins_pitch"],
        yaw=flt_d["ins_yaw"],
    )

    print(flight)

    detectors = DetectorCollection.of(
        Detector(
            location=CartesianCoord(0, 0, 0),
            sensor_type=MagSensorType.SCALAR,
            # loc_interactive=True,
        ),
        Detector(
            location=CartesianCoord(0, 0, 0),
            sensor_type=MagSensorType.VECTOR,
            # loc_interactive=True,
        ),
    )

    simluation = Simulation(mag_agent, IGRF, flight)
    simluation.compute_perm_interf_vector(detectors)
    simluation.compute_induced_interf_vector(detectors)
    # sampled_detectors = simluation.sample(detectors)
    # print(sampled_detectors)
