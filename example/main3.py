import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import pyvista as pv
from sgl2020 import SGL2020

from dvmss.agent import InducedParam, MagAgent, PermParam, VehicleParam
from dvmss.flight import Flight
from dvmss.geomag import IGRF

if __name__ == "__main__":
    # 通过plot载体3D模型，确认导入模型在当前坐标系下的初始朝向、最长轴对应机身还是机翼
    # cessna_172 = Path(__file__).resolve().parent / "cessna_172.stl"
    # pv.read(cessna_172).plot(show_grid=True)
    # 以这一cessna 172模型为例，最长轴对应机翼方向，初始机头朝向为y轴负方向
    # 姿态一般yaw以从北顺时针为正，pitch以向上为正，roll以向右舷为正
    # 若pitch旋转轴为x轴，roll旋转轴为y轴，yaw旋转轴为z轴
    # 则姿态的地理坐标系y轴正方向为正北，x轴正方向为正东，z轴垂直于地平面向地为正，呈右手系
    # 因此载体模型的初始朝向为正南向，由于后续计算均以载体朝向正北为初始，因此填入此信息后内部将自动旋转载体模型

    cessna_172_1_config = Path(__file__).resolve().parent / "cessna_172_1.yaml"
    mag_agent = MagAgent.setup_from_config(cessna_172_1_config)
    print(mag_agent)

    surv = SGL2020()
    flt_d = (
        surv.line(1002.02)
        .source(["tt", "lon", "lat", "utm_z", "ins_pitch", "ins_roll", "ins_yaw"])
        .take(include_line=True)
    )
    print(flt_d)
    flight = Flight.setup_from_series(
        timestamp=flt_d["tt"],
        longitude=flt_d["lon"],
        latitude=flt_d["lat"],
        elevation=flt_d["utm_z"],
        attitude=flt_d[["ins_pitch", "ins_roll", "ins_yaw"]],
    )
    print(flight)
