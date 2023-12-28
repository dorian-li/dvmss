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

if __name__ == "__main__":
    # 各世界坐标系到笛卡尔坐标系xyz的关系：
    # 1. pyvista笛卡尔坐标系：飞机机头朝向y轴正方向，右翼x轴正方向，z轴向上为正方向
    # 2. 姿态坐标系：roll旋转轴->y，pitch旋转轴->x，yaw旋转轴-> -z
    # (x, y, z) => (roll, pitch, yaw), NED
    # 3. 地磁场坐标系 (x,y,z)=>(northing, east, downward), NED
    # 4. simpeg: (x,y,z)=>(easting, northing, upward), ENU
    cessna_172_1_config = Path(__file__).resolve().parent / "cessna_172_1.yaml"
    mag_agent = MagAgent(cessna_172_1_config)
    # print(mag_agent)

    # pl = Plotter()
    # pl.add_mesh(mag_agent.config.vehicle.model_3d)
    # tail_sphere = pv.Sphere(center=(0, -7, -0.35), radius=0.3)
    # tail = pv.Cylinder(
    #     center=[0, -5, -0.35], direction=[0, -1, 0], radius=0.2, height=4
    # )
    # pl.add_mesh(tail_sphere, color="red")
    # pl.add_mesh(tail, color="orange")
    # pl.add_axes()
    # pl.show_grid()
    # pl.show()

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
                "mag_1_c",
                "mag_1_uc",
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
    from pyvista.plotting import Plotter
    from scipy.spatial.transform import Rotation as R

    # p = Plotter(window_size=[1600, 912])
    # p.open_movie("test.wmv", framerate=60, quality=10)
    # p.show_axes()
    # mesh = mag_agent.config.vehicle.model_3d.copy()
    # p.add_mesh(mesh)
    # # p.add_mesh(mesh.outline_corners())
    # # p.camera.azimuth = -180
    # p.show_grid()
    # p.add_arrows(np.array([0, 0, 3]), np.array([0, 1, 0]), color="lightcoral")
    # att_NED = flight.query(
    #     VehicleState.ROLL, VehicleState.PITCH, VehicleState.YAW
    # ).to_numpy()
    # # NED to ENU
    # att_ENU = np.column_stack((att_NED[:, 1], att_NED[:, 0], -att_NED[:, 2] + 90))
    # r = R.from_euler(
    #     "xyz",
    #     angles=att_ENU,
    #     degrees=True,
    # )  # yaw-pitch-roll顺序旋转
    # # loop for att_ENU each row
    # from time import sleep
    # from dvmss.utils import rotation_matrix_to_spatial_transformation_matrix
    # print(p.camera.model_transform_matrix)
    # # r0 = R.from_matrix(p.camera.model_transform_matrix[:3, :3])
    # # r = R.concatenate(r0, r)
    # att_matrixs = r.as_matrix()
    # att_matrixs_inv = r.inv().as_matrix()
    # # p.write_frame()
    # for i in range(att_matrixs.shape[0])[:10000]:
    #     m_spatial = rotation_matrix_to_spatial_transformation_matrix(
    #         att_matrixs[i, :, :]
    #     )
    #     m_spatial_inv = rotation_matrix_to_spatial_transformation_matrix(
    #         att_matrixs_inv[i, :, :]
    #     )
    #     mesh.transform(m_spatial)
    #     p.write_frame()
    #     mesh.transform(m_spatial_inv)
    #     sleep(0.01)
    # p.close()
    # exit()
    # print(flight)

    detectors = DetectorCollection.of(
        Detector(location=CartesianCoord(0, -7, 0), sensor_type=MagSensorType.SCALAR)
    )

    detectors = DetectorCollection.of(
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

    simluation = Simulation(mag_agent, IGRF, flight)
    sampled_detectors = simluation.sample(detectors, plot=True)

    # subplot all MagSensor
    fig, axes = plt.subplots(4, 4)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(sampled_detectors[0].sensor_data.iloc[:, i])
        ax.set_title(sampled_detectors[0].sensor_data.columns[i])
        ax.set_xticks([])
    plt.show()
