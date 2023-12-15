from __future__ import annotations

from datetime import datetime, timedelta
from os import PathLike

import numpy as np
import ppigrf
import pyvista as pv
from deinterf.utils.data_ioc import DataIoC, DataNDArray, UniqueData
from metalpy.mepa.process_executor import ProcessExecutor
from metalpy.scab.builder.simulation_builder import SimulationBuilder
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.modelled_mesh import ModelledMesh
from metalpy.scab.modelling.shapes import Obj2
from metalpy.scab.potential_fields.magnetics.simulation import Simulation3DDipoles
from metalpy.utils.bounds import Bounds
from numpy.typing import ArrayLike
from pyvista.plotting.plotting import Plotter
from pyvista.plotting.opts import ElementType
from scipy.spatial.transform import Rotation as R
from SimPEG.potential_fields.magnetics.simulation import Simulation3DIntegral
from typing_extensions import Dict, Literal, NamedTuple

from dvmss.utils.transform import expand_dcm, project_vector_3d


class InertialAttitude(DataNDArray):
    # 机体姿态角
    def __new__(cls, yaw: ArrayLike, pitch: ArrayLike, roll: ArrayLike):
        # unit: deg
        return super().__new__(cls, yaw, pitch, roll)


class LocationWGS84(DataNDArray, UniqueData):
    # WGS84坐标系下的飞行轨迹
    def __new__(cls, lon: ArrayLike, lat: ArrayLike, alt: ArrayLike):
        # alt unit: m
        return super().__new__(cls, lon, lat, alt)


class Date(NamedTuple):
    # 飞行日期
    year: int
    doy: int  # day of year


class Tmi(DataNDArray):
    # 磁总场探头测量值
    @classmethod
    def __build__(cls, container: DataIoC) -> Tmi:
        mag_vec_xyz = container[MagVectorXYZ]
        background_field_xyz = container[BackgroundFieldXYZ]
        tmi = project_vector_3d(mag_vec_xyz, background_field_xyz)
        return cls(tmi)


class MagVectorXYZ(DataNDArray):
    # 磁三分量探头测量值
    @classmethod
    def __build__(cls, container: DataIoC) -> MagVectorXYZ:
        bg_xyz = container[BackgroundFieldXYZ]
        perm_xyz = container[PermanentFieldXYZ]
        induced_xyz = container[InducedFieldXYZ]
        return cls(bg_xyz + perm_xyz + induced_xyz)


class BackgroundFieldXYZ(DataNDArray):
    # 磁三分量探头背景磁场成分
    @classmethod
    def __build__(cls, container: DataIoC) -> BackgroundFieldXYZ:
        geo_enu = container[IGRF]
        att_angle = container[InertialAttitude]  # (yaw, pitch, roll): DEN
        # DEN to ENU
        att_angle = att_angle[:, [1, 2, 0]]
        att_angle[:, 2] = -att_angle[:, 2]

        r = R.from_euler("xyz", att_angle, degrees=True)
        geo_xyz = r.apply(geo_enu, inverse=True)
        return cls(*geo_xyz.T)


class BackgroundFieldTmi(DataNDArray):
    # 磁总场探头背景磁场成分
    @classmethod
    def __build__(cls, container: DataIoC) -> BackgroundFieldTmi:
        geo_vectors = container[BackgroundFieldXYZ]
        geo_t = np.linalg.norm(geo_vectors, axis=1)
        return cls(geo_t)


class PermanentFieldXYZ(DataNDArray):
    # 磁三分量探头飞机恒定磁场成分
    @classmethod
    def __build__(cls, container: DataIoC) -> PermanentFieldXYZ:
        builder = SimulationBuilder.of(Simulation3DDipoles)
        builder.sources(container[Dipoles].pos)
        detector_locs = container[SensorPos]

        builder.receivers(detector_locs, ["bx", "by", "bz"])
        model = container[Dipoles].moment_vector.flatten("F")  # (source_num * 3,)
        perm_vectors = builder.build().dpred(model)
        perm_vectors *= 1e9  # 单位由T转换为nT
        perm_vectors_expanded = np.repeat(
            perm_vectors[:, np.newaxis], container[LocationWGS84].shape[0], axis=1
        ).T  # (flight_len, 3)
        return cls(*perm_vectors_expanded.T)


class PermanentFieldTmi(DataNDArray):
    # 磁总场探头飞机恒定磁场成分
    @classmethod
    def __build__(cls, container: DataIoC):
        xyz = container[PermanentFieldXYZ]
        bg_xyz = container[BackgroundFieldXYZ]
        return cls(project_vector_3d(xyz, bg_xyz))


class InducedFieldCoef(DataNDArray):
    # 飞机感应磁场补偿9系数
    @classmethod
    def __build__(cls, container: DataIoC) -> InducedFieldCoef:
        components = ["bx", "by", "bz"]
        builder = SimulationBuilder.of(Simulation3DIntegral)
        builder.source_field(strength=1, inc=1, dec=1)
        dectector_loc = container[SensorPos]
        builder.receivers(dectector_loc, components)
        builder.vector_model()
        builder.active_mesh(container[MagAgentMesh])
        builder.store_sensitivities(True)

        kernel = builder.build().G  # (3, active_mesh_num * 3)
        model_mag_direct = container[TmiProjection]  # (flight_len, 3)
        model_scalar = container[MagAgentMesh].get_active_model()  # (active_mesh_num, )
        if np.allclose(model_scalar, model_scalar[0]):
            n_active_mesh = len(model_scalar)
            kernel = np.sum(kernel.reshape(-1, 3, n_active_mesh), axis=2)  # (3, 3)
            coef = kernel * model_scalar[0]  # (3, 3)
        else:
            model_vector = np.einsum(
                "i, jk -> jik", model_scalar, model_mag_direct
            )  # (flight_len, active_mesh_num, 3)
            model_vector = model_vector.reshape(
                (model_mag_direct.shape[0], model_scalar.shape[0] * 3),
                order="F",
            ).T  # (active_mesh_num * 3, flight_len)

            z = kernel @ model_vector  # (3, flight_len)
            y = model_mag_direct.T  # (3, flight_len)
            coef = z @ np.linalg.pinv(y)  # (3, 3)
        return coef


class InducedFieldXYZ(DataNDArray):
    # 磁三分量探头飞机感应磁场成分
    @classmethod
    def __build__(cls, container: DataIoC) -> InducedFieldXYZ:
        coef = container[InducedFieldCoef]
        induced_vectors = coef @ container[BackgroundFieldXYZ].T  # (3, flight_len)
        induced_vectors = induced_vectors.T  # (flight_len, 3)
        return induced_vectors


class InducedFieldTmi(DataNDArray):
    # 磁总场探头飞机感应磁场成分
    @classmethod
    def __build__(cls, container: DataIoC):
        xyz = container[InducedFieldXYZ]
        bg_xyz = container[BackgroundFieldXYZ]
        return cls(project_vector_3d(xyz, bg_xyz))


class IGRF(DataNDArray):
    # 以IGRF模型作为背景场
    @classmethod
    def __build__(cls, container: DataIoC):
        lon, lat, alt = container[LocationWGS84].T

        # doy to datetime
        year, doy = container[Date]
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)

        geo_e, geo_n, geo_u = ppigrf.igrf(lon, lat, alt / 1000, date)
        geo = np.row_stack((geo_e, geo_n, geo_u)).T

        return cls(*geo.T)


class Dipoles(DataNDArray, UniqueData):
    # 磁偶极子作为飞机恒定场源
    def __new__(cls, *args: Dict[str, Any]):
        # 磁偶极子所在笛卡尔坐标位置
        pos = np.array([d["position"] for d in args])
        # 磁偶极子振幅 (A/m^2)
        moment = np.array([d["moment"] for d in args])
        # 磁偶极子方向，归一化单位矢量
        orientation = np.array([d["orientation"] for d in args])
        return super().__new__(cls, pos, moment, orientation)

    @property
    def pos(self):
        return self[:, :3]

    @property
    def moment_vector(self):
        # 磁矩矢量
        return self[:, 3][:, None] * self[:, 4:]


class SensorPos(DataNDArray):
    # 磁探头位置，每个位置隐式同时存在一个磁总场和磁三分量探头
    def __new__(cls, x: float, y: float, z: float):
        return super().__new__(cls, [x, y, z])


class TmiProjection(DataNDArray):
    # 背景磁场方向，作为各三分量磁场的投影方向
    @classmethod
    def __build__(cls, container: DataIoC) -> TmiProjection:
        bg_xyz = container[BackgroundFieldXYZ]  # (n, 3)
        tmi_projection = bg_xyz / np.linalg.norm(bg_xyz, axis=1)[:, None]
        return cls(tmi_projection)


class MagAgentMesh(ModelledMesh, UniqueData):
    # 飞机感应场源体素网格
    @classmethod
    def __build__(cls, container: DataIoC) -> MagAgentMesh:
        agent_obj = Obj2(
            model=container[Vehicle].model_3d,
            surface_range=[-0.1, 0.1],
            subdivide=True,
            ignore_surface_check=True,
        )
        # model原点信息丢失，重新平移至场景中心
        to_center = -agent_obj.center
        agent_obj.translate(*to_center, inplace=True)
        scene = Scene.of(
            agent_obj,
            models=container[InducedArgs].susceptibility,
        )
        return scene.build(
            cell_size=container[InducedArgs].cell_size,
            cache=True,
            executor=ProcessExecutor(4),
        )


class Vehicle(NamedTuple):
    # 飞行平台
    model_3d_path: PathLike
    # 实际飞机尺寸，用于将3d模型缩放为真实尺寸
    actual_wingspan: float  # 实际机翼长度，米
    actual_length: float  # 实际机身长度，米
    actual_height: float  # 实际飞机高度，米
    init_heading: ArrayLike | None

    @property
    def model_3d(self):
        # 模型经缩放、平移、旋转后，形成初始状态：真实飞机尺寸、机头朝北、处于场景中心
        origin = pv.read(self.model_3d_path)
        # rotate to northward
        heading = self.init_heading
        if self.init_heading is None:
            heading = self.choose_heading(origin)
        to_northward_r, _ = R.align_vectors(
            np.array([[0, 1, 0], [0, 1, 0]]),  # y轴正方向为正北
            np.array([heading, heading]),
        )  # Sicpy #17462
        to_northward = expand_dcm(to_northward_r.as_matrix())
        origin.transform(to_northward, inplace=True)
        # move to center
        to_center = -1 * np.array(origin.center)
        origin.translate(to_center, inplace=True)
        # scale to actual size
        bounds = Bounds(origin.bounds)
        actual_size = np.array(
            [self.actual_wingspan, self.actual_length, self.actual_height]
        )
        scale_factor: float = actual_size.max() / bounds.extent.max()
        origin.scale(scale_factor, inplace=True)
        return origin

    def choose_heading(self, model):
        # 交互式选择机头所在包围盒面
        heading = None

        def callback(mesh):
            nonlocal heading
            try:
                heading = Bounds(mesh.bounds).center - Bounds(model.bounds).center
            except AttributeError:
                pass

        pl = Plotter()
        pl.add_mesh(model)
        pl.add_mesh(
            pv.Cube(bounds=model.bounds),
            opacity=0.3,
            show_edges=True,
            pickable=True,
        )
        pl.add_axes()
        pl.show_grid()
        pl.enable_element_picking(
            callback=callback,
            mode=ElementType.FACE,
        )
        pl.add_text(
            "when confirm vehicle heading, just close the window", position="lower_left"
        )
        pl.show(auto_close=False)
        if heading is None:
            raise ValueError(
                "The initial heading is not specified. Please interactively select the face of the vehicle heading"
            )
        print(
            f"{heading=}, can be recorded in the configuration to avoid choosing again"
        )
        return heading


class InducedArgs(NamedTuple):
    # 飞机感应场源体素参数
    susceptibility: float  # 网格的磁化率，SI
    cell_size: float  # 网格尺寸，米


class PermanentFieldCoef(DataNDArray):
    # 飞机恒定磁场补偿3系数
    @classmethod
    def __build__(cls, container: DataIoC):
        return container[PermanentFieldXYZ][0, :]
