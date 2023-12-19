from dataclasses import astuple, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pyvista as pv
import yaml
from metalpy.mepa.process_executor import ProcessExecutor
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from metalpy.scab.modelling.transform import Rotation
from metalpy.utils.bounds import Bounds
from pyvista.plotting import Plotter
from pyvista.plotting.opts import ElementType
from scipy.spatial.transform import Rotation as R

from dvmss.utils import CartesianCoord, rotation_matrix_to_spatial_transformation_matrix


@dataclass
class MagDipoleParam:
    location: CartesianCoord  # 磁偶极子笛卡尔坐标系位置
    orientation: np.ndarray  # 磁偶极子方向，归一化单位矢量
    moment: float  # 磁偶极子振幅 (A/m^2)
    moment_vector: Optional[np.ndarray] = None  # 磁偶极子磁矩矢量

    def __post_init__(self):
        if self.moment_vector is None:
            self.moment_vector = self.moment * self.orientation
        else:
            if not np.allclose(self.moment_vector, self.moment * self.orientation):
                raise ValueError(
                    f"moment_vector must be moment * orientation: {self.moment_vector} != {self.moment} * {self.orientation}"
                )


@dataclass
class PermParam:
    sources: List[MagDipoleParam]  # 一系列磁偶极子作为恒定场干扰场源


@dataclass
class InducedParam:
    voxel_cell_size: float  # 载体3D模型体素化后网格尺寸（米）
    susceptibility: float  # 网格的磁化率，对所有载体体素网格均取相同值


@dataclass
class Orientation:
    x: float
    y: float
    z: float

    def to_numpy(self):
        return np.array(astuple(self))


@dataclass
class VehicleParam:
    model_3d: pv.DataSet  # 载体3D模型
    actual_wingspan: float  # 机翼展长
    actual_length: float  # 机身长度
    actual_height: float  # 机身高度
    init_orientation: Optional[Orientation] = None  # 载体3D模型初始朝向

    def pick_orientation(self):
        orientation: Orientation = None

        def callback(mesh):
            nonlocal orientation
            try:
                orientation = Orientation(
                    *(Bounds(mesh.bounds).center - Bounds(self.model_3d.bounds).center)
                )
            except AttributeError as e:
                pass
            print(orientation)

        pl = Plotter()
        pl.add_mesh(self.model_3d)
        pl.add_mesh(
            pv.Cube(bounds=self.model_3d.bounds),
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
        if orientation is None:
            raise ValueError(
                "The initial orientation is not specified. Please interactively select the face of the vehicle heading"
            )
        return orientation

    def rotate_to_northward(self):
        if self.init_orientation is None:
            self.init_orientation = self.pick_orientation()
        to_northward_r = R.align_vectors(
            self.init_orientation.to_numpy().reshape((1, 3)),
            np.array([(0, 1, 0)]),  # y轴正方向为正北
        )[0]
        to_northward = rotation_matrix_to_spatial_transformation_matrix(
            to_northward_r.as_matrix()
        )
        self.model_3d.transform(to_northward, inplace=True)

    def move_to_center(self):
        to_center = -1 * np.array(self.model_3d.center)
        self.model_3d.translate(to_center, inplace=True)

    def scale_to_actual_size(self):
        bounds = Bounds(self.model_3d.bounds)  # 模型已经变化，重新计算包围盒
        actual_size = np.array(
            [self.actual_wingspan, self.actual_length, self.actual_height]
        )
        scale_factor: float = actual_size.max() / bounds.extent.max()
        self.model_3d.scale(scale_factor, inplace=True)

    def initialize_model(self):
        self.rotate_to_northward()
        self.move_to_center()
        self.scale_to_actual_size()
        # self.model_3d.plot(show_grid=True)

    def __post_init__(self):
        self.initialize_model()


@dataclass
class MagAgent:
    config: "MagAgent.Config"

    @dataclass
    class Config:
        vehicle: VehicleParam
        interf: "MagAgent.Config.Interf"

        @dataclass
        class Interf:
            perm: PermParam
            induced: InducedParam

    @classmethod
    def setup(
        cls,
        vehicle_param: VehicleParam,
        perm_param: PermParam,
        induced_param: InducedParam,
    ):
        interf = cls.Config.Interf(perm=perm_param, induced=induced_param)
        config = cls.Config(vehicle=vehicle_param, interf=interf)
        return cls(config=config)

    @classmethod
    def setup_from_config(cls, config_path: PathLike):
        config_folder = Path(config_path).parent
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        vehicle = config["vehicle"]
        vehicle_param = VehicleParam(
            model_3d=pv.read(config_folder / Path(vehicle["model_3d"])),
            actual_wingspan=vehicle["actual_wingspan"],
            actual_length=vehicle["actual_length"],
            actual_height=vehicle["actual_height"],
            init_orientation=Orientation(*vehicle["init_orientation"])
            if vehicle.get("init_orientation") is not None
            else None,
        )

        perm_sources = []
        for source in config["interference"]["permanent_field"]["sources"]:
            location = source.get("location")
            if location is not None:
                location = CartesianCoord(*location)
            else:
                pass
            perm_sources.append(
                MagDipoleParam(
                    location=location,
                    orientation=np.array(source["orientation"]),
                    moment=source["moment"],
                )
            )
        perm_param = PermParam(sources=perm_sources)

        induced_config = config["interference"]["induced_field"]
        induced_param = InducedParam(
            voxel_cell_size=induced_config["voxel_cell_size"],
            susceptibility=induced_config["susceptibility"],
        )

        return cls.setup(vehicle_param, perm_param, induced_param)
