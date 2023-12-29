import os
from dataclasses import astuple, dataclass, field
from typing import List, Optional

import dacite
import numpy as np
import pyvista as pv
import schema as sc
import yaml
from metalpy.utils.bounds import Bounds
from pyvista.plotting import Plotter
from pyvista.plotting.opts import ElementType
from scipy.spatial.transform import Rotation as R

from dvmss.utils import CartesianCoord, rotation_matrix_to_spatial_transformation_matrix


@dataclass
class CartesianOrientation:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_numpy(self):
        return np.array(astuple(self))

    @classmethod
    def from_list(cls, l):
        return cls(*l)


@dataclass
class NormalizedUnitVector(CartesianOrientation):
    def to_numpy(self):
        return super().to_numpy()

    @classmethod
    def from_list(cls, l):
        return super().from_list(l)

    def __post_init__(self):
        norm = np.linalg.norm(self.to_numpy())
        self.x /= norm
        self.y /= norm
        self.z /= norm


@dataclass
class FieldSourceConfig:
    orientation: NormalizedUnitVector  # 磁偶极子方向，归一化单位矢量
    moment: float  # 磁偶极子振幅 (A/m^2)
    location: CartesianCoord  # 磁偶极子笛卡尔坐标系位置
    moment_vector: Optional[CartesianOrientation] = None  # 磁偶极子磁矩矢量

    def __post_init__(self):
        expected_moment_vector = self.moment * self.orientation.to_numpy()
        if self.moment_vector is None:
            self.moment_vector = CartesianOrientation(*expected_moment_vector)
        else:
            if not np.allclose(self.moment_vector.to_numpy(), expected_moment_vector):
                raise ValueError(
                    f"moment_vector must be moment * orientation: {self.moment_vector} != {self.moment} * {self.orientation}"
                )


@dataclass
class PermanentFieldConfig:
    sources: List[FieldSourceConfig] = field(default_factory=list)  # 一系列磁偶极子作为恒定场干扰场源


@dataclass
class InducedFieldConfig:
    voxel_cell_size: float = 0.0  # 载体3D模型体素化后网格尺寸（米）
    susceptibility: float = 0.0  # 网格的磁化率，对所有载体体素网格均取相同值


@dataclass
class InterferenceConfig:
    permanent_field: PermanentFieldConfig = field(default_factory=PermanentFieldConfig)
    induced_field: InducedFieldConfig = field(default_factory=InducedFieldConfig)


@dataclass
class VehicleConfig:
    model_3d_path: str = ""  # 载体3D模型路径
    model_3d: pv.DataSet = None  # 载体3D模型
    actual_wingspan: float = 0.0  # 翼展，米
    actual_length: float = 0.0  # 机身长度，米
    actual_height: float = 0.0  # 机身高度，米
    init_orientation: Optional[NormalizedUnitVector] = None  # 载体3D模型初始朝向

    def choose_orientation(self):
        orientation: NormalizedUnitVector = None

        def callback(mesh):
            nonlocal orientation
            try:
                orientation = NormalizedUnitVector(
                    *(Bounds(mesh.bounds).center - Bounds(self.model_3d.bounds).center)
                )
            except AttributeError as e:
                pass

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
        print(f"{orientation=}, can be recorded in the configuration to avoid choosing again")
        return orientation

    def rotate_to_northward(self):
        if self.init_orientation is None:
            self.init_orientation = self.choose_orientation()
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
        bounds = Bounds(self.model_3d.bounds)
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

    def load_model(self):
        if self.model_3d_path and os.path.isfile(self.model_3d_path):
            self.model_3d = pv.read(self.model_3d_path)
        else:
            raise FileNotFoundError(
                f"3D model file not found at '{self.model_3d_path}'"
            )

    def __post_init__(self):
        self.load_model()
        self.initialize_model()


@dataclass
class MagAgentConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)


class MagAgent:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        self.validate_config(config_data)
        # self.config = MagAgentConfig(**config_data)
        self.config = dacite.from_dict(
            MagAgentConfig,
            config_data,
            dacite.Config(
                type_hooks={
                    CartesianCoord: CartesianCoord.from_list,
                    CartesianOrientation: CartesianOrientation.from_list,
                    NormalizedUnitVector: NormalizedUnitVector.from_list,
                }
            ),
        )

    def validate_config(self, config_data):
        config_schema = sc.Schema(
            {
                "vehicle": {
                    "model_3d_path": sc.And(
                        sc.Use(str), lambda file: os.path.isfile(file)
                    ),
                    "actual_wingspan": sc.And(sc.Use(float), lambda n: n > 0),
                    "actual_length": sc.And(sc.Use(float), lambda n: n > 0),
                    "actual_height": sc.And(sc.Use(float), lambda n: n > 0),
                    sc.Optional("init_orientation"): [float, float, float],
                },
                "interference": {
                    "permanent_field": {
                        "sources": [
                            {
                                "orientation": [float, float, float],
                                "moment": sc.And(sc.Use(float), lambda n: n >= 0),
                                sc.Optional("location"): [float, float, float],
                            }
                        ],
                    },
                    "induced_field": {
                        "voxel_cell_size": sc.And(sc.Use(float), lambda n: n > 0),
                        "susceptibility": float,
                    },
                },
            }
        )
        config_schema.validate(config_data)

    def __repr__(self) -> str:
        return repr(self.config)
