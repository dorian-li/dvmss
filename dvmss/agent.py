from dataclasses import dataclass
from pathlib import Path

import pyvista as pv
from metalpy.mepa.process_executor import ProcessExecutor
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from metalpy.scab.modelling.transform import Rotation
from metalpy.utils.bounds import Bounds


class VehicleAgentCreator:
    def __init__(self, model_path, true_length, northward, longest_as_wing=True):
        self.model_path = model_path
        self.true_length = true_length
        self.longest_as_wing = longest_as_wing
        self.model = None
        self.agent = None

    def load_model(self):
        self.model = pv.read(self.model_path)

    def calculate_scaling_factor(self):
        original_bounds = Bounds(self.model.bounds)
        extents = original_bounds.extent
        longest_axis_index = extents.index(
            max(extents[:2])
        )  # Compare x and y extents (0 and 1)
        scale_axis_index = (
            0 if self.longest_as_wing else 1
        )  # Choose the opposite axis if longest_as_wing is False
        if longest_axis_index != scale_axis_index:
            scale_axis_index = longest_axis_index
        scaling_factor = self.true_length / extents[scale_axis_index]
        return scaling_factor

    def rotate_to_north(self):
        northward = Rotation(180, 0, 0, degrees=True, seq="zyx")
        return northward

    def create_agent(self):
        scaling_factor = self.calculate_scaling_factor()
        northward = self.rotate_to_north()
        self.agent = Obj2(
            model=self.model,
            scale=scaling_factor,
            surface_range=[-0.1, 0.1],
            subdivide=True,
            ignore_surface_check=True,
        )
        to_scene_center = -1 * self.agent.center
        self.agent = self.agent.translate(*to_scene_center).apply(northward)

    def build_scene(self):
        scene = Scene()
        scene.append(self.agent, models=1)
        self.agent = scene.build(
            cell_size=self.calculate_scaling_factor(),
            cache=True,
            executor=ProcessExecutor(),
        )
        return self.agent

    def create_vehicle_agent(self):
        self.load_model()
        self.create_agent()
        return self.build_scene()


# 使用示例
# creator = VehicleAgentCreator(
#     "path_to/cessna_172.stl", true_length=11, longest_as_wing=True
# )
# agent = creator.create_vehicle_agent()


from os import PathLike
from typing import Any, List, Optional

import numpy as np
import pyvista as pv
import yaml

from dvmss.utils import CartesianCoord


@dataclass
class MagDipoleParam:
    location: CartesianCoord  # 磁偶极子笛卡尔坐标系位置
    orientation: np.ndarray  # 磁偶极子方向，归一化单位矢量
    moment: float  # 磁偶极子振幅 (A/m^2)
    moment_vector: Optional[np.ndarray] = None  # 磁偶极子磁矩矢量

    def __post_init__(self):
        if self.moment_vector is None:
            self.moment_vector = self.moment * self.orientation


@dataclass
class PermParam:
    sources: List[MagDipoleParam]  # 一系列磁偶极子作为恒定场干扰场源


@dataclass
class InducedParam:
    voxel_cell_size: float  # 载体3D模型体素化后网格尺寸
    susceptibility: float  # 网格的磁化率，对所有载体体素网格均取相同值


@dataclass
class VehicleParam:
    model_3d: Any  # 载体3D模型
    init_orientation: np.ndarray  # 载体初始方向
    real_longest_axis_length: float  # 现实中此载体最长轴长度 (m)


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
            init_orientation=vehicle["init_orientation"],
            real_longest_axis_length=vehicle["real_longest_axis_length"],
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
