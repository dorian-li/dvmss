from pathlib import Path

import pyvista as pv
from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2
from metalpy.scab.modelling.transform import Rotation
from metalpy.utils.bounds import Bounds
from metalpy.mepa.process_executor import ProcessExecutor


def create_vehicle_agent():
    # 读取载体模型
    cessna_172_path = Path(__file__).resolve().parent / "cessna_172.stl"
    cessna_172 = pv.read(cessna_172_path)
    # 计算载体模型缩放至其真实尺寸的缩放倍率
    # cessna_172.plot(show_grid=True)
    original_bounds = Bounds(cessna_172.bounds)  # x: 机翼方向, y: 机身方向, 如agent.plot()中所示
    cessna_172_wing_true = 11  # 单位: 米
    scaling_factor = cessna_172_wing_true / original_bounds.extent[0]
    # 规定载体的初始朝向均为北方，根据agent.plot()所示原始朝向，给出载体模型旋转至其机头朝向北方的旋转角度
    northward = Rotation(180, 0, 0, degrees=True, seq="zyx")
    # 体素化载体模型，缩放至真实尺寸，将载体模型移至场景中心，将载体模型的朝向旋转至北方，形成载体的体素代理
    agent = Obj2(
        model=cessna_172,
        scale=scaling_factor,
        # 下面的设置有助于更好的体素化
        surface_range=[-0.1, 0.1],
        subdivide=True,
        ignore_surface_check=True,
    )
    to_scene_center = -1 * agent.center
    agent = agent.translate(*to_scene_center).apply(northward)
    scene = Scene()
    scene.append(agent, models=1)
    agent = scene.build(
        cell_size=scaling_factor,
        cache=True,
        executor=ProcessExecutor(),
    )
    return agent


def set_vehicle_interf_sources(agent):
    # use mouse click in pv.plotter to set a point in scene and return the coordinates of the point
    def callback(point):
        pass

    p = pv.Plotter()
    p.add_sphere_widget(callback)
    p.add_mesh(agent.to_polydata().threshold(1e-3))
    p.show_grid()
    p.show()

    pass


def set_vehicle_sensor_config():
    # total magnetic sensor
    # vector magnetic sensor
    pass


def set_simulate_scene():
    # background field
    # vehicle motion
    pass


if __name__ == "__main__":
    agent = create_vehicle_agent()
    set_vehicle_interf_sources(agent)
    
