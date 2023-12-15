# 各世界坐标系到笛卡尔坐标系xyz的关系：
# 1. pyvista笛卡尔坐标系：飞机机头朝向y轴正方向，右翼x轴正方向，z轴向上为正方向
# 2. 姿态坐标系：roll旋转轴->y，pitch旋转轴->x，yaw旋转轴-> -z
# (x, y, z) => (roll, pitch, yaw), NED
# 3. 地磁场坐标系 (x,y,z)=>(northing, east, downward), NED
# 4. simpeg: (x,y,z)=>(easting, northing, upward), ENU

from pathlib import Path
import matplotlib.pyplot as plt
from sgl2020 import Sgl2020

from dvmss.data import (
    DataIoC, # 依赖容器
    Date, # 航线日期
    Dipoles, # 恒定场源参数
    InducedArgs, # 感应场源参数
    InertialAttitude, # 飞行姿态
    LocationWGS84, # 地理位置
    SensorPos, # 探头位置
    Vehicle, # 测量平台
    MagAgentMesh, # 体素模型网格
    Tmi, # 磁总场探头测量值
    PermanentFieldTmi, # 磁总场恒定场分量
    InducedFieldTmi, # 磁总场感应场分量
    MagVectorXYZ, # 磁三分量探头测量值
    PermanentFieldXYZ, # 磁三分量恒定场分量
    InducedFieldXYZ, # 磁三分量感应场分量
)

# 以加拿大数据一航线的飞行姿态和轨迹仿真
surv = (
    Sgl2020()
    .line(["1002.02"])
    .source(
        [
            "ins_yaw",
            "ins_pitch",
            "ins_roll",
            "lon",
            "lat",
            "utm_z",
        ]
    )
    .take()
)
flt_d = surv["1002.02"]

# 提供所需的各种信息，内部自动根据依赖关系构建所需项
# 缺少依赖会有异常处理
data = DataIoC().with_data(
    InertialAttitude(
        yaw=flt_d["ins_yaw"], pitch=flt_d["ins_pitch"], roll=flt_d["ins_roll"]
    ),
    LocationWGS84(lon=flt_d["lon"], lat=flt_d["lat"], alt=flt_d["utm_z"]),
    Date(year=2015, doy=177),
    Dipoles(
        {
            "position": [0, 1.9, 0],
            "moment": 154.8,
            "orientation": [0.9848077, 0.0, -0.17364818],
        },
        {
            "position": [0, 2.7, 0],
            "moment": 179,
            "orientation": [-9.84807753e-01, 1.20604166e-16, 1.73648178e-01],
        },
    ),
    SensorPos[0](1, 1, 1),
    Vehicle(
        model_3d_path=Path(__file__).parent / "cessna_172.stl",
        actual_wingspan=11.0,
        actual_length=8.3,
        actual_height=2.7,
        init_heading=[0.0, -142.45333862304688, 0.0],
    ),
    InducedArgs(susceptibility=1.0, cell_size=0.1),
)

plt.plot(data[Tmi], label="tmi")
plt.plot(data[MagVectorXYZ], label="mag_vector_xyz")
plt.legend()
plt.grid()
plt.show()
