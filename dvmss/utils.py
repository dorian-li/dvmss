from dataclasses import dataclass
from typing import List, Union

# 笛卡尔坐标系的坐标
@dataclass
class CartesianCoord:
    x: float
    y: float
    z: float
