from dataclasses import dataclass
from decimal import Decimal
from typing import List, Union

import numpy as np
import pandas as pd
from metalpy.scab.utils.format import check_components
from numpy.typing import ArrayLike


# 笛卡尔坐标系的坐标
@dataclass
class CartesianCoord:
    x: float
    y: float
    z: float


def flatten_tuple(tup):
    """将嵌套的tuple展开为一维的tuple""" ""
    result = tuple()
    for item in tup:
        if isinstance(item, tuple):
            result += flatten_tuple(item)
        else:
            result += (item,)
    return result


def count_decimal_places(x: str):
    """计算浮点数文本的小数位数"""
    return Decimal(x).as_tuple().exponent * -1


def rotation_matrix_to_spatial_transformation_matrix(
    rotation_matrix: ArrayLike,
) -> ArrayLike:
    """将旋转矩阵转换为空间变换矩阵"""
    if not isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = np.array(rotation_matrix)
    if rotation_matrix.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix shape must be (3, 3): {rotation_matrix.shape}"
        )
    return np.vstack(
        (
            np.hstack((rotation_matrix, np.zeros((3, 1)))),
            np.array([0, 0, 0, 1]),
        )
    )


def format_pandas(data, components, rx_loc):
    components = check_components(components)
    """将模拟结果转换为pandas.DataFrame"""
    rx_num = rx_loc.shape[0]
    flight_len = data.shape[1]
    ret = pd.DataFrame(
        data.reshape((rx_num * 3, flight_len), order="F").T,
        columns=[f"{c}_{i}" for c in components for i in range(rx_num)],
    )
    ret["time"] = np.arange(flight_len)
    return ret
