from dataclasses import astuple, dataclass
from decimal import Decimal
from typing import List, Union

import numpy as np
import pandas as pd
import pyvista as pv
from metalpy.scab.utils.format import check_components
from numpy.typing import ArrayLike
from pyvista.plotting import Plotter
from scipy.signal import butter, detrend, filtfilt


# 笛卡尔坐标系的坐标
@dataclass
class CartesianCoord:
    x: float
    y: float
    z: float

    @classmethod
    def from_list(cls, l):
        return cls(*l)


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


def project_vectors_to_orientations(
    vectors: ArrayLike, orientations: ArrayLike
) -> ArrayLike:
    """将一组三维向量投影到另一组方向上"""
    vectors = np.array(vectors)
    orientations = np.array(orientations)
    if vectors.shape[0] != orientations.shape[0]:
        raise ValueError(
            f"vectors and orientations must have same length: {vectors.shape[0]} != {orientations.shape[0]}"
        )
    if vectors.shape[1] != 3:
        raise ValueError(f"vectors shape must be (n, 3): {vectors.shape}")
    return np.einsum("ij, ij -> i", vectors, orientations)  # shape: (n, )


def NED_to_ENU(x: ArrayLike):
    """将北东地坐标系转换为东北天坐标系"""
    x = np.array(x)
    if x.shape[1] != 3:
        raise ValueError(f"vectors shape must be (n, 3): {x.shape}")
    return np.column_stack((x[:, 1], x[:, 0], -x[:, 2]))


def compute_noise_level(tmi: ArrayLike, sampling_rate=10):
    """计算噪声水平"""
    tmi = np.array(tmi)
    if tmi.ndim != 1:
        raise ValueError(f"tmi shape must be (n, ): {tmi.shape}")
    b, a = butter(
        4,
        [0.1, 0.6],
        btype="bandpass",
        fs=sampling_rate,
        output="ba",
    )
    filtered_tmi = filtfilt(b, a, tmi, axis=0)
    return np.std(filtered_tmi)
