import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import check_array


@validate_params(
    {
        "vector": ["array-like"],
        "direction": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def project_vector_3d(vector: ArrayLike, direction: ArrayLike) -> np.ndarray:
    vector = check_array(vector, ensure_min_features=3, copy=True)  # shape: (n, 3)
    direction = check_array(
        direction, ensure_min_features=3, copy=True
    )  # shape: (n, 3)
    # (n, 3) * (n, 3) => (n, 3)
    return np.sum(vector * direction, axis=1) / np.linalg.norm(direction, axis=1)


@validate_params(
    {
        "dcm": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def expand_dcm(dcm: ArrayLike) -> ArrayLike:
    """将3x3旋转矩阵拓展为4x4空间变换矩阵"""
    dcm = check_array(dcm, ensure_min_features=3, ensure_min_samples=3, copy=True)
    return np.vstack(
        (
            np.hstack((dcm, np.zeros((3, 1)))),
            np.array([0, 0, 0, 1]),
        )
    )

def vector_direction(vector: ArrayLike) -> np.ndarray:
    """计算向量的方向"""
    vector = check_array(vector, ensure_min_features=3, copy=True)
    return vector / np.linalg.norm(vector, axis=1)