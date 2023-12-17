from dataclasses import dataclass
from decimal import Decimal
from typing import List, Union

import numpy as np
import pandas as pd
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
