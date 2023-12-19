import numpy as np
from numpy.testing import assert_allclose

# 创建示例数组
a = np.random.rand(101, 1)  # 形状为 (101, 1)
b = np.random.rand(123, 3)  # 形状为 (123, 3)

# 使用 numpy.einsum 进行高维数组乘法
# 我们希望对 a 的每个标量和 b 的每个矢量进行乘法，得到一个 (123, 101, 3) 的数组
result = np.einsum("i, jk -> jik", a[:, 0], b)

# 将结果数组重塑为 (123, 303) 的形状，使用 Fortran 的列存储格式
final_result = result.reshape(123, 303, order="F")

b_0 = b[1, :].reshape(((1, -1)))  # shape (1, 3)
tmp = a @ b_0
print(f"{tmp.shape=}")
tmp_F = tmp.flatten("F")
print(f"{tmp_F.shape=}")
print(f"{tmp_F=}")
print(f"{final_result[1, :]=}")

print(final_result.shape)

assert_allclose(final_result[0, :], tmp_F)
print(final_result.shape)
