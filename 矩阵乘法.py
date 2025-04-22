# 矩阵乘法
# 矩阵乘法的条件: 第一个矩阵的列数等于第二个矩阵的行数

import torch

t1 = torch.randint(0, 4, (2, 3))
t2 = torch.randint(0, 4, (3, 4))

print(torch.mm(t1, t2).shape)
print(torch.matmul(t1, t2).shape)
print((t1 @ t2).shape)

t1 = torch.randint(0, 4, (5, 2, 3))
t2 = torch.randint(0, 4, (5, 3, 4))
# 批量矩阵相乘
print(torch.bmm(t1, t2).shape)
