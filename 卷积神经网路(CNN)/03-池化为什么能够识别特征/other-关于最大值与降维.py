import torch

a = torch.tensor([[1, 2, 3, 4]])  # (1,4)

# torch.max 可以在二维或者一维空间中求解一个最大值(输出结构为一个值结构)
# b = torch.max(a)
# print(b.dim())

# torch.amax 可以指定在某个维度求最大值，且会保留其余的维度
b = torch.amax(a, dim=1)
print(b)
