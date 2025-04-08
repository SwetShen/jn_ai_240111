import torch

a = torch.linspace(0, 1, 16).reshape(4, 2, 2)
print(a.dim())  # 输出有几维的大小
