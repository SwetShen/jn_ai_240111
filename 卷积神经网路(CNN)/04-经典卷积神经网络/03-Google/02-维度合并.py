import torch

# 模拟三个分支的卷积结果
branch1 = torch.randn(5, 64, 64, 64)
branch2 = torch.randn(5, 128, 64, 64)
branch3 = torch.randn(5, 16, 64, 64)
# 在通道维度上进行合并
y = torch.cat([branch1, branch2, branch3], dim=1)
print(y.shape)  # -> torch.Size([5, 208, 64, 64]) 通道维度被合并
# 208 = 64 + 128 + 16
