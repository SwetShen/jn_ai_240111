"""
BCELoss 二值交叉熵（极大似然值公式）
"""
import torch
from torch import nn

p = torch.tensor([0., 1., 0., 1.])  # 真实概率
q = torch.tensor([0.2, 0.9, 0.1, 0.8])  # 预测概率

loss1 = -torch.mean(p * torch.log(q) + (1 - p) * torch.log(1 - q))
print(loss1)

criterion = nn.BCELoss()  # 二值交叉熵损失
loss2 = criterion(q, p)
print(loss2)
