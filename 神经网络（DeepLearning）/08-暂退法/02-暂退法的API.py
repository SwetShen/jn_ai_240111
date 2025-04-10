import torch
from torch import nn

dropout = nn.Dropout(0.5)  # 可以定义削减神经元的概率值

features = torch.normal(0, 1, (6, 2))

result = dropout(features)
print(result)
