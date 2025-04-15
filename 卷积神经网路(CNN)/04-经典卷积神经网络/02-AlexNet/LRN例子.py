import torch
from torch import nn

"""
LRN : 局部响应归一化
https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html#torch.nn.LocalResponseNorm
"""

lrn = nn.LocalResponseNorm(2)  # n 邻域大小
signal_2d = torch.randn(32, 5, 24, 24)
signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
output_2d = lrn(signal_2d)
output_4d = lrn(signal_4d)
print(output_2d)
print(output_4d)
