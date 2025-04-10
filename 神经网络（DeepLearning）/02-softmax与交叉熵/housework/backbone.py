"""
定义主干网络（训练的神经网络）
"""
from torch import nn


class ClassificationNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            # =============== 特征抽象（细化）===========
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            # =============== 特征决定输出类别 ===========
            nn.Dropout(),  # 添加暂退法
            nn.Linear(10, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)
