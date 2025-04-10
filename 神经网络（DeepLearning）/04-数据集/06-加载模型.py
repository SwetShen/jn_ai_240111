import torch
from torch import nn


class CustomNet(nn.Module):
    def __init__(self, num_classes=1000):  # num_classes 分类的数量
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)


# 加载模型结构和参数
# model = torch.load("./save/best.pt",weights_only=False)

# 加载模型的参数
model = CustomNet(3)
state = torch.load("./save/best.pt", weights_only=True)  # 只加载模型参数
model.load_state_dict(state)
