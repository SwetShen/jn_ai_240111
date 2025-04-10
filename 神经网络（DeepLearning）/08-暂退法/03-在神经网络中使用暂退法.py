import torch
from torch import nn
from torchsummary import summary  # pip install torchsummary

"""
两种定义模型的方法
"""


class Net1(nn.Module):
    def __init__(self, num_classes=1000):
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


class Net2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # =============== 特征抽象（细化）===========
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.tanh = nn.Tanh()
        # =============== 特征决定输出类别 ===========
        self.dropout = nn.Dropout()  # 添加暂退法
        self.fc = nn.Linear(10, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer2(x))
        x = self.dropout(x)
        return self.softmax(self.fc(x))


if __name__ == '__main__':
    model1 = Net1(3)
    model2 = Net2(3)

    summary(model1, (2,), batch_size=10, device="cpu")
    summary(model2, (2,), batch_size=10, device="cpu")
