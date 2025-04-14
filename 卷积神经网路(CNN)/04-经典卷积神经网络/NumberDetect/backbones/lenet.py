import torch
from torch import nn
from torchsummary import summary


class LetNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.avg_pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avg_pool2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.tanh = nn.Tanh()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.avg_pool1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.avg_pool2(x))
        x = self.tanh(self.conv3(x))

        # x = x.reshape(-1, 120)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


if __name__ == '__main__':
    model = LetNet(2)
    # 1、通过torchsummary 测算整个模型
    # summary(model, (1, 32, 32), batch_size=10, device="cpu")

    # 2、用伪造数据测试模型是否能达到预期效果
    # image = torch.rand(10, 1, 32, 32)
    # result = model(image)
    # print(result.shape)

    # print(model)
