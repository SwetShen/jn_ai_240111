import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self, in_channels, c1x1, rec3x3, c3x3, rec5x5, c5x5, pool):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1x1, 1),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, rec3x3, 1),
            nn.ReLU(),
            nn.Conv2d(rec3x3, c3x3, 3, 1, 1),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, rec5x5, 1),
            nn.ReLU(),
            nn.Conv2d(rec5x5, c5x5, 5, 1, 2),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, pool, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=1000):
        super().__init__()

        # AdaptiveAvgPool2d 自适应平局池化
        # 假设最终的图像大小为 7x7 --> 自适应会让池化大小为 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.softmax(self.fc2(x))
        return x


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.relu = nn.ReLU()

        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.softmax2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        self.softmax0 = InceptionAux(512, 1024)
        self.softmax1 = InceptionAux(528, 1024)

    def forward(self, x):
        x0 = None
        x1 = None
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.conv2(x))

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool(x)
        x = self.inception_4a(x)
        if self.training:  # 如果当前模型开启训练模式
            x0 = self.softmax0(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        if self.training:
            x1 = self.softmax1(x)
        x = self.inception_4e(x)
        x = self.max_pool(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x2 = self.softmax2(x)

        if self.training:
            return x0, x1, x2
        else:
            return x2


if __name__ == '__main__':
    model = GoogleNet()
    summary(model, (3, 224, 224), device="cpu")
    # 12,444,264
