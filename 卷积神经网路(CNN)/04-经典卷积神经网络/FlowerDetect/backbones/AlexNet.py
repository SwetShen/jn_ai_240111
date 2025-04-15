import torch
from torch import nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, device=None):
        super().__init__()
        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # device设置主要在卷积和线性神经网络中（BatchNorm2d）
        self.conv1 = nn.Conv2d(3, 96, 11, 4, device=self.device)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2, device=self.device)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1, device=self.device)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1, device=self.device)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1, device=self.device)

        self.max_pool = nn.MaxPool2d(3, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6 * 6 * 256, 4096, device=self.device)
        self.fc2 = nn.Linear(4096, 4096, device=self.device)
        self.fc3 = nn.Linear(4096, num_classes, device=self.device)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = x.to(self.device)  # 输入特征设备与模型设备保持一致。

        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv5(self.conv4(self.conv3(x)))))
        x = self.flatten(x)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.relu(self.fc2(self.dropout(x)))

        return self.softmax(self.fc3(x))


if __name__ == '__main__':
    model = AlexNet(10, torch.device("cuda"))
    # summary(model, (3, 227, 227), batch_size=10, device="cpu")

    image = torch.rand(1, 3, 227, 227)
    result = model(image)
    print(result.shape)
