from torch import nn


class AlexNet(nn.Module):
    # num_classes: 分类数
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.c1 = nn.Conv2d(3, 96, 11, stride=4)
        self.c2 = nn.Conv2d(96, 256, 5, padding=2)
        self.c3 = nn.Conv2d(256, 384, 3, padding=1)
        self.c4 = nn.Conv2d(384, 384, 3, padding=1)
        self.c5 = nn.Conv2d(384, 256, 3, padding=1)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.max_pool = nn.MaxPool2d(3, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        # Nx3x227x227
        x = self.c1(x)
        # Nx96x55x55
        x = self.relu(x)
        x = self.max_pool(x)
        # Nx96x27x27
        x = self.c2(x)
        # Nx256x27x27
        x = self.relu(x)
        x = self.max_pool(x)
        # Nx256x13x13
        x = self.c3(x)
        # Nx384x13x13
        x = self.relu(x)
        x = self.c4(x)
        # Nx384x13x13
        x = self.relu(x)
        x = self.c5(x)
        # Nx256x13x13
        x = self.relu(x)
        x = self.max_pool(x)
        # Nx256x6x6
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # Nx4096
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # Nx4096
        x = self.relu(x)
        x = self.fc3(x)
        # Nx1000
        y = self.log_softmax(x)
        return y


if __name__ == '__main__':
    import torch
    model = AlexNet()
    x = torch.rand(5, 3, 227, 227)
    y = model(x)
    print(y.shape)
