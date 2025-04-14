from torch import nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, 4)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)

        self.max_pool = nn.MaxPool2d(3, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.max_pool(self.relu(self.conv1(x)))
        x = self.max_pool(self.relu(self.conv2(x)))
        x = self.max_pool(self.relu(self.conv5(self.conv4(self.conv3(x)))))
        x = self.flatten(x)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.relu(self.fc2(self.dropout(x)))

        return self.softmax(self.fc3(x))


if __name__ == '__main__':
    model = AlexNet(10)
    summary(model, (3, 227, 227), batch_size=10, device="cpu")
