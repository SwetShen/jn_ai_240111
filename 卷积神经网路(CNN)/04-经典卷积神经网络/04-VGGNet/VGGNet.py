import torch
from torch import nn
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU()
            ))
            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.block1 = ConvBlock(3, 64, 2)
        self.block2 = ConvBlock(64, 128, 2)
        self.block3 = ConvBlock(128, 256, 4)
        self.block4 = ConvBlock(256, 512, 4)
        self.block5 = ConvBlock(512, 512, 4)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.max_pool(self.block1(x))
        x = self.max_pool(self.block2(x))
        x = self.max_pool(self.block3(x))
        x = self.max_pool(self.block4(x))
        x = self.max_pool(self.block5(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(self.dropout(x)))
        x = self.relu(self.fc2(self.dropout(x)))
        return self.softmax(self.fc3(x))


if __name__ == '__main__':
    model = VGG19()
    summary(model, (3, 224, 224), device="cpu")
