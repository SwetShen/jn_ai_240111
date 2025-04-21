from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.c3 = nn.Conv2d(16, 120, 4)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # Nx1x32x32
        x = self.c1(x)
        x = self.relu(x)
        # Nx6x28x28
        x = self.max_pool(x)
        # Nx6x14x14
        x = self.c2(x)
        # Nx16x10x10
        x = self.relu(x)
        x = self.max_pool(x)
        # Nx16x5x5
        x = self.c3(x)
        # Nx120x1x1
        x = self.flatten(x)
        # Nx120
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    import torch

    x = torch.rand(5, 1, 28, 28)
    model = LeNet5()
    y = model(x)
    print(y.shape)
