from torchsummary import summary
from torch import nn

layer1 = nn.Sequential(
    nn.Conv2d(3, 3, 3, 1),
    nn.Conv2d(3, 3, 3, 1),
    nn.Conv2d(3, 3, 3, 1),
    nn.Conv2d(3, 3, 3, 1),
    nn.Conv2d(3, 3, 3, 1),
    nn.Conv2d(3, 3, 3, 1),
    nn.Conv2d(3, 3, 3, 1)
)

layer2 = nn.Sequential(
    nn.Conv2d(3, 3, 3, 2, 1)
)

summary(layer1, (3, 28, 28), device="cpu")
print("=" * 30)
summary(layer2, (3, 28, 28), device="cpu")
