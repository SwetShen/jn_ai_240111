import torch
from torch import nn


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


if __name__ == '__main__':
    block = ConvBlock(3, 6, 4)
    print(block)
    # image = torch.rand(1, 3, 10, 10)
    # result = block(image)
    # print(result.shape)
