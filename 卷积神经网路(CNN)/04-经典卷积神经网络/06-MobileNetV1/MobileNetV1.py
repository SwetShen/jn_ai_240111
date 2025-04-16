from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchsummary import summary


class DsConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 深度可分离卷积
        self.depth_wise = nn.Sequential(
            # depthwise 深度卷积
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # pointwise 逐点卷积
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.depth_wise(x)


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = Conv2dNormActivation(3, 32,
                                          3, 2, 1,
                                          norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU)

        self.ds_conv1 = DsConv(32, 64)
        self.ds_conv2 = DsConv(64, 128, 2)
        self.ds_conv3 = DsConv(128, 128)
        self.ds_conv4 = DsConv(128, 256, 2)
        self.ds_conv5 = DsConv(256, 256)
        self.ds_conv6 = DsConv(256, 512, 2)

        self.ds_conv7 = nn.Sequential(*[DsConv(512, 512) for _ in range(5)])

        self.ds_conv8 = DsConv(512, 512)
        self.ds_conv9 = DsConv(512, 1024, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(1024, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)
        x = self.ds_conv4(x)
        x = self.ds_conv5(x)
        x = self.ds_conv6(x)
        x = self.ds_conv7(x)
        x = self.ds_conv8(x)
        x = self.ds_conv9(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.softmax(self.fc(x))


if __name__ == '__main__':
    model = MobileNetV1()
    summary(model, (3, 224, 224), batch_size=1, device="cpu")
