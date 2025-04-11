import torch
from torch import nn
import cv2
import numpy as np

# ================= 输入数据集 ===================
img = cv2.imread("./data/numbers/3/0.png")  # (28 x 28 x 3) => (1,3,28,28)
img = np.expand_dims(img, 0)  # (1 x 28 x 28 x 3)
# reshape 可以改变一个矩阵形状(而reshape本身的缺点就是，可能会改变数据在内存中的连续结构)
img = torch.from_numpy(img)  # torch.tensor (1,28,28,3)
img = img.permute([0, 3, 1, 2])  # 交换维度的顺序 (1,3,28,28)

# ================= 卷积 ======================
# 卷积输入的要求 （batch_size,channels,height,width）
conv_layer = nn.Sequential(
    # nn.Conv2d(输入通道数,输出通道数, 卷积核大小,步长,...)
    # 输出通道数：是卷积核的个数
    nn.Conv2d(3, 6, 3, 1)
)

# ================= 开始卷积 ======================
result = conv_layer(img.float())
print(result.shape)  # (1,6,26,26)
