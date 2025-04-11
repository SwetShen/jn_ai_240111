import torch
from torch import nn
import cv2  # BGR
import numpy as np

# import PIL # RGB  pillow

img = cv2.imread("./imgs/test.jpg")  # 524 x 524 x 3
h, w, c = img.shape
# 线性模型要求的输入格式 （1, 524 x 524 x 3）
img = np.expand_dims(img, 0)  # (1, 524,524,3)
img = img.reshape(1, -1)
# 将输入值类型从numpy转化为torch.tensor
img = torch.from_numpy(img)

# 此处的模型只作为演示，并非实际的神经网络设计
model = nn.Sequential(
    nn.Linear(h * w * c, 2),  # 输出分类
    nn.Softmax(dim=-1)
)

result = model(img.float())
print(result)