import torch
from torch import nn
import numpy as np
import cv2

# =============== 加载保存的模型权重 ==================
static_dict = torch.load("./save/best.pt", weights_only=True)
model = nn.Sequential(
    # 输入层
    nn.Conv2d(3, 16, 3, 1, 1),  # (16,28,28)
    nn.ReLU(),
    # 隐藏层
    nn.Conv2d(16, 32, 3, 2, 1),  # (32,14,14)
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, 1, 1),  # (32,14,14)
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 2, 1),  # (64,7,7)
    nn.ReLU(),
    # 输出层
    nn.Flatten(),  # (batch_size,64x7x7)
    nn.Dropout(),
    nn.Linear(64 * 7 * 7, 1024),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(1024, 10),  # 10是数字的分类数量
    nn.LogSoftmax(dim=-1)
)
model.load_state_dict(static_dict)
# =============== 加载测试数据 ==================
# 测试模型时，一定要使用除数据集以外的图像内容进行测试
image = cv2.imread("./data/test/9.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.expand_dims(image, 0)
image = torch.from_numpy(image).permute([0, 3, 1, 2])

model.eval()  # 注意：模型重如果有dropout，一定要开启评估模式，关闭dropout
result = model(image.float())  # (1,10)
indices = torch.argmax(result, dim=-1)
# 注意：此时因为是0-9的数字识别，indices与文件下标是一致的，因此不需要字典映射
#      在其他的图像分类任务中，一定要获取一个字典映射
print(indices)
