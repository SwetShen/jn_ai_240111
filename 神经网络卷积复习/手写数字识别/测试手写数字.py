import torch

from 神经网络卷积复习.手写数字识别.设计模型 import LeNet5
from 数据预处理_封装 import preprocess

# 加载模型
model = LeNet5()
model.load_state_dict(torch.load('weights/LeNet5.pth', weights_only=True, map_location='cpu'))
model.eval()

# 预处理图片
inputs = preprocess('data/numbers1.jpg')

# 预测
with torch.no_grad():
    y = model(inputs)

# 激活得到概率分布
y = y.softmax(-1)
print(y)
# 求 topk
# topk 就是最大的 k 个概率值和索引
values, indices = y.topk(k=3)
print(values)
print(indices)
