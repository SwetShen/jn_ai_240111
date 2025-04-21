# 测试模型的目的用于检测模型的准确率
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from 神经网络卷积复习.手写数字识别.设计模型 import LeNet5

# 加载数据集
ds = MNIST(
    root='./data',
    train=False,
    download=False,
    transform=ToTensor(),
    target_transform=lambda label: torch.tensor(label)
)

dl = DataLoader(ds, batch_size=1000, shuffle=False)

# 加载模型
model = LeNet5()
# map_location: 用于映射设备参数到指定设备上
# 由于训练时采用了 cuda，但测试时没有 GPU，所以将参数映射到 cpu 上
model.load_state_dict(torch.load('weights/LeNet5.pth', weights_only=True, map_location='cpu'))
# 评估模型
model.eval()

# 回答正确的数量
correct_count = 0

# 前向传播，统计预测正确的数量
for i, (inputs, labels) in enumerate(dl):
    # 关闭梯度追踪
    with torch.no_grad():
        y = model(inputs)
    # y (1000, 10)
    # 用 softmax 激活，找出概率最大的索引
    # y = torch.nn.Softmax(dim=-1)(y)
    # y = torch.nn.functional.softmax(y, dim=-1)
    y = torch.softmax(y, dim=-1)
    idx = y.argmax(dim=-1)
    correct_count += (idx == labels).sum().item()

print(f'准确率: {correct_count / len(ds) * 100:.2f}%')
