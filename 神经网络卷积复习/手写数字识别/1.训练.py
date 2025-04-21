# 总结训练步骤
# 1. 超参数
# 2. 数据集
# 3. 模型
# 4. 损失函数和优化器
# 5. 训练
#   5.1 清空梯度
#   5.2 前向传播
#   5.3 计算损失
#   5.4 反向传播
#   5.5 更新参数
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from 神经网络卷积复习.手写数字识别.设计模型 import LeNet5

# 1. 超参数
EPOCH = 10  # 总训练轮数
lr = 1e-3  # 学习率
batch_size = 1000  # 批次数，用于小批量训练

# 2. 数据集
train_ds = MNIST(
    # 数据集下载并存放的路径
    root='data',
    # 是否下载训练集
    train=True,
    # 是否下载
    download=False,
    # 输入数据转换器
    transform=ToTensor(),
    # 标签转换器
    target_transform=lambda label: torch.tensor(label)
)
# 创建一个数据加载器，用于批量加载数据
# shuffle: 是否乱序
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 3. 模型
model = LeNet5()

# 4. 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 记录泛化损失(平均损失)
total_loss = 0.
count = 0

# 5. 训练
for epoch in range(EPOCH):
    # i: 批次数
    # inputs: 输入数据
    # labels: 标签
    for i, (inputs, labels) in enumerate(train_dl):
        # 5.1 清空梯度
        optimizer.zero_grad()
        # 5.2 前向传播
        y = model(inputs)
        # 5.3 计算损失
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        # 5.4 反向传播
        loss.backward()
        # 5.5 更新参数
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'EPOCH: [{epoch + 1}/{EPOCH}]; batch: {i + 1}, loss: {total_loss / count}')
