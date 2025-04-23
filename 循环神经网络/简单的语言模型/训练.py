import torch
from torch import nn

from 循环神经网络.简单的语言模型.数据集 import LangDataset, vocab
from 循环神经网络.简单的语言模型.模型 import LangModel

# 超参数
EPOCH = 1000
lr = 1e-1

# 数据集
ds = LangDataset()
inputs, labels = ds[0]

# 模型
model = LangModel(len(vocab))
# 加载
try:
    model.load_state_dict(torch.load('weights/model.pth', weights_only=True))
except:
    print('加载失败')

# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

total_loss = 0.
count = 0

# 训练循环
for epoch in range(EPOCH):
    optimizer.zero_grad()
    y = model(inputs)
    loss = loss_fn(y, labels)
    total_loss += loss.item()
    count += 1
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}], loss: {total_loss / count}')

# 保存
torch.save(model.state_dict(), 'weights/model.pth')
