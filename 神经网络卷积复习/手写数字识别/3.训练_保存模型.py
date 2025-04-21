import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from 神经网络卷积复习.手写数字识别.设计模型 import LeNet5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 10
lr = 1e-3
batch_size = 1000

train_ds = MNIST(
    root='data',
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=lambda label: torch.tensor(label)
)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = LeNet5()
# 加载模型参数
try:
    model.load_state_dict(torch.load('weights/LeNet5.pth', weights_only=True))
    print('加载模型成功')
except:
    print('加载模型失败')
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    for i, (inputs, labels) in enumerate(train_dl):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y = model(inputs)
        loss = loss_fn(y, labels)
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f'EPOCH: [{epoch + 1}/{EPOCH}]; batch: {i + 1}, loss: {total_loss / count}')

# 保存模型参数
torch.save(model.state_dict(), 'weights/LeNet5.pth')
