import torch
from torch import nn

from Transformer.注意力作业.dataset import MyDataset
from Transformer.注意力作业.model import MyModel

EPOCH = 1000
lr = 1e-1

ds = MyDataset()

model = MyModel()
try:
    model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
    print('模型加载成功')
except:
    print('模型加载失败')

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    for i in range(len(ds)):
        optim.zero_grad()
        inp, label = ds[i]
        y = model(inp)
        loss = loss_fn(y, label.reshape(1))
        total_loss += loss.item()
        count += 1
        loss.backward()
        optim.step()

    if (epoch + 1) % 100 == 0:
        print(f'EPOCH: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
