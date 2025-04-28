import torch
from torch import nn
from torch.utils.data import DataLoader

from Transformer.注意力作业2.dataset import MyDataset, embedding
from Transformer.注意力作业2.model import MyModel

EPOCH = 1000
lr = 1e-2

ds = MyDataset()
dl = DataLoader(ds, batch_size=1, shuffle=True)

model = MyModel(embedding.tokenizer.vocab_size)
try:
    model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
    print('加载成功')
except:
    print('加载失败')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    for i, (inps, labels) in enumerate(dl):
        optimizer.zero_grad()
        y = model(inps)
        loss = loss_fn(y.reshape(-1, embedding.tokenizer.vocab_size), labels.reshape(-1))
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}/{EPOCH}, loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
