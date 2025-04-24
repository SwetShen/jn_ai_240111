import torch
from torch import nn

from Encoder_Decoder实现 import EncoderDecoder
from 数据集 import LangDataset, vocab

EPOCH = 1000
lr = 1e-1

ds = LangDataset()
(src, tgt), label = ds[0]

model = EncoderDecoder(len(vocab), len(vocab), 512, 0.2)
try:
    model.load_state_dict(torch.load('weights/model.pth', weights_only=True))
    print('模型加载成功')
except:
    print('模型加载失败')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    optimizer.zero_grad()
    y = model(src, tgt)
    loss = loss_fn(y, label)
    total_loss += loss.item()
    count += 1
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'EPOCH: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pth')
