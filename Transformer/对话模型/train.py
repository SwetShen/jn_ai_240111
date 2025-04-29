import torch
from torch import nn
from torch.utils.data import DataLoader

from Transformer.对话模型.LangModel import LangModel
from Transformer.对话模型.dataset import LangDataset

EPOCH = 100
lr = 1e-3

ds = LangDataset()
dl = DataLoader(ds, batch_size=1, shuffle=True)

model = LangModel()
try:
    model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
    print('加载成功')
except:
    print('加载失败')

# ignore_index: 忽略损失的索引
# 因为 [PAD] 的索引是 0，我们不希望 [PAD] 参与损失计算，所以添加 ignore_index=0
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    for i, (src, tgt, src_key_padding_mask, tgt_key_padding_mask, labels) in enumerate(dl):
        optimizer.zero_grad()
        y = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        loss = loss_fn(y.reshape(y.shape[0] * y.shape[1], -1), labels.reshape(-1))
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'epoch: [{epoch + 1}/{EPOCH}]; loss: {total_loss / count}')

torch.save(model.state_dict(), 'weights/model.pt')
