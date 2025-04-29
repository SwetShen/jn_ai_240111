import torch
from torch import nn
from torch.utils.data import DataLoader

from Transformer.对话模型_多数据.LangModel import LangModel
from Transformer.对话模型_多数据.dataset import LangDataset
from embedding_model import tokenizer

# 定义一个设备，用于统一设备进行运算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 100
batch_size = 5
lr = 1e-6

ds = LangDataset()
dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

model = LangModel()
try:
    model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
    print('加载成功')
except:
    print('加载失败')
# 迁移模型到设备上
model.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2)

total_loss = 0.
count = 0

for epoch in range(EPOCH):
    for i, (src, tgt, label) in enumerate(dl):
        optimizer.zero_grad()

        # 分词
        src, src_key_padding_mask = tokenizer(src)
        tgt, tgt_key_padding_mask = tokenizer(tgt)
        label, _ = tokenizer(label)
        src, tgt, src_key_padding_mask, tgt_key_padding_mask = src.to(device), tgt.to(device), src_key_padding_mask.to(
            device), tgt_key_padding_mask.to(device)
        label = label.to(device)

        y = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        loss = loss_fn(y.reshape(y.shape[0] * y.shape[1], -1), label.reshape(-1))
        total_loss += loss.item()
        count += 1
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"epoch:{epoch + 1}/{EPOCH}, loss:{total_loss / count}")

torch.save(model.state_dict(), 'weights/model.pt')
