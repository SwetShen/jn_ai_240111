import torch

from Transformer.注意力作业.model import MyModel
from Transformer.注意力作业.dataset import token_map

model = MyModel()
model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
model.eval()

while 1:
    name = input('请输入: ')
    with torch.no_grad():
        y = model(name)
    # 激活得到概率分布
    logits = y.softmax(-1)
    idx = logits.argmax(-1)
    print(token_map[idx.item()])
