import torch

from Transformer.注意力作业2.model import MyModel
from dataset import embedding

model = MyModel(embedding.tokenizer.vocab_size)
model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
model.eval()


def chat(texts, max_iter=20):
    # 预测出的索引
    ids = []
    # 预测出的token
    tokens = []

    for i in range(max_iter):
        # 编码
        x = embedding.encode(texts, convert_to_tensor=True).unsqueeze(0)
        # 预测
        with torch.no_grad():
            y = model(x)
        # 激活
        logits = y.softmax(-1)
        # 求索引
        _ids = logits.argmax(-1)
        # 解码
        _tokens = embedding.tokenizer.convert_ids_to_tokens(_ids[0].tolist())
        last_id = _ids[0, -1].item()
        last_token = _tokens[-1]
        ids.append(last_id)
        tokens.append(last_token)

        print(tokens)

        # 判断是否跳出循环
        if last_id == 102:
            break
        # 追加预测的最后一个字到输入中
        texts.append(last_token)

    return ids, tokens


if __name__ == '__main__':
    chat(['[CLS]'])
