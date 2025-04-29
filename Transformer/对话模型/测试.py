import torch

from Transformer.对话模型.LangModel import LangModel
from embedding_model import embedding

model = LangModel()
model.load_state_dict(torch.load('weights/model.pt', weights_only=True))
model.eval()


def _tokenizer(text):
    result = embedding.tokenizer(text, add_special_tokens=False, max_length=50, padding='max_length',
                                 return_tensors='pt')
    ids = result['input_ids']
    attention_mask = result['attention_mask'].float()
    attention_mask[attention_mask == 0] = float('-inf')
    attention_mask[attention_mask == 1] = 0
    return ids, attention_mask


def chat(src, max_len=50):
    # 分词
    src, src_key_padding_mask = _tokenizer(src)
    # 声明目标序列
    tgt = '[CLS]'

    ids = []
    tokens = []

    for i in range(max_len):
        _tgt, tgt_key_padding_mask = _tokenizer(tgt)
        # 预测
        with torch.no_grad():
            y = model(src, _tgt, src_key_padding_mask, tgt_key_padding_mask)
        # 截取有效部分
        valid_y = y[tgt_key_padding_mask == 0]
        # 激活并找出最大值索引
        _ids = valid_y.softmax(-1).argmax(-1)
        last_id = _ids[-1].item()
        last_token = embedding.tokenizer.convert_ids_to_tokens([last_id])[0]
        ids.append(last_id)
        tokens.append(last_token)
        if last_id == 102:
            break
        tgt = tgt + last_token

    return ids, tokens


if __name__ == '__main__':
    chat('你好，吃了没？')
