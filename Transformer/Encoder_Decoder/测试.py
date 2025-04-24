import time

import torch

from Encoder_Decoder实现 import EncoderDecoder
from 数据集 import vocab

eos_idx = vocab.index('<eos>')

# 关闭自由奔跑
model = EncoderDecoder(len(vocab), len(vocab), 512, 0)
model.load_state_dict(torch.load('weights/model.pth', weights_only=True))
model.eval()


def chat(text, max_iter=20):
    # 拆分文本
    words = text.split()
    # 转换为索引
    idx = torch.tensor([vocab.index(word) for word in words])
    # one-hot编码
    src = torch.nn.functional.one_hot(idx, num_classes=len(vocab)).float()
    tgt_idx = torch.tensor([vocab.index('<sos>')])
    tgt = torch.nn.functional.one_hot(tgt_idx, num_classes=len(vocab)).float()

    # 输出的索引
    output_idx = []
    texts = []

    for i in range(max_iter):
        # 预测
        y = model(src, tgt)
        # 激活得到概率分布
        logits = y.softmax(-1)
        # 取概率最大的索引
        max_idx = logits.argmax(-1)
        # 取预测结果中最后一个的索引
        last_idx = max_idx[-1].item()
        output_idx.append(last_idx)
        # 保存预测的文本
        last_text = vocab[last_idx]
        texts.append(last_text)

        if last_idx == eos_idx:
            break

        time.sleep(1)
        print(last_text, end=' ')

        # 将预测的最后一个字的索引拼接到 tgt_idx 中
        tgt_idx = torch.tensor(tgt_idx.tolist() + [last_idx])
        # one-hot编码
        tgt = torch.nn.functional.one_hot(tgt_idx, num_classes=len(vocab)).float()

    return output_idx, texts, ' '.join(texts[:-1])


if __name__ == '__main__':
    user_input = input('User: ')
    print('AI: ', end='')
    o, t, s = chat(user_input)
