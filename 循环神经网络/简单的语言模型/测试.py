import time

import torch

from 循环神经网络.简单的语言模型.模型 import LangModel
from 数据集 import vocab

eos_idx = vocab.index('<eos>')

# 加载模型
model = LangModel(len(vocab))
model.load_state_dict(torch.load('weights/model.pth', weights_only=True))
model.eval()


def chat(text: str, max_len=20):
    # 分割字符
    words = text.split()
    # 添加 <sos> 开头
    words = ['<sos>'] + words
    # 转换为索引
    idx = torch.tensor([vocab.index(word) for word in words])
    # one_hot 编码
    inputs = torch.nn.functional.one_hot(idx, num_classes=len(vocab)).float()

    # 输出的索引
    output_idx = []
    texts = []

    # 循环输出
    for i in range(max_len):
        y = model(inputs)
        # 激活概率分布
        logits = y.softmax(-1)
        # 求最大概率索引
        max_idx = logits.argmax(-1)
        # 获取最后一个值的索引
        last_idx = max_idx[-1].item()

        # 转成文本
        word = vocab[last_idx]

        output_idx.append(last_idx)
        texts.append(word)
        # 是否跳出循环
        if last_idx == eos_idx:
            break

        time.sleep(1)
        print(word, end=' ')
        # 将 last_idx 拼接到输入索引中
        idx = torch.tensor(idx.tolist() + [last_idx])
        inputs = torch.nn.functional.one_hot(idx, num_classes=len(vocab)).float()
    return output_idx, texts, ''.join(texts[:-1])


if __name__ == '__main__':
    text = input('User: ')
    print('AI: ', end='')
    chat(text)
    print()
