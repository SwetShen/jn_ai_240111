# 为了让模型识别文字，我们需要对文字进行编码
# 最简单的编码方式为索引编码，按照文本在词库中出现的索引位置，进行编码
# one_hot(独热) 编码是一种索引编码，他将创建一个 0 序列，词的索引位置为 1，其余位置为 0，所以叫做 one_hot
# 词库: ['<sos>', '<eos>', 'how', 'are', 'you', '?', 'i', 'am', 'fine', '.']
# <sos>: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# how:   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

import torch

# 词库
vocab = '<sos> <eos> how are you ? i am fine .'.split()
print(vocab)

# 文本
text = 'how are you ?'
words = text.split()
# 文本对应的索引
idx = torch.tensor([vocab.index(word) for word in words])
print(idx)

encoded = torch.nn.functional.one_hot(idx, num_classes=len(vocab))
print(encoded)
