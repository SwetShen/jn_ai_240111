import random

import torch
from torch import nn


# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.LSTMCell(input_size, hidden_size)

    # src: 输入序列 (L, input_size)
    def forward(self, src):
        L, input_size = src.shape
        h = torch.zeros(self.hidden_size)
        c = torch.zeros(self.hidden_size)
        # 循环编码
        for i in range(L):
            h, c = self.cell(src[i], (h, c))
        memory = (h, c)
        return memory


20000
512


# 解码器
class Decoder(nn.Module):
    # vocab_size: 词典大小
    # free_running_rate: 自由奔跑概率
    def __init__(self, vocab_size, input_size, hidden_size, free_running_rate=0.1):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        # 将 vocab_size 的数据转换为 input_size
        self.fc_input_size = nn.Linear(vocab_size, input_size)
        self.free_running_rate = free_running_rate

    # memory: 编码器输出的记忆，因为使用了 LSTM 模型，所以是个元组 (c, h)
    # tgt: 目标序列 (L, input_size)
    def forward(self, memory, tgt):
        L, input_size = tgt.shape
        y = []
        out = None
        # 是否开启自由奔跑模式
        free_running = random.random() < self.free_running_rate
        for i in range(L):
            # 若自由奔跑且 out 存在输出则上一轮的 out 作为本轮的输入
            # 否则目标序列作为本轮的输入
            inp = out if free_running and out is not None else tgt[i]
            memory = self.cell(inp, memory)
            # 输出一个字
            out = self.fc_out(memory[0])
            y.append(out)
            # 转换形状
            # 此处因为我们处于学习阶段，并没有使用词嵌入模型
            # 所以我们姑且使用全连接将输出进行转换，构造下一轮的输入
            out = self.fc_input_size(out)
        y = torch.stack(y)
        return y


# 编解码器模型
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, free_running_rate=0.1):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(vocab_size, input_size, hidden_size, free_running_rate)

    # src: 输入序列
    # tgt: 目标序列
    def forward(self, src, tgt):
        memory = self.encoder(src)
        y = self.decoder(memory, tgt)
        return y


if __name__ == '__main__':
    model = EncoderDecoder(20, 512, 64, free_running_rate=1)
    src = torch.rand(5, 512)
    tgt = torch.rand(8, 512)
    y = model(src, tgt)
    print(y.shape)
