import torch
from torch import nn


class PositionEncoding(nn.Module):
    # embed_dim: 词嵌入的维度
    # max_len: 词序列的最大长度
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        # 存放所有正余弦值的张量
        self.P = torch.zeros(1, max_len, embed_dim)
        # 正余弦波的输入
        pos = torch.arange(max_len).reshape(-1, 1).float()
        # 10000 的指数
        top = torch.arange(0, embed_dim, 2) / embed_dim
        # 公式分母部分
        fm = torch.pow(10000, top).unsqueeze(0)
        X = pos / fm
        # 求正余弦
        sin = torch.sin(X)
        cos = torch.cos(X)
        # 保存正余弦波
        self.P[:, :, 0::2] = sin
        self.P[:, :, 1::2] = cos

    # x (N, L, embed_dim)
    # P (1, max_len, embed_dim)
    def forward(self, x):
        # 叠加位置编码
        x = x + self.P[:, :x.shape[1]]
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pe = PositionEncoding(embed_dim=512)
    x = torch.rand(5, 10, 512)
    x = pe(x)
    print(x.shape)

    fig, ax = plt.subplots()
    ax.plot(torch.arange(1000), pe.P[0, :, 201])
    plt.show()


