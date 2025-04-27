# 总结，注意事项:
# 1. 冻结 embedding 层的参数，因为这是别人训练好的模型，我们不用训练它
# 2. 使用归一化和残差，缓解梯度消失的问题


import math

import torch
from torch import nn
from sentence_transformers import SentenceTransformer


class MyModel(nn.Module):
    # nhead: 分头的个数
    def __init__(self, nhead=2):
        super().__init__()
        self.nhead = nhead
        self.embed_dim = 512
        # 每个头的维度
        self.head_dim = self.embed_dim // self.nhead
        self.embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')
        # 冻结 embedding 参数，从而让他不要被训练
        for p in self.embedding.parameters():
            p.requires_grad = False
        tokens = '张三 是 法外狂徒 ， 罗翔 是 律师 ， 老王 是 牛头人高手 。'.split()
        # 词向量
        self.word_vectors = torch.from_numpy(self.embedding.encode(tokens))
        # 全连接层
        self.fc_attention = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_v = nn.Linear(self.embed_dim, self.embed_dim)
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        # 归一化
        self.norm = nn.LayerNorm(256)

    # input_str: 输入字符串 `张三` `罗翔` 或 `老王`
    def forward(self, input_str):
        # 编码->词向量
        input_vector = torch.from_numpy(self.embedding.encode(input_str)).unsqueeze(0)
        # 计算QKV矩阵
        Q = self.fc_q(input_vector)
        K = self.fc_k(self.word_vectors)
        V = self.fc_v(self.word_vectors)
        # 分头
        Q = Q.view(Q.shape[0], self.nhead, self.head_dim).transpose(0, 1)
        K = K.view(K.shape[0], self.nhead, self.head_dim).transpose(0, 1)
        V = V.view(V.shape[0], self.nhead, self.head_dim).transpose(0, 1)
        # 归一化
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        # 计算点积缩放注意力
        scores = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weight = scores.softmax(-1)
        Z = torch.bmm(weight, V)
        Z = torch.concat(Z.reshape(-1, self.head_dim).chunk(2, dim=0), dim=1)
        attention = self.fc_attention(Z)
        # (1, 512)
        # 残差
        attention = attention + input_vector
        # 全连接进行分类
        return self.classifier(attention)


if __name__ == '__main__':
    model = MyModel()
    y = model('张三')
    print(y.shape)
