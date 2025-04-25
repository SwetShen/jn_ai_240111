import math

import torch
from sentence_transformers import SentenceTransformer

embedding_dim = 512

embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')
texts = '我爱中共！'
ids = embedding.tokenizer(texts, add_special_tokens=False)['input_ids']
tokens = embedding.tokenizer.convert_ids_to_tokens(ids)
x = torch.from_numpy(embedding.encode(tokens))
# 文本个数
L = x.shape[0]

# QKV 权重
W = torch.randn(3 * embedding_dim, embedding_dim)
Wq, Wk, Wv = W.chunk(3, dim=0)

# 计算 QKV 矩阵
Q = x @ Wq
K = x @ Wk
V = x @ Wv

# 指定头数
nhead = 2
# 每个头的维度
head_dim = embedding_dim // nhead
# 分头
# Q (L, embedding_dim) -> (L, 2 * head_dim) -> (L, nhead=2, head_dim) -> (nhead=2, L, head_dim)
Q = Q.view(L, nhead, head_dim).transpose(0, 1)
K = K.view(L, nhead, head_dim).transpose(0, 1)
V = V.view(L, nhead, head_dim).transpose(0, 1)

# 计算点积缩放注意力
scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(head_dim)

# 计算注意力权重
weight = scores.softmax(-1)

Z = torch.bmm(weight, V)
print(Z)

# 拆分并连接
Z = torch.concat(Z.reshape(nhead * L, head_dim).chunk(nhead, dim=0), dim=1)

Wo = torch.randn(embedding_dim, embedding_dim)

attention = Z @ Wo
print(attention.shape)
