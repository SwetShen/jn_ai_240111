import math

import torch
from sentence_transformers import SentenceTransformer

embedding_dim = 512

embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')

texts = '我 爱 中 国 ！'

idx = embedding.tokenizer(texts, add_special_tokens=False)['input_ids']
texts = embedding.tokenizer.convert_ids_to_tokens(idx)
x = torch.from_numpy(embedding.encode(texts))
W = torch.rand(3, embedding_dim, embedding_dim)
Wq, Wk, Wv = torch.chunk(W, 3, dim=0)
Wq, Wk, Wv = Wq.squeeze(), Wk.squeeze(), Wv.squeeze()

Q = x @ Wq
K = x @ Wk
V = x @ Wv


def cosine_similarity(Q, K, eps=1e-8):
    return (Q @ K.T) / (torch.sqrt(Q @ Q.T) * torch.sqrt(K @ K.T) + eps)


# scores = cosine_similarity(Q, K)
# 采用点积的方式替换余弦相似度，称为点积相似度
# scores = Q @ K.T
# 在点击相似度的基础上增加了一个缩放系数 sqrt(dk)
# scores = Q @ K.T / sqrt(dk)
# dk: 词嵌入维度，此处就等于 embedding_dim = 512
# 所以这种相似度算法称为: 点积缩放注意力
scores = Q @ K.T / math.sqrt(embedding_dim)

weights = scores.softmax(dim=-1)

attention = weights @ V
print(attention.shape)
