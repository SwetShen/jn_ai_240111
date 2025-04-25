# 总结
# 注意力步骤
# 1. 求 QKV 矩阵
# 2. 求注意力得分
# 3. 激活注意力得分，求得注意力权重
# 4. 注意力权重和 V 加权求和


import torch
# 导入词嵌入模型
from sentence_transformers import SentenceTransformer

# 定义词嵌入维度
embedding_dim = 512

# 构造模型
embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')

texts = '我 爱 中 国 ！'

# 分词
idx = embedding.tokenizer(texts, add_special_tokens=False)['input_ids']
# 转成文本
texts = embedding.tokenizer.convert_ids_to_tokens(idx)
# 编码
# (5, 512)
x = torch.from_numpy(embedding.encode(texts))

# 文本全连接后输出的向量 QKV，他们不一定来自于同一句话的同一个字，若不同，我们需要通过全连接将特征维度统一成一样长
# QKV全连接的权重
W = torch.rand(3, embedding_dim, embedding_dim)
# (512, 512)
Wq, Wk, Wv = torch.chunk(W, 3, dim=0)
Wq, Wk, Wv = Wq.squeeze(), Wk.squeeze(), Wv.squeeze()

# 全连接输出 QKV 矩阵
# (5, 512)
Q = x @ Wq
K = x @ Wk
V = x @ Wv


# 余弦先相似度
def cosine_similarity(Q, K, eps=1e-8):
    return (Q @ K.T) / (torch.sqrt(Q @ Q.T) * torch.sqrt(K @ K.T) + eps)

# cosine_similarity = torch.nn.CosineSimilarity(dim=1)


# 注意力得分
# 因为 scores 是 Q @ K.T 的产物
# 所以 scores 的每一行代表 一个 Q 对 所有 K 的相似度
# scores 形状为 (L, L); L 是序列长度，字数
scores = cosine_similarity(Q, K)

# 计算注意力权重
weights = scores.softmax(dim=-1)

# 计算加权后的 V
# (5, 512)
attention = weights @ V
print(attention.shape)
