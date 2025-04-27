# 注意力掩码
# 什么是注意力掩码？
# 掩码 mask: 用于计算 QK 点积时，叠加一个值（源码中是在计算QK点积前 torch.nn.functional 5439行）
# 形象的理解就是在输入序列上蒙住了部分的序列成员，让其不参与运算
# 有什么用？
# 用于抑制部分 QK 点积的结果，让其不参与注意力运算，通俗的理解就是，屏蔽部分序列中的成员在注意力计算时的效果
# 被抑制的成员，其注意力权重接近 0，意为完全不重要
import math

import torch
from sentence_transformers import SentenceTransformer

texts = '我 爱 祖国 ! [PAD] [PAD]'.split()

embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')
embedding_dim = embedding.get_sentence_embedding_dimension()

# (6, 512)
x = embedding.encode(texts, convert_to_tensor=True)

W = torch.randn(3 * embedding_dim, embedding_dim)
Wq, Wk, Wv = W.chunk(3, dim=0)

Q = x @ Wq
K = x @ Wk
V = x @ Wv

# 注意力分数
# (6, 6)
scores = Q @ K.T / math.sqrt(embedding_dim)

# 声明注意力掩码
# 0: 不掩盖对应位置的值
# float('-inf'): 掩盖对应位置的值
# attn_mask = torch.tensor([
#     [0., 0., float('-inf'), 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0.],
#     [0., float('-inf'), 0., 0., 0., 0.],
#     [0., 0., 0., 0., 0., 0.],
#     [0., 0., 0., float('-inf'), 0., 0.],
#     [0., 0., 0., 0., 0., 0.],
# ])

# 因果注意力掩码
# 常见的一种注意力掩码是三角形的注意力掩码，创建方法如下
# tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)), diagonal=1) == 1
# tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 1, float('-inf'))
# torch.triu: 通过对角线，将对角线下方的值归零，对角线是从左上到右下
# diagonal: 对角线的移动单位
one_tensor = torch.ones_like(scores)
attn_mask = torch.triu(one_tensor, diagonal=1)
attn_mask[attn_mask == 1] = float('-inf')

# 最终形状如下:
# tensor([[0., -inf, -inf],
#         [0., 0.  , -inf],
#         [0., 0.  , 0.  ]])
# 该形状在后续的解码器自注意力中很有用
# 以语言模型为例:
# 解码器中的每个词只应该参考到自己这个词为止的词，不应该参考未来出现的词
# 所以未来位置的词被掩码遮盖
# 上面张量中，行代表自注意力查询是的每个词，列代表输出序列中每个词对于该行这个词的重要程度
# 例如输出序列是: 我 爱 你
#     我    爱    你
# 我 [0., -inf, -inf] “我” 在计算注意力池时，不应该计算 “爱你” 注意力
# 爱 [0., 0.  , -inf] “爱” 在计算注意力池时，不应该计算 “你” 注意力
# 你 [0., 0.  , 0.  ] “你” 在计算注意力池时，可以参考所有已出现的词

# 在 nn.MultiheadAttention 的输入参数中，可以添加对应的掩码

# padding_mask: 填充掩码
padding_mask = torch.tensor([0., 0., 0., 0., float('-inf'), float('-inf')])

# padding_mask 叠加到 attn_mask 上
# (1, 6) -> (6, 6)
padding_mask = padding_mask.unsqueeze(0).expand_as(scores)
attn_mask = attn_mask + padding_mask

# 叠加掩码
scores = scores + attn_mask

# 注意力权重
weights = torch.nn.functional.softmax(scores, dim=-1)
print(weights)
