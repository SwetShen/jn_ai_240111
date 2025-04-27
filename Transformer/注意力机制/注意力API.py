# torch.nn.functional.scaled_dot_product_attention 此 API 用于计算点乘缩放注意力
# nn.MultiheadAttention 此 API 用于计算多头注意力
# nn.Transformer.generate_square_subsequent_mask 生成掩码
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

texts = '张三 是 法外狂徒 [PAD] [PAD]'.split()
embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')
x = embedding.encode(texts, convert_to_tensor=True)
# (1, 5, 512)
x = x.unsqueeze(0)

# 官方api文档 https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#multiheadattention

# 参数:
# embed_dim: 嵌入维度，如果是语言模型的注意力计算，则该参数指的是词嵌入的长度
# num_heads: 多头注意力的头数
# dropout: 避免过拟合的随机抑制概率
# bias: 线性映射时是否添加偏置
# add_bias_kv: 是否给线性映射后的kv再添加一层偏置，该偏置不会加到最终结果，而是相当于多了一个中间特征，再计算过程中会增加一个嵌入长度的数据，最后矩阵乘法时被融合掉
#               例如: 我传入四个词嵌入，形状为 (4, 100)，add_bias_kv 为 True 时，kv 会被计算成 (5, 100)，多出来的 (1, 100) 就是添加的偏置信息，最后通过矩阵相乘，不影响最终结果形状
# add_zero_attn: 和 add_bias_kv 类似，会添加一层全是 0 的数据到 kv 中
# kdim、vdim: kv 维度的长度，默认和 embed_dim 相同。基本上 qkv 长度都应该相同，在 encoder-decoder 模型中，嵌入模型通常共享权重，输出长度相同
# batch_first: 批次维度是否放到首位
mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=4,
    dropout=0.2,
    bias=False,
    add_bias_kv=True,
    add_zero_attn=True,
    kdim=512,
    vdim=512,
    batch_first=True
)

padding_mask = torch.tensor([
    [0., 0., 0., float('-inf'), float('-inf')]
])

# attn_mask = torch.tensor([
#     [[0., float('-inf'), 0., 0., 0.],
#      [0., 0., 0., 0., 0.],
#      [0., 0., float('-inf'), 0., 0.],
#      [0., 0., 0., 0., 0.],
#      [float('-inf'), 0., 0., 0., 0.]]
# ])

# 创建因果注意力掩码
# 参数为序列长度
attn_mask = nn.Transformer.generate_square_subsequent_mask(5).unsqueeze(0)

attn_mask = attn_mask.expand(4, -1, -1)

# 输入参数:
# query、key、value: QKV映射前的矩阵
# key_padding_mask: key 中对于填充占位符的掩码
# need_weights: 是否返回注意力权重
# attn_mask: QK 点积后的掩码
# average_attn_weights: 返回各头的平均权重，否则返回每个头的权重
# is_causal: 是否使用因果掩码 若设置了 is_causal 为 True 则必须提供掩码
attention, weights = mha(
    query=x,
    key=x,
    value=x,
    key_padding_mask=padding_mask,
    need_weights=True,
    attn_mask=attn_mask,
    average_attn_weights=True,
    is_causal=True
)

print(attention.shape)
print(weights)
print(weights.shape)
