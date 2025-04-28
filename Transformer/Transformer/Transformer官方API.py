# api: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
import torch
from torch import nn

# 使用 pytorch 官方 api nn.Transformer
# 参数:
# d_model: 输入输出嵌入的特征维度
# nhead: 多头注意力的头数
# num_encoder_layers: 编码器层数
# num_decoder_layers: 解码器层数
# dim_feedforward: 前馈神经网络的中间输出特征
# dropout: dropout 概率
# activation: 激活函数 带选项: relu or gelu
# custom_encoder: 自定义编码器
# custom_decoder: 自定义解码器
# layer_norm_eps: 层归一化的 eps 系数
# batch_first: 是否把皮次数放到数据的第一个维度
# norm_first: 是否先归一化在执行编解码，预归一化可以使 Transformers 的训练更加有效或高效
model = nn.Transformer(
    d_model=512,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=5,
    dim_feedforward=2048,
    dropout=0.1,
    activation=nn.ReLU(),
    layer_norm_eps=1e-8,
    batch_first=True,
    norm_first=True,
    bias=False,
)

src = torch.randn(5, 10, 512)
tgt = torch.randn(5, 20, 512)

# 官方 Transformer 的输入参数如下
# src: 输入序列张量
# tgt: 输出序列张量
# src_mask: 输入序列的掩码
# tgt_mask: 输出序列的掩码
# memory_mask: 编码器输出结果的掩码
# src_key_padding_mask: 输入序列计算出 K 后的掩码
# tgt_key_padding_mask: 输出序列 K 的掩码
# memory_key_padding_mask: 编码器输出结果 memory，在执行编解码器注意力前，充当 key 时的掩码，例如:
# x, weights = self.mha(x, memory, memory, attn_mask=src_mask)，此时第二个参数 memory 就是 key，可以在调用 mha 前，对 memory 进行掩码处理
# src_is_causal: 是否使用因果掩码，若使用，则需要提供对应的掩码
# torch.nn.functional 5288 行提示: 可以使用 nn.Transformer.generate_square_subsequent_mask 函数生成掩码
# tgt_is_causal: 是否使用因果掩码，若使用，则需要提供对应的掩码
# memory_is_causal: 是否使用因果掩码，若使用，则需要提供对应的掩码
# 若设置了 is_causal 为 True 则必须提供对应的掩码

# 假设输入序列 src 中 后面 5 个字全是 pad 则添加填充掩码
src_key_padding_mask = torch.tensor([
    [0, 0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    [0, 0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    [0, 0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    [0, 0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
    [0, 0, 0, 0, 0, float('-inf'), float('-inf'), float('-inf'), float('-inf'), float('-inf')],
])
memory_key_padding_mask = src_key_padding_mask
tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])

y = model(
    src,
    tgt,
    tgt_mask=tgt_mask,
    src_key_padding_mask=src_key_padding_mask,
    memory_key_padding_mask=memory_key_padding_mask,
    tgt_is_causal=True
)

print(y.shape)
