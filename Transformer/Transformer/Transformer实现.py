import torch
from torch import nn
from Transformer.位置编码实现 import PositionEncoding
from sentence_transformers import SentenceTransformer


class FFN(nn.Module):
    # hidden_dim: 隐藏层的维度
    def __init__(self, embed_dim, dim_feedforward=2048, dropout=0.):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )

    def forward(self, x):
        return self.stack(x)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.):
        super().__init__()
        # 编码器自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, dim_feedforward=dim_feedforward, dropout=dropout)

    # src: 输入序列
    def forward(self, src, key_padding_mask=None, attn_mask=None):
        # 恒等映射
        identity = src
        # 自注意力
        x, weights = self.self_attn(
            query=src,
            key=src,
            value=src,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=False,
        )
        # 残差
        x = x + identity
        # 归一化
        x = self.norm(x)
        # 恒等映射
        identity = x
        # 前馈神经网络
        x = self.ffn(x)
        # 残差
        x = x + identity
        # 归一化
        memory = self.norm(x)
        return memory


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.):
        super().__init__()
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # 编码器-解码器注意力
        self.encoder_decoder_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, memory, tgt, key_padding_mask=None, encoder_decoder_attn_mask=None):
        # 恒等映射
        identity = tgt
        # 因果自注意力
        attn_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        x, weights = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            is_causal=True,
        )
        # 残差
        x = x + identity
        # 归一化
        x = self.norm(x)
        # 恒等映射
        identity = x
        # 编码器-解码器注意力
        x, weights = self.encoder_decoder_attn(
            query=tgt,
            key=memory,
            value=memory,
            # key_padding_mask: 此处的掩码用于频闭 memory 中出现 pad 的位置
            key_padding_mask=key_padding_mask,
            attn_mask=encoder_decoder_attn_mask,
        )
        # 残差
        x = x + identity
        # 归一化
        x = self.norm(x)
        # 恒等映射
        identity = x
        # 前馈神经网络
        x = self.ffn(x)
        # 残差
        x = x + identity
        # 归一化
        y = self.norm(x)
        return y


class Transformer(nn.Module):
    # num_layers: 堆叠编码器和解码器的层数
    def __init__(self, embed_dim, num_heads, num_layers=1, dim_feedforward=2048, dropout=0.):
        super().__init__()
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            *(Encoder(embed_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout) for _ in
              range(num_layers))
        ])
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            *(Decoder(embed_dim, num_heads, dim_feedforward=dim_feedforward, dropout=dropout) for _ in
              range(num_layers))
        ])

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = src
        # 调用编码器层
        for layer in self.encoder_layers:
            memory = layer(memory, key_padding_mask=src_key_padding_mask, attn_mask=src_mask)
        y = tgt
        # 调用解码器层
        for layer in self.decoder_layers:
            y = layer(memory, y, key_padding_mask=tgt_key_padding_mask, encoder_decoder_attn_mask=tgt_mask)
        return y


# 假设一个聊天机器人模型，我们做一个 Transformer 的封装
class ChatBot(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.transformer = Transformer(embed_dim=embed_dim, num_layers=2, num_heads=4, dropout=0.)
        self.pe = PositionEncoding(embed_dim)
        self.embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh').eval()
        # 冻结参数
        for p in self.embedding.parameters():
            p.requires_grad = False
        # 分类全连接
        self.fc_out = nn.Linear(embed_dim, self.embedding.tokenizer.vocab_size)

    # ids: 带批次的索引序列
    def embed_fn(self, ids):
        tokens = [self.embedding.tokenizer.convert_ids_to_tokens(_ids) for _ids in ids]
        return torch.stack([self.embedding.encode(token, convert_to_tensor=True) for token in tokens])

    # src, tgt: 文本索引
    def forward(self, src, tgt):
        # 词嵌入
        src = self.embed_fn(src)
        tgt = self.embed_fn(tgt)
        # 位置编码
        src = self.pe(src)
        tgt = self.pe(tgt)
        # 调用 transformer 模型
        y = self.transformer(src, tgt)
        # y (N, L, embed_dim)
        y = self.fc_out(y)
        return y


if __name__ == '__main__':
    model = ChatBot()
    model([[123, 234, 456, 789]], [[123, 234, 456, 789]])
