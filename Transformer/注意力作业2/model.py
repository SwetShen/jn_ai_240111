from torch import nn
from torchvision.ops import MLP


class MyModel(nn.Module):
    # vocab_size: 词库长度
    def __init__(self, vocab_size, embed_dim=512, num_heads=4, dropout=0.):
        super().__init__()
        # 多头注意力
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        # 多层感知机
        self.mlp = MLP(embed_dim, [2048, 2048, vocab_size])

    # x: 输入的词向量
    # (N, L, embedding_dim)
    def forward(self, x):
        # 归一化
        x = self.norm(x)

        N, L, embedding_dim = x.shape
        # 构造因果注意力掩码
        attn_mask = nn.Transformer.generate_square_subsequent_mask(L)
        # 求注意力
        attn, weights = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            is_causal=True
        )

        # 残差
        x = attn + x
        # 多层感知机，求词库上的概率分布
        y = self.mlp(x)
        return y


if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer

    embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')
    texts = '[CLS] 我 爱 中 国 ！'.split()
    # 编码
    x = embedding.encode(texts, convert_to_tensor=True).unsqueeze(0)
    model = MyModel(embedding.tokenizer.vocab_size)
    y = model(x)
    print(y.shape)
