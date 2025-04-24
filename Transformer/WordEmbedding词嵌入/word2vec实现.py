from torch import nn


class Word2Vec(nn.Module):
    # embedding_dim: 词向量的维度
    # padding_idx: 填充字符的索引
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(
            # 词库的长度
            num_embeddings=vocab_size,
            # 词嵌入向量的维度
            # 词向量的维度
            embedding_dim=embedding_dim,
            # 填充值的索引
            padding_idx=padding_idx,
            # 词向量的最大长度
            max_norm=5.
        )
        # 线性层
        self.fc = nn.Linear(embedding_dim, vocab_size)

    # x: 文本索引
    # x (N, L)
    def forward(self, x):
        # embedding: 嵌入后的向量值
        # embedding (N, L, embedding_dim)
        embedding = self.embedding(x)
        # 合并向量
        # 求平均嵌入
        # avg_embedding (N, embedding_dim)
        avg_embedding = embedding.mean(dim=1)
        # 线性层
        y = self.fc(avg_embedding)
        return y


if __name__ == '__main__':
    import torch

    vocab = '<sos> <eos> <pad> my name is very hard to remember .'.split()
    idx = torch.tensor([
        [vocab.index(word) for word in 'my name very hard'.split()],
        [vocab.index(word) for word in 'very hard remember .'.split()],
    ])

    model = Word2Vec(len(vocab), 512, vocab.index('<pad>'))
    y = model(idx)
    print(y.shape)

    # ignore_index: 被忽略的索引，被忽略的索引将不会计算损失
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.index('<pad>'))
    # 我们用上下文预测中间字
    labels = torch.tensor([vocab.index('is'), vocab.index('to')])
    loss = loss_fn(y, labels)
    print(loss)
    loss.backward()
