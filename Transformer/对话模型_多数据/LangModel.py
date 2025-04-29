import torch
from torch import nn
from embedding_model import embedding, tokenizer
from Transformer.位置编码实现 import PositionEncoding


class LangModel(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = embedding.get_sentence_embedding_dimension()
        self.nhead = 8
        self.pe = PositionEncoding(embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=self.nhead,
            num_encoder_layers=16,
            num_decoder_layers=16,
            dim_feedforward=4096,
            batch_first=True,
            norm_first=True,
            dropout=0.5
        )
        self.fc_out = nn.Linear(embed_dim, embedding.tokenizer.vocab_size)

    # ids: 带批次的索引列表 (N, 50)
    def embedding(self, ids):
        return embedding[0].auto_model.embeddings.word_embeddings(ids)

    # src: 带批次的索引列表
    # tgt: 带批次的索引列表
    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        # 词嵌入和位置编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        src = self.pe(src)
        tgt = self.pe(tgt)

        # 构造解码器的因果注意力掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).unsqueeze(0).expand(
            src.shape[0] * self.nhead, -1, -1)
        # 将新建的张量放到模型对应的设备上
        tgt_mask = tgt_mask.to(src.device)

        # 调用transformer
        y = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_is_causal=True,
        )

        y = self.fc_out(y)
        return y


if __name__ == '__main__':
    from dataset import LangDataset
    from torch.utils.data import DataLoader

    model = LangModel()

    ds = LangDataset()
    dl = DataLoader(ds, batch_size=5, shuffle=True)
    for i, (src, tgt, label) in enumerate(dl):
        src, src_key_padding_mask = tokenizer(src)
        tgt, tgt_key_padding_mask = tokenizer(tgt)
        label, _ = tokenizer(label)
        y = model(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        print(y.shape)
