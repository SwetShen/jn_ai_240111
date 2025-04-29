import torch
from torch.utils.data import Dataset
from embedding_model import embedding


class LangDataset(Dataset):
    # max_len: 序列的最大长度
    def __init__(self, max_len=50):
        super().__init__()
        self.max_len = max_len
        self.src = '你好，吃了没？'
        self.tgt = '[CLS]吃了，你呢？'
        self.label = '吃了，你呢？[SEP]'

    def __len__(self):
        return 1

    def embedding(self, texts):
        result = embedding.tokenizer(texts, add_special_tokens=False, max_length=self.max_len, padding='max_length',
                                     return_tensors='pt')
        ids = result['input_ids'][0]
        # tokens = embedding.tokenizer.convert_ids_to_tokens(ids)
        # word_vector = embedding.encode(tokens, convert_to_tensor=True)
        padding_mask = result['attention_mask'][0].float()
        padding_mask[padding_mask == 0] = float('-inf')
        padding_mask[padding_mask == 1] = 0
        return ids, padding_mask

    def __getitem__(self, idx):
        src, src_key_padding_mask = self.embedding(self.src)
        tgt, tgt_key_padding_mask = self.embedding(self.tgt)
        label = \
            embedding.tokenizer(self.label, add_special_tokens=False, max_length=self.max_len, padding='max_length',
                                return_tensors='pt')['input_ids'][0]
        return src, tgt, src_key_padding_mask, tgt_key_padding_mask, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = LangDataset()
    src, tgt, src_key_padding_mask, tgt_key_padding_mask, label = ds[0]
    print(src, tgt, src_key_padding_mask.shape, tgt_key_padding_mask.shape, label.shape)
    dl = DataLoader(ds, 1)
    for i, (src, tgt, src_key_padding_mask, tgt_key_padding_mask, label) in enumerate(dl):
        print(src)
