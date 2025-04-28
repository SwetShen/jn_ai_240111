import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

embedding = SentenceTransformer(r'D:\projects\ai_models\bge-small-zh')


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        input_text = '[CLS] 我 爱 中 国 ！'.split()
        label_text = '我 爱 中 国 ！ [SEP]'.split()
        self.inp = embedding.encode(input_text, convert_to_tensor=True)
        self.lab = torch.tensor(embedding.tokenizer.convert_tokens_to_ids(label_text))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.inp, self.lab


if __name__ == '__main__':
    ds = MyDataset()
    inp, lab = ds[0]
    print(inp.shape)
    print(lab)
