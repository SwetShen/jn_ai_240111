import torch
from torch.utils.data import Dataset

text = '<sos> <eos> how are you ? i am fine .'
vocab = text.split()


class LangDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.src = 'how are you ?'
        self.tgt = '<sos> i am fine .'
        self.label = 'i am fine . <eos>'
        # src 和 tgt 需要 one-hot 编码
        self.src = self.encode_text(self.src)
        self.tgt = self.encode_text(self.tgt)
        # label 需要转换成索引
        self.label = torch.tensor([vocab.index(word) for word in self.label.split()])

    def encode_text(self, text):
        # 拆分文本
        words = text.split()
        # 索引
        idx = torch.tensor([vocab.index(word) for word in words])
        # one-hot 编码
        return torch.nn.functional.one_hot(idx, num_classes=len(vocab)).float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.src, self.tgt), self.label


if __name__ == '__main__':
    ds = LangDataset()
    print(ds[0])
