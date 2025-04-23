import torch.nn.functional
from torch.utils.data import Dataset

# 词库
vocab = '<sos> <eos> how are you ? i am fine .'.split()


# 语言数据集
class LangDataset(Dataset):
    def __init__(self):
        super().__init__()
        text = 'how are you ? i am fine .'
        words = text.split()
        # 输入数据
        inputs = ['<sos>'] + words
        # 输出数据
        outputs = words + ['<eos>']
        idx = torch.tensor([vocab.index(word) for word in inputs])
        # one_hot 编码
        self.inputs = torch.nn.functional.one_hot(idx, num_classes=len(vocab)).to(torch.float)
        # 文本预测本质是多分类任务，所以标签因该是分类索引
        self.labels = torch.tensor([vocab.index(word) for word in outputs])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.inputs, self.labels


if __name__ == '__main__':
    ds = LangDataset()
    print(ds[0])
