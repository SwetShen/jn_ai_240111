import torch
from torch.utils.data import Dataset

token_map = ['法外狂徒', '律师', '牛头人高手']


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = ['张三', '罗翔', '老王']
        self.labels = [0, 1, 2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inp = self.inputs[idx]
        label = torch.tensor(self.labels[idx])
        return inp, label


if __name__ == '__main__':
    ds = MyDataset()
    print(len(ds))
    print(ds[0])
    print(ds[1])
    print(ds[2])
