import pandas as pd
import json

from torch.utils.data import Dataset


class LangDataset(Dataset):
    # max_row: 代表获取 data.csv 中的最大行数
    def __init__(self, max_row=5):
        super().__init__()
        # 加载元数据
        self.df = pd.read_csv('data.csv', nrows=max_row, encoding='utf-8')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        conversation = json.loads(row['conversation'])[0]
        src = conversation['human']
        assistant = conversation['assistant']
        tgt = '[CLS]' + assistant
        label = assistant + '[SEP]'
        return src, tgt, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    ds = LangDataset()
    # print(ds[0])
    dl = DataLoader(ds, 5)
    for src, tgt, label in dl:
        print(src, tgt, label)
