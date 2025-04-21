# 自定义数据集，必须继承 torch.utils.data.Dataset 类
# 必须实现 Dataset 的 __init__ __len__ 和 __getitem__
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import pandas as pd


class MyDataset(Dataset):
    # 通常在 init 中加载元数据
    # 通常元数据是 .csv 格式文件
    # transform: 用于将数据转换成张量的函数
    # target_transform: 用于将标签转换成张量的函数
    def __init__(self, transform=None, target_transform=None):
        super().__init__()
        # 读元数据
        self.df = pd.read_csv('meta.csv', encoding='gbk')
        self.transform = transform
        self.target_transform = target_transform

    # 返回数据集的长度
    def __len__(self):
        return len(self.df)

    # 返回用于训练的数据
    # 通常包含两个值，分别为数据和标签
    # idx: 要加载的数据索引
    def __getitem__(self, idx):
        # 加载一行数据
        row = self.df.iloc[idx]
        img_path = row['img_path']
        label = row['label']
        # 加载图片
        img = Image.open(img_path)
        # 判断转换器是否可调用
        if callable(self.transform):
            # 调用转换器，转换图片为张量
            img = self.transform(img)
        if callable(self.target_transform):
            # 转换标签为张量
            label = self.target_transform(label)
        # 返回数据和标签
        return img, label


if __name__ == '__main__':
    tsf = ToTensor()


    # img: PIL.Image.Image
    # def transform(img):
    #     tensor = tsf(img)
    #     return tensor


    # 标签对数字的映射表
    # label_map = {'头盔': 0, '情书': 1}
    label_map = ['头盔', '情书']


    def target_transform(label):
        return torch.tensor(label_map.index(label))


    ds = MyDataset(transform=tsf, target_transform=target_transform)
    print(len(ds))
    print(ds[0])
