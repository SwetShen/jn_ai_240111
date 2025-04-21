# torchvision 官方库中已经囊括了一些常用的数据集，参考: https://pytorch.org/vision/stable/datasets.html?highlight=torchvision+datasets
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as T

ds = MNIST(
    # 数据集下载并存放的路径
    root='data',
    # 是否下载训练集
    train=True,
    # 是否下载
    download=False,
    # 输入数据转换器
    transform=T.ToTensor(),
    # 标签转换器
    target_transform=lambda label: torch.tensor(label)
)

print(len(ds))
print(ds[0])
