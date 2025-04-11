import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder  # 按目录加载图像文件，并按照目录自动分类
from torchvision.transforms import transforms  # 图像数据转化工具（图像增强）

# ======================  加载图像数据集 =========================
transform = transforms.Compose(
    transforms.ToTensor()  # 将图像的numpy类型转化为torch.tensor
)
dataset = ImageFolder("./data/numbers", transform=transform)  # 分类的目录在什么位置就取到哪里
# print(dataset.classes) # 分类下标集合
# print(dataset.imgs) # 针对每个图像都有各自的分类