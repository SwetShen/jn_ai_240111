import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split

root_path = "./data/rps"


def _generate_dict():
    """
    返回一个类别的映射表
    :return:
    """
    files = os.listdir(root_path)
    return {str(i): filename for i, filename in enumerate(files)}


def generate_dataloader(batch_size=10):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # 随机翻转
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # 随机角度变换
        transforms.RandomRotation((-5, 5)),
        # 数据需要转化pytorch矩阵
        transforms.ToTensor(),
        # 图像数据归一化(复杂图形中的颜色、场景、内容更丰富，降低所有内容带来的影响)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root_path, transform=transform)
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader
