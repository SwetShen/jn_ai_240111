import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder  # 按目录加载图像文件，并按照目录自动分类
from torchvision.transforms import transforms  # 图像数据转化工具（图像增强）
from train_utils import train

# ======================  加载图像数据集 =========================
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像的numpy类型转化为torch.tensor
])
dataset = ImageFolder("./data/numbers", transform=transform)  # 分类的目录在什么位置就取到哪里
# print(dataset.classes) # 分类下标集合
# print(dataset.imgs) # 针对每个图像都有各自的分类
train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1000)
# ======================  模型 =========================
model = nn.Sequential(
    # 输入层
    nn.Conv2d(3, 16, 3, 1, 1),  # (16,28,28)
    nn.ReLU(),
    # 隐藏层
    nn.Conv2d(16, 32, 3, 2, 1),  # (32,14,14)
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, 1, 1),  # (32,14,14)
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 2, 1),  # (64,7,7)
    nn.ReLU(),
    # 输出层
    nn.Flatten(),  # (batch_size,64x7x7)
    nn.Dropout(),
    nn.Linear(64 * 7 * 7, 1024),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(1024, 10),  # 10是数字的分类数量
    nn.LogSoftmax(dim=-1)
)

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(train_loader, valid_loader, model, criterion, optimizer, 1000)
