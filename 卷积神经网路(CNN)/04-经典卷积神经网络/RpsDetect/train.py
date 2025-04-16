from datasets.dataloader import generate_dataloader
from utils.train_utils import train
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch import nn

# 微调
if __name__ == '__main__':
    # 加载预训练模型(第一次运行时，会将该模型缓存到电脑中)
    # C:\Users\当前电脑的用户名\.cache\torch\hub\checkpoints
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 修改官方模型的输出
    model.fc = nn.Linear(512, 3)

    device = torch.device("cuda")
    model.device = device  # 在模型中设置一个device属性
    model = model.to(device)  # 设置模型的device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, valid_loader = generate_dataloader(10)
    train(train_loader, valid_loader, model, criterion, optimizer, 1000)
