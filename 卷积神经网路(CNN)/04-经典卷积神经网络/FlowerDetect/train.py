import torch
from torch import nn
from backbones.AlexNet import AlexNet
from datasets.dataloader import generate_dataloader
from utils.train_utils import train

if __name__ == '__main__':
    device = torch.device("cuda")
    model = AlexNet(5, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, valid_loader = generate_dataloader(10)
    train(train_loader, valid_loader, model, criterion, optimizer, 1000)
