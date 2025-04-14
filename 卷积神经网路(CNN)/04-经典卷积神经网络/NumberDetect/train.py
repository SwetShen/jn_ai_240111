import torch
from torch import nn
from backbones.lenet import LetNet
from datasets.dataloader import generate_loaders, _generate_dict
from utils.train_utils import train

if __name__ == '__main__':
    model = LetNet(10)
    train_loader, valid_loader = generate_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(train_loader, valid_loader, model, criterion, optimizer, 1000)
