from backbone import ClassificationNet
from datasets import generate_dataloader
from train_utils import train
import torch
from torch import nn

if __name__ == '__main__':
    model = ClassificationNet(2)
    train_loader, valid_loader = generate_dataloader(name="moon")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(train_loader, valid_loader, model, criterion, optimizer, 1000)
