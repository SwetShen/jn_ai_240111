import torch
from torch import nn


def train_epoch(model, features, labels, criterion, optimizer):
    optimizer.zero_grad()
    predict_labels = model(features.float())
    loss = criterion(predict_labels, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def valid_epoch(model, features, labels):
    predict_labels = model(features.float())
    predict_labels = torch.argmax(predict_labels, dim=-1)
    acc = sum(predict_labels == labels) / len(labels)

    return acc


def train(train_loader, valid_loader, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        loss_list = []
        acc_list = []
        model.train()
        for features, labels in train_loader:
            loss = train_epoch(model, features, labels, criterion, optimizer)
            loss_list.append(loss)

        for features, labels in valid_loader:
            acc = valid_epoch(model, features, labels)
            acc_list.append(acc)

        avg_loss = sum(loss_list) / len(loss_list)
        avg_acc = sum(acc_list) / len(acc_list)
        print(f"-- avg_loss:{avg_loss:.4f} -- avg_acc:{avg_acc * 100:.2f}%")
