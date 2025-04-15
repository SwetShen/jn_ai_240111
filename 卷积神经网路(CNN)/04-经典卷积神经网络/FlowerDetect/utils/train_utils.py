import torch
from tqdm import tqdm


def train_epoch(model, features, labels, criterion, optimizer):
    features = features.to(model.device)
    labels = labels.to(model.device)
    optimizer.zero_grad()
    predict_labels = model(features.float())
    loss = criterion(predict_labels, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def valid_epoch(model, features, labels):
    features = features.to(model.device)
    labels = labels.to(model.device)
    predict_labels = model(features.float())
    predict_labels = torch.argmax(predict_labels, dim=-1)
    acc = sum(predict_labels == labels) / len(labels)

    return acc


def train(train_loader, valid_loader, model, criterion, optimizer, epochs):
    best_acc = 0
    for epoch in range(epochs):
        loss_list = []
        acc_list = []
        model.train()
        loop1 = tqdm(train_loader)
        for features, labels in loop1:
            loss = train_epoch(model, features, labels, criterion, optimizer)
            loss_list.append(loss)
            loop1.set_description(f"train_loss:{loss:.4f}")
        loop1.clear()
        loop1.close()

        loop2 = tqdm(valid_loader)
        for features, labels in loop2:
            acc = valid_epoch(model, features, labels)
            acc_list.append(acc)
            loop2.set_description(f"valid_acc:{acc * 100:.2f}%")
        loop2.clear()
        loop2.close()

        avg_loss = sum(loss_list) / len(loss_list)
        avg_acc = sum(acc_list) / len(acc_list)
        print(f"-- avg_loss:{avg_loss:.4f} -- avg_acc:{avg_acc * 100:.2f}%")

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), "./save/best.pt")
            best_acc = avg_acc
