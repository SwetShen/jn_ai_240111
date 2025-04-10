import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn


# ================== 构建多分类数据 ==================
def _generate_points(x, y, noise, num=20, label=0):
    points_x = torch.normal(x, noise, (num, 1))
    points_y = torch.normal(y, noise, (num, 1))
    points_label = torch.ones((num, 1)) * label
    return torch.concatenate([points_x, points_y, points_label], dim=1)


points_a = _generate_points(0.5, 0.5, 0.2, num=40, label=0)
points_b = _generate_points(1.5, 1.5, 0.2, num=40, label=1)
points_c = _generate_points(0.5, 1.5, 0.2, num=40, label=2)
points = torch.concatenate([points_a, points_b, points_c], dim=0)


# ================== 构建数据集类 ==================

class MyDataset(Dataset):
    def __init__(self, data):  # data 初始化时从外部传入的数据
        super().__init__()
        self.data = data
        self.features = self.data[:, :-1]
        self.labels = self.data[:, -1]

    def __len__(self):  # 返回数据集的长度
        return len(self.features)

    def __getitem__(self, index):  # 返回数据集中的指定的某一个数据
        return self.features[index], self.labels[index]


# ================== 初始化数据集 ==================
dataset = MyDataset(points)
train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
# 引入数据加载器 batch_size=100 每一100条数据为一个批次  shuffle=True 将数据顺序打乱
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=10)


# ================== 创建模型 ==================
class CustomNet(nn.Module):
    def __init__(self, num_classes=1000):  # num_classes 分类的数量
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)


# ================== 训练模型 ==================
model = CustomNet(3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 100
for epoch in range(epochs):
    print(f"epoch:{epoch + 1} / {epochs}")
    loss_list = []
    acc_list = []
    model.train()  # 开启训练模式 (训练集)
    for i, (features, labels) in enumerate(train_loader):  # batch_size
        optimizer.zero_grad()
        predict_labels = model(features)
        loss = criterion(predict_labels, labels.long())
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        print(f"batch:{i + 1} -- loss:{loss.item():.4f}")
    model.eval()  # 开启验证模式 (验证集)
    for i, (features, labels) in enumerate(valid_loader):  # batch_size
        predict_labels = model(features)
        predict_labels = torch.argmax(predict_labels, dim=-1)
        acc = sum(predict_labels == labels) / len(labels)

        acc_list.append(acc)
        print(f"batch:{i + 1} -- acc:{acc * 100:.2f}%")

    # 计算平均损失，平均准确率
    avg_loss = sum(loss_list) / len(loss_list)
    avg_acc = sum(acc_list) / len(acc_list)
    print(f"-- avg_loss:{avg_loss:.4f} -- avg_acc:{avg_acc * 100:.2f}%")
