import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ================== 构建多分类数据 ==================
def _generate_points(x, y, noise, num=20, label=0):
    points_x = torch.normal(x, noise, (num, 1))
    points_y = torch.normal(y, noise, (num, 1))
    points_label = torch.ones((num, 1)) * label
    return torch.concatenate([points_x, points_y, points_label], dim=1)


points_a = _generate_points(0.5, 0.5, 0.2, label=0)
points_b = _generate_points(1.5, 1.5, 0.2, label=1)
points_c = _generate_points(0.5, 1.5, 0.2, label=2)
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
