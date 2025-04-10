import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import make_blobs, make_moons, make_circles


def _generate_np_data(name="circle", n_samples=1000):
    if name == "circle":
        return make_circles(n_samples=n_samples)  # 环形数据集
    if name == "blob":
        return make_blobs(n_samples=n_samples)  # 簇类数据集
    if name == "moon":
        return make_moons(n_samples=n_samples)  # 月牙数据集


def _generate_tensor_data(name="circle", n_samples=1000):
    X, y = _generate_np_data(name=name, n_samples=n_samples)
    return torch.from_numpy(X), torch.from_numpy(y)


class ClassificationDataset(Dataset):
    def __init__(self, name="circle", n_samples=1000):
        super().__init__()
        self.features, self.labels = _generate_tensor_data(name=name, n_samples=n_samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def generate_dataloader(name="circle", n_samples=1000, batch_size=100):
    dataset = ClassificationDataset(name=name, n_samples=n_samples)
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader,valid_loader
