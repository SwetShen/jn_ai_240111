"""
LetNet 原始的数据需要1通道的32x32作为输入内容
"""
import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split

root_path = "./data/numbers"


def _generate_dict():
    """
    返回一个类别的映射表
    :return:
    """
    files = os.listdir(root_path)
    return {str(i): filename for i, filename in enumerate(files)}


def _generate_features():
    dict = _generate_dict()
    num_classes = len(dict)
    features = []
    labels = []
    for i in range(num_classes):
        cls_path = os.path.join(root_path, dict[str(i)])
        for file_name in os.listdir(cls_path):
            image = cv2.imread(os.path.join(root_path, dict[str(i)], file_name))
            image = cv2.resize(image, (32, 32))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = torch.from_numpy(image).unsqueeze(0)  # (1,32,32)
            features.append(image)
            labels.append(i)
    return {"features": features, "labels": torch.tensor(labels, dtype=torch.long)}


class NumberDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.features = data["features"]
        self.labels = data["labels"]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)


def generate_loaders(batch_size=1000):
    data = _generate_features()
    dataset = NumberDataset(data)
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size)
    return train_loader, valid_loader


# if __name__ == '__main__':
#     train_loader, valid_loader = generate_loaders()
#     for features, labels in train_loader:
#         print(features.shape)
#         print(labels)
#         break
