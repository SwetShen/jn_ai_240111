"""
训练集和验证集是互斥的
（验证集中不可以包含训练集的内容）

现实中的数据集：
1、已经设置好了训练集和验证集 （kaggle）
2、数据集只有一个（手动进行数据分割）
"""
import torch


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


# ================== 自定义数据对象 ==================
def random_split(features, pro=(0.8, 0.2)):
    if features.dim() < 2:
        raise IOError("Input features dim < 2")
    indices = torch.randperm(features.shape[0])
    features = features[indices]
    # 将第一个比例的数据，设置为训练集的长度
    train_len = int(pro[0] * len(features))
    return features[:train_len, :], features[train_len:, :]


train_points,valid_points = random_split(points, (0.7, 0.3))
print(points.shape)
print(train_points.shape,valid_points.shape)