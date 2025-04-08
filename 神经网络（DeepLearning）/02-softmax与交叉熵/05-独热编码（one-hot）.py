import torch
import collections

a = torch.tensor([0, 3, 1, 1, 1, 2])


# 对a进行独热编码
def one_hot(features):
    # 输出不重复的类别数
    cls_len = len(collections.Counter(features.numpy()).keys())
    # 值的长度
    f_len = len(features)
    # 构建为0矩阵
    features_map = torch.zeros((f_len, cls_len))
    for i, feature in enumerate(features):
        features_map[i, feature] = 1
    return features_map


features_map = one_hot(a)
print(features_map)
