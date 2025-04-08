import collections

import torch
from torch import nn

# 输出的类别值(实际运算中是one_hot 独热编码格式)
p1 = torch.tensor([[1., 0., 0.],
                   [0., 1., 0.],
                   [1., 0., 0.],
                   [0., 0., 1.]])
p2 = torch.tensor([0, 1, 0, 2])
# 预测概率,每一种类别都有一个概率
q = torch.tensor([[0.2, 0.7, 0.1],
                  [0.8, 0.1, 0.1],
                  [0.1, 0.7, 0.2],
                  [0.6, 0.2, 0.2]])


# ===========框架下的交叉熵=======
# # 1、CrossEntropyLoss 内部会将预测概率值进行softmax运算
# criterion = nn.CrossEntropyLoss()  # ce = - p * log(q) - (1-p) *log(1-q)
# loss1 = criterion(q, p1)  # 2、使用交叉熵时，实际标记可以不需要one_hot(独热编码)
# print(loss1)
# loss2 = criterion(q, p2)
# print(loss2)
# ===========自定义交叉熵=======

def one_hot(features):
    cls_len = len(collections.Counter(features.numpy()).keys())
    f_len = len(features)
    features_map = torch.zeros((f_len, cls_len))
    for i, feature in enumerate(features):
        features_map[i, feature] = 1
    return features_map


def softmax(input, dim=1):
    tmp = torch.exp(input)
    return tmp / torch.sum(tmp, dim=dim).unsqueeze(-1)


def cross_entropy_loss(q, p):
    # 无论在外部是否进行了softmax，都需要在交叉熵中执行一次softmax运算
    q = softmax(q)
    # 当真实值的维度只有一个维度时，需要进行one_hot
    if p.dim() == 1:
        p = one_hot(p)
    # 计算交叉熵
    result = torch.sum(- p * torch.log(q), dim=-1) # 三种不同的熵的概率相加
    return torch.mean(result)


print(cross_entropy_loss(q, p2))
print(cross_entropy_loss(q, p1))
