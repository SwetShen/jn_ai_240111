import torch


def dropout(features, persent=0.5):  # 0.5 随机削减约50%神经元（权重设置为0）
    if persent == 0:
        return features
    if persent == 1:
        return torch.zeros(features.shape)
    features[features < persent] = 0  # 防止方向传播更新
    return features


features = torch.normal(0, 1, (6, 2))
result = dropout(features)
print(result)
