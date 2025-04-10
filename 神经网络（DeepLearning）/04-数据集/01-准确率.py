import torch

# 预测值（softmax输出的概率值）
predict = torch.tensor([
    [0.1, 0.9],
    [0.7, 0.3],
    [0.2, 0.8],
    [0.9, 0.1],
    [0.51, 0.49]
])
# 真实值
labels = torch.tensor([0, 1, 1, 1, 1])
# 计算准确率
predict_labels = torch.argmax(predict, dim=-1)
acc = sum(predict_labels == labels) / len(labels)
print(f"acc:{acc * 100:.2f}%")
