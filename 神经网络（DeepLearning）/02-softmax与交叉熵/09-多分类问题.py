import torch
from torch import nn
import matplotlib.pyplot as plt

# ====================== 构建图表 ======================
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


# ====================== 绘制图表 ======================
def draw_point(features, labels, ax):
    for feature, label in zip(features, labels):
        x, y = feature
        if label == 0:
            ax.plot([x.item()], [y.item()], 'ro')
        elif label == 1:
            ax.plot([x.item()], [y.item()], 'bo')
        elif label == 2:
            ax.plot([x.item()], [y.item()], 'go')


# ====================== 构建多分类场景 ======================
def _generate_points(x, y, noise, num=20, label=0):
    points_x = torch.normal(x, noise, (num, 1))
    points_y = torch.normal(y, noise, (num, 1))
    points_label = torch.ones((num, 1)) * label
    return torch.concatenate([points_x, points_y, points_label], dim=1)


# 簇类A
points_a = _generate_points(0.5, 0.5, 0.2, label=0)
# 簇类B
points_b = _generate_points(1.5, 1.5, 0.2, label=1)
# 簇类C
points_c = _generate_points(0.5, 1.5, 0.2, label=2)
# ====================== 将簇类数据进行预处理 ======================
points = torch.concatenate([points_a, points_b, points_c], dim=0)
indices = torch.randperm(points.shape[0])
points = points[indices]
features = points[:, :-1]  # (60,2)
labels = points[:, -1]  # (60,)
draw_point(features, labels, ax1)
# ====================== 构建模型 ======================
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 3),
    nn.LogSoftmax(dim=-1)  # 防止交叉熵进行二次softmax
)
# ====================== 损失、优化器 ======================
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# ====================== 训练 ======================
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    predict_labels = model(features)
    # 在交叉熵损失运算时，要求labels必须是long类型
    loss = criterion(predict_labels, labels.long())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch: {epoch + 1}/{epochs} -- loss:{loss.item():.4f}")
# ====================== 预测 ======================
# 伪造数据
x = torch.linspace(0, 2, 20)
y = torch.linspace(0, 2, 20)
x, y = torch.meshgrid([x, y], indexing='ij')
# x:(20,20,1) y:(20,20,1) => (20,20,2)
# stack 可以叠加不存在的维度（升维），concatenate 只能叠加存在的维度（维度不变）
test_features = torch.stack((x, y), dim=-1).reshape(-1, 2)  # (400,2)
model.eval()  # 开始评估模式
test_labels = model(test_features)
# argmax()取出该分类中最大概率的下标
predict = torch.argmax(test_labels, dim=-1)
draw_point(test_features, predict, ax2)
plt.show()
