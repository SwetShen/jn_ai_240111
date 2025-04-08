import torch
from torch import nn
import matplotlib.pyplot as plt
import collections

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


# ==================== 绘制图表 ===================
def drawPoint(features, labels, ax):
    for feature, label in zip(features, labels):
        x, y = feature
        if label == 0:
            ax.plot([x.item()], [y.item()], 'ro')
        elif label == 1:
            ax.plot([x.item()], [y.item()], 'bo')


# ==================== 构建簇类图 ===================
noise = 0.2
# 簇类A label:0
points_a = torch.normal(0.5, noise, (30, 2))
labels_a = torch.zeros((30, 1))
features_a = torch.concatenate((points_a, labels_a), dim=1)
# 簇类B label:1
points_b = torch.normal(1.5, noise, (30, 2))
labels_b = torch.ones((30, 1))
features_b = torch.concatenate((points_b, labels_b), dim=1)
data = torch.concatenate((features_a, features_b), dim=0)
# 将数据进行随机打乱
indices = torch.randperm(data.shape[0])
data = data[indices]
# 声明输入输出内容
features = data[:, :-1]  # (60,2)
labels = data[:, -1]  # (60,) 此处加：为了保留维度
drawPoint(features, labels, ax1)
# ==================== 构建模型 ===================
model = nn.Sequential(
    nn.Linear(2, 5),
    nn.Tanh(),
    nn.Linear(5, 2),
    nn.Softmax(dim=-1)  # 输出的两种分类的概率是互斥的
)


# ================ 独热编码 ===================
def one_hot(features):
    # 输出不重复的类别数
    cls_len = len(collections.Counter(features.numpy()).keys())
    # 值的长度
    f_len = len(features)
    # 构建为0矩阵
    features_map = torch.zeros((f_len, cls_len))
    for i, feature in enumerate(features):
        features_map[i, feature.long()] = 1
    return features_map


# ================ 训练前准备 =================
# 损失(极大似然值公式，二值交叉熵损失)
criterion = nn.BCELoss()
# 优化（梯度下降）
optimizer = torch.optim.SGD(model.parameters(), 0.1)
# ================ 训练 =================
model.train()
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    predict_labels = model(features)
    # 二值交叉熵损失 要求左右两侧的值形状一致。
    loss = criterion(predict_labels, one_hot(labels))
    loss.backward()
    optimizer.step()

    print(f"epoch:{epoch + 1} / {epochs} -- loss:{loss.item():.4f}")

# ================ 预测模型 =================
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
drawPoint(test_features, predict, ax2)
plt.show()
