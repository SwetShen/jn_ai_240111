import torch
from torch import nn
import matplotlib.pyplot as plt

# ================ 曲线图(线性回归) =================
x = torch.linspace(0, 1, 20).reshape(-1, 1)
y = x ** 2
# ================ 构建图表 ===============
plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')
# ================ 构建模型 =================
model1 = nn.Sequential(
    nn.Linear(1, 1),  # x -> y
    nn.Sigmoid()
)
model2 = nn.Sequential(
    nn.Linear(1, 1),
    # nn.ReLU() # relu更多作用于神经网络之间
    nn.Tanh()
)
# ================ 训练前准备 =================
criterion = nn.MSELoss()
# 优化（梯度下降）
optimizer1 = torch.optim.SGD(model1.parameters(), 0.1)
optimizer2 = torch.optim.SGD(model2.parameters(), 0.1)
# ================ 训练 =================
model1.train()
model2.train()
epochs = 5000
for epoch in range(epochs):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    predict_y1 = model1(x)
    predict_y2 = model2(x)
    loss1 = criterion(predict_y1, y)
    loss2 = criterion(predict_y2, y)
    loss1.backward()
    loss2.backward()
    optimizer1.step()
    optimizer2.step()

    # .item() 将torch的tensor转化为正常的数值
    print(f"epoch:{epoch + 1} / {epochs} -- loss_sigmoid:{loss1.item():.4f} -- loss_tanh{loss2.item():.4f}")
