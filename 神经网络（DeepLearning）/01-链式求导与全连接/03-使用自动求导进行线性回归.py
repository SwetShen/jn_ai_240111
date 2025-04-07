"""
关于链式求导问题
"""
import torch
from torch import nn
import matplotlib.pyplot as plt

# ================ 散点图(线性回归) =================
noise = 0.2
x = torch.linspace(0, 1, 20)  # (20,)
x = x.reshape(-1, 1)  # (20,1)
y = 3 * x + 2  # (20,1)
y += torch.normal(0, noise, y.shape)  # (20,)
# ================ 构建图表 ===============
plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')
# ================ 设置训练的参数 ===============
w = torch.tensor(0.1, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)

predict_y = w * x + b
line, = plt.plot(x.detach().numpy(), predict_y.detach().numpy(), 'b--')
# ================ 训练 =================
epochs = 1000
for epoch in range(epochs):
    predict_y = w * x + b
    # 动态更新预测线
    line.set_data(x.detach().numpy(), predict_y.detach().numpy())
    loss = (predict_y - y) ** 2
    torch.sum(loss).backward()
    # 以下步骤对应：optimizer.step()  optimizer.zero_grad()
    with torch.no_grad():  # 以下内容不需要累加导数
        w -= w.grad * 0.01
        b -= b.grad * 0.01
        w.grad.zero_()  # 清除w中梯度的缓存
        b.grad.zero_()  # 清除b中梯度的缓存

    print(f"epoch:{epoch + 1} / {epochs} -- loss:{torch.sum(loss).item():.4f}")
    # 延迟时间
    plt.pause(0.1)  # 单位秒
