import torch
import matplotlib.pyplot as plt

# ============= 多元梯度下降的图表 ===================
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

x1 = torch.linspace(-2, 2, 20)
x2 = torch.linspace(-2, 2, 20)
x1, x2 = torch.meshgrid([x1, x2], indexing='ij')
y = x1 ** 2 + 2 * x2 ** 2

ax1.plot_surface(x1, x2, y)
ax2.contour(x1, x2, y)
# ============= 多元梯度下降的图表 ===================
w1 = torch.tensor(2., requires_grad=True)
w2 = torch.tensor(2., requires_grad=True)
# 定义自适应学习率衰减算法的参数
eta = 0.5  # 学习率
S1 = 1.
S2 = 1.
epsilon = 1e-6  # 防止分母为0
beta = 0.9

w1_list = []
w2_list = []
for i in range(10):
    w1_list.append(w1.item())
    w2_list.append(w2.item())
    loss = w1 ** 2 + 2 * w2 ** 2
    loss.backward()
    with torch.no_grad():
        # RMSprop 自适应学习率衰减法
        S1 = beta * S1 + (1 - beta) * w1.grad ** 2
        S2 = beta * S2 + (1 - beta) * w2.grad ** 2
        w1 -= eta / (S1 ** 0.5 + epsilon) * w1.grad
        w2 -= eta / (S2 ** 0.5 + epsilon) * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

ax2.plot(w1_list, w2_list, 'ro-')
plt.show()
