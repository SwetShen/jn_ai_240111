import torch
import matplotlib.pyplot as plt

x1 = torch.linspace(-2, 2, 40)
x2 = torch.linspace(-2, 2, 40)
x1, x2 = torch.meshgrid([x1, x2], indexing='ij')
y = x1 ** 2 + 2 * x2 ** 2

plt.contour(x1, x2, y)
# ============= 多元梯度下降的图表 ===================
w1 = torch.tensor(2., requires_grad=True)
w2 = torch.tensor(1.5, requires_grad=True)
# 设置w1,w2 动量参数
v1 = torch.tensor(1.)
v2 = torch.tensor(1.)
# 衰减因子
beta = 0.9

w1_list = []
w2_list = []
for i in range(20):
    w1_list.append(w1.item())
    w2_list.append(w2.item())
    loss = w1 ** 2 + 2 * w2 ** 2
    loss.backward()
    with torch.no_grad():
        # 动量衰减法
        v1 = beta * v1 + (1 - beta) * w1.grad  # 梯度向量累加/ 动量累加
        v2 = beta * v2 + (1 - beta) * w2.grad
        w1 -= 0.1 * v1
        w2 -= 0.1 * v2
        w1.grad.zero_()
        w2.grad.zero_()

plt.plot(w1_list, w2_list, 'ro-')
plt.show()
