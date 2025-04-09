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
# 设置多项式衰减的参数
eta_0 = 0.1  # 就是固定的学习率
beta = 0.01
t = 0  # 时间步
alpha = 2

w1_list = []
w2_list = []
for i in range(10):
    w1_list.append(w1.item())
    w2_list.append(w2.item())
    loss = w1 ** 2 + 2 * w2 ** 2
    loss.backward()
    with torch.no_grad():
        # 多项式衰减法(学习率的时间步自适应) ==> Lr_Scheduler 学习率衰减法
        # 真实的训练场景中，t（时间步）就是epoch（比如：每100个epoch衰减一次）
        w1 -= w1.grad * eta_0 * (beta * t + 1) ** (- alpha)
        w2 -= w2.grad * eta_0 * (beta * t + 1) ** (- alpha)
        w1.grad.zero_()
        w2.grad.zero_()
    # 时间步变化
    t += 1

plt.plot(w1_list, w2_list, 'ro-')
plt.show()