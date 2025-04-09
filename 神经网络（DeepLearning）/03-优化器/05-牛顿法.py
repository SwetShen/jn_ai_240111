import torch
import matplotlib.pyplot as plt
from random import random

x1 = torch.linspace(-1, 1, 20)
x2 = torch.linspace(-1, 1, 20)
x1, x2 = torch.meshgrid([x1, x2], indexing='ij')
y = x1 ** 2 + x2 ** 2

plt.contour(x1, x2, y)
# ============= 多元梯度下降的图表 ===================
w1 = torch.tensor(1., requires_grad=True)
w2 = torch.tensor(0.8, requires_grad=True)

w1_list = []
w2_list = []
for i in range(10):
    w1_list.append(w1.item())
    w2_list.append(w2.item())
    loss = w1 ** 2 + w2 ** 2
    loss.backward()
    with torch.no_grad():
        # 牛顿法（万能逼近定理）：w = w - w的一阶导 / w的二阶导
        w1 -= 2 * w1 / 2
        w2 -= 2 * w2 / 2
        w1.grad.zero_()
        w2.grad.zero_()

plt.plot(w1_list, w2_list, 'ro-')
plt.show()
