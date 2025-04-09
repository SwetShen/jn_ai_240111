import torch
import matplotlib.pyplot as plt
import numpy as np

# ============= 构建一个复杂的3D场景 ===================
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x1 = torch.linspace(-5, 5, 40)
x2 = torch.linspace(-5, 5, 40)
x1, x2 = torch.meshgrid([x1, x2], indexing='ij')
y = torch.cos(x1) * x1 + torch.cos(x2) * x2

ax.plot_surface(x1, x2, y, cmap=plt.cm.YlGnBu_r, alpha=0.8)

# ============= 绘制起始点 ===================
w1 = torch.tensor(-3., requires_grad=True)
w2 = torch.tensor(-3., requires_grad=True)
# y = torch.cos(w1) * w1 + torch.cos(w2) * w2
# ax.plot([w1.item()], [w2.item()], [y.item()], 'ro')

points = []
# ============= 梯度下降 ===================
optimizer = torch.optim.SGD([w1, w2], lr=0.1, momentum=0.9)
for i in range(100):
    optimizer.zero_grad()
    y = torch.cos(w1) * w1 + torch.cos(w2) * w2
    points.append([w1.item(), w2.item(), y.item()])
    y.backward()
    optimizer.step()

points = np.array(points)
plt.plot(points[:, 0], points[:, 1], points[:, 2], 'ro')
plt.show()
