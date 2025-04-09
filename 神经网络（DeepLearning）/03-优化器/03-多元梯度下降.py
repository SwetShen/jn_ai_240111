import torch
import matplotlib.pyplot as plt

# ============= 多元梯度下降的图表 ===================
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

x1 = torch.linspace(-1, 1, 20)
x2 = torch.linspace(-1, 1, 20)
x1, x2 = torch.meshgrid([x1, x2], indexing='ij')
y = x1 ** 2 + 2 * x2 ** 2

ax1.plot_surface(x1, x2, y)
ax2.contour(x1, x2, y)
# ============= 多元梯度下降的图表 ===================
w1 = torch.tensor(1., requires_grad=True)
w2 = torch.tensor(0.8, requires_grad=True)

w1_list = []
w2_list = []
for i in range(10):
    w1_list.append(w1.item())
    w2_list.append(w2.item())
    loss = w1 ** 2 + 2 * w2 ** 2
    loss.backward()
    with torch.no_grad():
        # SGD 方法下的学习率需要自定义
        w1 -= w1.grad * 0.1
        w2 -= w2.grad * 0.1
        w1.grad.zero_()
        w2.grad.zero_()

ax2.plot(w1_list, w2_list, 'ro-')
plt.show()
