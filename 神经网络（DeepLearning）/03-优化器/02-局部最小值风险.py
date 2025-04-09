import torch
import matplotlib.pyplot as plt

# ==================绘制loss和w的图线 ==================
x = torch.linspace(-5, 5, 40)
y = torch.cos(x) * x

plt.plot(x.detach().numpy(), y.detach().numpy(), 'r-')
plt.xlabel("w")
plt.ylabel("loss")
# ================== 设置w以及loss ==================
w = torch.tensor(-3., requires_grad=True)
w_list = []
loss_list = []
for i in range(10):
    w_list.append(w.item())
    loss = torch.cos(w) * w
    loss_list.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 0.2
        w.grad.zero_()

plt.plot(w_list, loss_list, 'bo-')
plt.show()
