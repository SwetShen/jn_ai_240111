import torch
from torch import nn
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# 简单模型
net01 = nn.Sequential(
    nn.Linear(1, 1)
)

# 复杂模型
net02 = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

x = torch.linspace(0, 1, 10).reshape(-1, 1)
y = 3 * x + 2
y += torch.normal(0, 0.2, y.shape)

ax1.plot(x.detach().numpy(), y.detach().numpy(), 'ro-')
ax2.plot(x.detach().numpy(), y.detach().numpy(), 'ro-')

criterion = nn.MSELoss()
optimizer1 = torch.optim.SGD(net01.parameters(), 0.1)
optimizer2 = torch.optim.SGD(net02.parameters(), 0.1)

epochs = 10000
for epoch in range(epochs):
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    predict1 = net01(x)
    predict2 = net02(x)
    loss1 = criterion(predict1, y)
    loss2 = criterion(predict2, y)
    print(f"epoch {epoch+1}/{epochs} -- loss1:{loss1.item():.4f} -- loss2:{loss2.item():.4f}")
    loss1.backward()
    optimizer1.step()
    loss2.backward()
    optimizer2.step()

predict1 = net01(x)
ax1.plot(x.detach().numpy(), predict1.detach().numpy(), 'b^--')
predict2 = net02(x)
ax2.plot(x.detach().numpy(), predict2.detach().numpy(), 'b^--')
plt.show()
