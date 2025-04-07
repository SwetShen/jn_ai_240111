"""
关于链式求导问题
"""
import torch
from torch import nn
import matplotlib.pyplot as plt

# ================ 曲线图(线性回归) =================
x = torch.linspace(0, 1, 20)  # (20,)
x = x.reshape(-1, 1)  # (20,1)
y = x ** 2  # (20,1)
# ================ 构建图表 ===============
plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')
# ================ 构建模型 =================
model = nn.Sequential(
    # 输入 / 输出
    nn.Linear(1, 1),  # x -> y
    # 激活(曲线)
    nn.Sigmoid()
)
# ================ 训练前准备 =================
# 损失（评估每一次训练的结果） 衡量预测模型与真实模型之间欧式距离
criterion = nn.MSELoss()
# 优化（梯度下降）
optimizer = torch.optim.SGD(model.parameters(), 0.1)
# ================ 训练 =================
model.train()  # 开启训练模式 （允许requires_grad=True可以进行更新）
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()  # 需要每次重新求梯度（清除梯度缓存）
    predict_y = model(x)  # 预测结果（前向传播）
    # 注意：MSELoss 均方差损失要求，里面的预测值与真实值的形状必须一致
    loss = criterion(predict_y, y)  # 计算预测值与真实值之间的差距
    loss.backward()  # 开启求导模式（反向传播）
    optimizer.step()  # 更新所有的w，b

    # .item() 将torch的tensor转化为正常的数值
    print(f"epoch:{epoch + 1} / {epochs} -- loss:{loss.item():.4f}")
# ================ 预测 =================
model.eval()  # 开启预测模式（让所有的requires_grad属性变为False）
predict_y = model(x)
plt.plot(x.detach().numpy(), predict_y.detach().numpy(), 'b--')
plt.show()
