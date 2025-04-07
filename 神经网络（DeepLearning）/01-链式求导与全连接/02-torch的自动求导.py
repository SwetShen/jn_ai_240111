# torch自动求导的过程
import torch

"""
自动求导的流程：
步骤1：哪个变量需要求导，就在该变量上加入requires_grad=True 属性（允许该变量可以被求导）
步骤2：需要将计算结果(总和、总的平均值)执行backward() 执行反向传播（开启求导）
步骤3：通过“变量.grad”直接获取导数
"""

# ---------------------------------------------------
# x = torch.tensor([2., 3., 4., 5.], requires_grad=True)
# ---------------------------------------------------
# x = torch.tensor([2., 3., 4., 5.])
# x.requires_grad = True  # 可以单独设置该属性
# ---------------------------------------------------
x = torch.tensor([2, 3, 4, 5])
x = x.float() # 将求导的内容设置为浮点数
x.requires_grad = True
y = x ** 2

torch.sum(y).backward()
print(x.grad)
