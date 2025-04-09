import torch
from torchvision.models import alexnet

"""
官方文档：
https://pytorch.org/docs/stable/optim.html#module-torch.optim
"""
model = alexnet(weights=None)

optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9)
"""
params (iterable) –参数的 iterable 或named_parameters进行优化 或定义参数组的 dict 的 iterable 。使用 named_parameters 时， 所有组中的所有参数都应命名为
lr (float, Tensor, optional) – 学习率 (default: 1e-3)
momentum (float, optional) –动量因子 (default: 0)
dampening (float, optional) – 动量阻尼 (default: 0)
weight_decay (float, optional) – 权重衰减（L2 惩罚）(default: 0)
nesterov (bool, optional) –启用 Nesterov momentum.仅适用于 当动量不为零时。（默认值：False）
"""

torch.optim.Adagrad(model.parameters(),lr=0.01)
"""
params (iterable) –参数的 iterable 或named_parameters进行优化 或定义参数组的 dict 的 iterable 。使用 named_parameters 时， 所有组中的所有参数都应命名为
lr (float, Tensor, optional) –学习率 (default: 1e-2)
lr_decay (float, optional) – 学习率衰减 (default: 0)
weight_decay (float, optional) – 权重衰减（L2 惩罚） (default: 0)
initial_accumulator_value (float, optional) – 初始值 梯度的平方和 (default: 0)
eps (float, optional) – term 添加到分母中以改善 数值稳定性(default: 1e-10)
"""

torch.optim.RMSprop(model.parameters(),lr=0.01)
"""
params (iterable) – 参数的 iterable 或named_parameters进行优化 或定义参数组的 dict 的 iterable 。使用 named_parameters 时， 所有组中的所有参数都应命名为
lr (float, Tensor, optional) –学习率 (default: 1e-2)
alpha (float, optional) – 平滑常数 (default: 0.99)
eps (float, optional) – term 添加到分母中以改善 数值稳定性 (default: 1e-8)
weight_decay (float, optional) – 权重衰减（L2 惩罚） (default: 0)
momentum (float, optional) – 动量因子(default: 0)
centered (bool, optional) – 如果 ，计算居中的 RMSProp， 梯度通过其方差的估计进行归一化True
"""

torch.optim.Adam(model.parameters(),lr=0.01)
"""
params (iterable) –参数的 iterable 或named_parameters进行优化 或定义参数组的 dict 的 iterable 。使用 named_parameters 时， 所有组中的所有参数都应命名为
lr (float, Tensor, optional) – 学习率 （默认值：1e-3）。张量 LR 尚不支持我们的所有实现。请使用浮点数 LR（如果您未同时指定 fused=True 或 capturable=True）.
betas (Tuple[float, float], optional) – 用于计算的系数 梯度及其平方的移动平均值 (default: (0.9, 0.999))
eps (float, optional) – term 添加到分母中以改善 数值稳定性 (default: 1e-8)
weight_decay (float, optional) –权重衰减 （L2 惩罚） （默认值：0）
"""