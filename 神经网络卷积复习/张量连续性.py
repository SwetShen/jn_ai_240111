# 张量连续性: 指的是逻辑的连续性，不是物理内存的连续性，因为内存一定是连续的
# 总结:
# 连续性指的是张量排列顺序是否和内存顺序一致
# 连续的张量优点 算得快
# view vs reshape
# 1. view reshape 和 连续性无关
# 2. view 是视图，只会改变你看到的样子，不会改变内存
# 3. reshape 是创建一个新张量，分配新的内存空间

import torch

# t = torch.arange(12)
# print(t)
# # is_contiguous 用于判断逻辑连续性是否连续
# print(t.is_contiguous())
# t = t.reshape(2, 2, 3)
# print(t)
# print(t.is_contiguous())
# t = t.permute(2, 0, 1)
# print(t)
# print(t.is_contiguous())
# # contiguous 的作用，强制将内存顺序按照当前张量的逻辑顺序进行排列
# t = t.contiguous()
# print(t.is_contiguous())


t = torch.arange(12)
t1 = t.view(3, 4)
print(t1.is_contiguous())
t2 = t.reshape(3, 4)
print(t2.is_contiguous())
