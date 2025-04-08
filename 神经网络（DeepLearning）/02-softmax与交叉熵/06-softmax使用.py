import torch
from torch import nn

input = torch.tensor([
    [2, 3, -2],  # 三种分类的得分
    [1.3, 3.4, 2.2],
    [7, -10, 5]
]).float()

# ===========框架下的softmax=======
softmax1 = nn.Softmax(dim=-1)

output1 = softmax1(input)
print(output1)


# ===========自定义softmax=======
def softmax2(input, dim=1):
    tmp = torch.exp(input)  # (3,3)
    # torch.sum(tmp, dim=dim).unsqueeze(-1) # (3,1) 此处才可以使用广播机制进行相除
    return tmp / torch.sum(tmp, dim=dim).unsqueeze(-1)


output2 = softmax2(input)
print(output2)
