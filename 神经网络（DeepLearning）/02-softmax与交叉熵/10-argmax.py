import torch

a = torch.tensor([
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.8, 0.1, 0.1]
])

# argmax 求行或者列中最大值的下标
b = torch.argmax(a, dim=1)
print(b)
c = torch.argmax(a, dim=0)
print(c)
