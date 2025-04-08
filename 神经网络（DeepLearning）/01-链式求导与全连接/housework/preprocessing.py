import pandas as pd
import numpy as np
import torch
from torch import nn

pd_data = pd.read_csv("SAheart.csv")
pd_data['famhist'] = pd_data['famhist'].map({"Absent": 0, "Present": 1})
data = pd_data.to_numpy(dtype=np.float32)

print(data)
print("===========================================")

h, w = data.shape
for i in range(w):
    column = data[:, i]
    data[:, i] = (column - column.mean()) / column.var()

print(data)

# dataset = torch.from_numpy(data)
#
# features = dataset[:, :-1]
# labels = dataset[:, -1]
#
# model = nn.Sequential(
#     nn.Linear(9, 16),
#     nn.Tanh(),
#     nn.Linear(16, 32),
#     nn.Tanh(),
#     nn.Linear(32, 64),
#     nn.Tanh(),
#     nn.Linear(64, 2),
#     nn.LogSoftmax(dim=-1)
# )
#
# criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# epochs = 10000
# for i in range(epochs):
#     optimizer.zero_grad()
#     predict = model(features)
#     loss = criterion(predict, labels)
#     loss.backward()
#     print(f"loss:{loss.item():.4f}")
#     optimizer.step()
