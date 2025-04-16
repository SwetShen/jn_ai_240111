import torch
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchsummary import summary

model = mobilenet_v3_large(weights=None)

# 预训练模型 ：已经在数据集中训练好的模型 IMAGENET1K_V1: 在ImageNet数据集中训练的权重
# model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1) # 加载预训练模型

device = torch.device("cuda")
model = model.to(device=device)

summary(model, (3, 224, 224), device="cuda")
