import torch
from backbones.AlexNet import AlexNet
import cv2
import numpy as np
from datasets.dataloader import _generate_dict
from torchvision.transforms import transforms
from PIL import Image

device = torch.device("cpu")
model = AlexNet(5, device=device)
model.load_state_dict(torch.load("./save/best.pt", weights_only=True))

image = cv2.imread("./data/flower_photos/roses/14683774134_6367640585.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (227,227,3)
image = Image.fromarray(image)  # 将opencv numpy类型转化为PIL类型 (3,227,227)

# 注意：当时数据增强的时候，加工图片的方式，必须在预测图像的位置，也需要对应的加工。
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),  # PIL 转化为torch.tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = transform(image).unsqueeze(0)  # (1,3,227,227)

model.eval()
result = model(image.float())
print(torch.argmax(result, dim=-1).item())
dict = _generate_dict()
print(dict[str(torch.argmax(result, dim=-1).item())])
