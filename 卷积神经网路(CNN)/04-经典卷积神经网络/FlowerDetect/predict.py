import torch
from backbones.AlexNet import AlexNet
import cv2
import numpy as np
from datasets.dataloader import _generate_dict

device = torch.device("cpu")
model = AlexNet(5, device=device)
model.load_state_dict(torch.load("./save/best.pt", weights_only=True))

image = cv2.imread("./data/flower_photos/roses/14683774134_6367640585.jpg")
image = cv2.resize(image, (227, 227))
image = np.expand_dims(image, 0)  # (1,227,227,3)
image = torch.from_numpy(image).permute([0, 3, 1, 2])  # (1,3,227,227)

model.eval()
result = model(image.float())
print(torch.argmax(result, dim=-1).item())
dict = _generate_dict()
print(dict[str(torch.argmax(result, dim=-1).item())])