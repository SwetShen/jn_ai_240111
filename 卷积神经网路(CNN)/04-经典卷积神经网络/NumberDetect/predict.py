import torch
from backbones.lenet import LetNet
import cv2

model = LetNet(10)
model.load_state_dict(torch.load("./save/best.pt", weights_only=True))

image = cv2.imread("./data/test/3.png")
image = cv2.resize(image, (32, 32))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,32,32)

model.eval()
result = model(image.float())
print(torch.argmax(result, dim=-1))
