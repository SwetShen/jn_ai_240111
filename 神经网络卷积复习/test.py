from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn as nn

img = Image.open('./Lenna.png')
x = ToTensor()(img)
print(x.shape)

x = nn.MaxPool2d(3, padding=1, stride=1)(x)
print(x.shape)
img = ToPILImage()(x)
img.show()
