import cv2
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

image = cv2.imread("./img/car.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(image).permute([2, 0, 1])

# 数据增强不会让数据持久化，也就是说每一轮epoch下的数据变换是随机的
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # 将图像的所有边缩减到227x227
    # transforms.Resize(227),  # 将最短边设置为227
    # transforms.CenterCrop(227), # 在图像的中央位置截取227的大小
    # transforms.RandomVerticalFlip(),  # 随机的垂直翻转
    # transforms.RandomRotation((-5,5)), # 随机（-5° ~ 5°）旋转
    transforms.RandomHorizontalFlip() # 随机的水平翻转
])

result = transform(image)

ax1.imshow(image.permute([1, 2, 0]).numpy())
ax2.imshow(result.permute([1, 2, 0]).numpy())
plt.show()
