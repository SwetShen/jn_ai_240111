import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./imgs/test.jpg")  # BGR 彩色
print(img.shape)

# blue_channel = img[:, :, 0]
# cv2.imshow("content", blue_channel)  # 单通道（黑白图）
# cv2.waitKey(0)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[:, :, ::-1]
plt.imshow(img)  # RGB
plt.show()
