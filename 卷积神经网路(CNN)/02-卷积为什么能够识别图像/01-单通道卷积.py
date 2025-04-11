import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./data/numbers/3/0.png", 0)

# =========== 设置图表 =============
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# =========== 设置卷积核 =============
kernel = np.float32([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

big_kernel = np.float32([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])


# =========== 设置卷积过程 =============
def conv(img, kernel, stride):
    h = (img.shape[0] - kernel.shape[0]) // stride + 1
    w = (img.shape[1] - kernel.shape[1]) // stride + 1
    k_h, k_w = kernel.shape
    conv_result = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            conv_result[i, j] = np.sum(img[i:k_h + i, j:k_w + j] * kernel)
    return conv_result


# =========== 执行卷积 =============
ax1.imshow(img, "gray")
ax1.set_title(f"{img.shape}")

# result = conv(img, kernel, 1)
# ax2.imshow(result, "gray")
# ax2.set_title(f"{result.shape}")

result = conv(img, big_kernel, 1)
ax2.imshow(result, "gray")
ax2.set_title(f"{result.shape}")

plt.show()
