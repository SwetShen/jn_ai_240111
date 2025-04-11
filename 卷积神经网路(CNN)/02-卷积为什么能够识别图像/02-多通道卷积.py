import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./data/numbers/3/0.png")  # (28 x 28 x 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========== 设置图表 =============
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# =========== 设置卷积核 =============
kernel = np.random.normal(0, 0.5, (3, 3, 6))


# =========== 卷积过程 =============
def conv(img, kernel, stride):
    h = (img.shape[0] - kernel.shape[0]) // stride + 1
    w = (img.shape[1] - kernel.shape[1]) // stride + 1
    c = img.shape[-1]
    k_h, k_w, k_c = kernel.shape
    final = 0
    for k in range(c):
        conv_result = np.zeros((h, w, k_c))
        for i in range(h):
            for j in range(w):
                channel_img = img[:, :, k, None]
                conv_result[i, j] = np.sum(channel_img[i:k_h + i, j:k_w + j] * kernel)
        final += conv_result

    return final


# =========== 执行卷积 =============
ax1.imshow(img)
ax1.set_title(f"{img.shape}")

result = conv(img, kernel, 1)  # (26,26,6)

ax2.imshow(result[:, :, 0], "gray")
ax2.set_title(f"{result.shape}")

plt.show()
