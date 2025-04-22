import cv2 as cv
import torch

from 捕获文字_封装 import get_xyxy
from torchvision.transforms import ToTensor
from PIL import Image

to_tensor = ToTensor()

# 边缘填充
edge_pad_px = 8

# 所有图片的张量
images = []

img = cv.imread('data/numbers.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
threshold, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
xyxy = get_xyxy(binary)
for x1, y1, x2, y2 in xyxy:
    # 切图
    _img = binary.copy()[y1:y2, x1:x2]
    # 获取图片宽高
    h, w = _img.shape
    # 填充图片为正方形
    if h != w:
        # 判断是否左右填充
        left_right_pad = h > w
        # 填充的像素值
        pad_px = (h - w if left_right_pad else w - h) / 2
        # 是否整除
        perfect_div = pad_px % 1 == 0
        pad1 = int(pad_px)
        pad2 = pad1 + 1 if not perfect_div else pad1
        # 填充值 （顺序为 上下左右）
        pad_value = (0, 0, pad1, pad2) if left_right_pad else (pad1, pad2, 0, 0)

        # 填充
        _img = cv.copyMakeBorder(
            _img,
            *pad_value,
            borderType=cv.BORDER_CONSTANT,
            value=(0,)
        )
    # 填充边界
    _img = cv.copyMakeBorder(
        _img,
        *(edge_pad_px, edge_pad_px, edge_pad_px, edge_pad_px),
        borderType=cv.BORDER_CONSTANT,
        value=(0,)
    )
    # 重置大小为 28x28
    _img = cv.resize(_img, (28, 28))
    image = Image.fromarray(_img)
    image = to_tensor(image)
    images.append(image)

# 将列表转换成张量
# torch.stack 堆叠
# 堆叠操作将有两个行为
# 1. 增加一个维度
# 2. 指定增加哪个维度
images = torch.stack(images, dim=0)
