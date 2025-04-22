import cv2 as cv
import torch

from 捕获文字_封装 import get_xyxy
from torchvision.transforms import ToTensor
from PIL import Image

to_tensor = ToTensor()


def preprocess(img_path, target_size=28, edge_pad_px=16):
    images = []
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    xyxy = get_xyxy(binary)
    for x1, y1, x2, y2 in xyxy:
        _img = binary.copy()[y1:y2, x1:x2]
        h, w = _img.shape
        if h != w:
            left_right_pad = h > w
            pad_px = (h - w if left_right_pad else w - h) / 2
            perfect_div = pad_px % 1 == 0
            pad1 = int(pad_px)
            pad2 = pad1 + 1 if not perfect_div else pad1
            pad_value = (0, 0, pad1, pad2) if left_right_pad else (pad1, pad2, 0, 0)
            _img = cv.copyMakeBorder(
                _img,
                *pad_value,
                borderType=cv.BORDER_CONSTANT,
                value=(0,)
            )
        _img = cv.copyMakeBorder(
            _img,
            *(edge_pad_px, edge_pad_px, edge_pad_px, edge_pad_px),
            borderType=cv.BORDER_CONSTANT,
            value=(0,)
        )
        # cv.imshow('i', _img)
        # cv.waitKey()
        _img = cv.resize(_img, (target_size, target_size))
        image = Image.fromarray(_img)
        image = to_tensor(image)
        images.append(image)
    images = torch.stack(images, dim=0)
    return images


if __name__ == '__main__':
    images = preprocess('data/numbers.jpg')
    print(images.shape)
