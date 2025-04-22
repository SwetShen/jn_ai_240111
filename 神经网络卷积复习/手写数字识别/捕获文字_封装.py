import cv2 as cv


# 获取图片中分割出来的 x1 x2 或 y1 y2 坐标
# axis: 取图片的哪个轴, 0: x轴, 1: y轴
# valid_threshold: 有效像素点阈值，小于该值的 xy 坐标点将被忽略
# continue_threshold: 连续像素点阈值，迭代每个像素点并寻找边框时，若点间隔小于该值，则认为是同一条边，不会被划分成多个边框
def get_xx(binary, axis, valid_threshold=5, continue_threshold=5):
    indices = cv.findNonZero(binary)
    statistics = {}
    for p in indices[:, 0]:
        x = p[axis].item()
        if x not in statistics:
            statistics[x] = 0
        statistics[x] += 1
    statistics = {k: v for k, v in statistics.items() if v > valid_threshold}
    sorted_x = sorted(statistics.keys())
    xx = []
    x1 = sorted_x[0]
    x2 = None
    last_x = x1
    for x in sorted_x:
        if x - last_x > continue_threshold:
            x2 = last_x
            xx.append([x1, x2])
            x1 = x
        last_x = x
    xx.append([x1, sorted_x[-1]])
    return xx


# 获取图片中分割出来的角坐标
def get_xyxy(binary):
    xx = get_xx(binary, 0)
    # 角坐标集合
    xyxy = []
    for p in xx:
        x1, x2 = p
        # 切图片
        _img = binary.copy()[:, x1:x2]
        y1, y2 = get_xx(_img, 1)[0]
        xyxy.append([x1, y1, x2, y2])
    return xyxy


if __name__ == '__main__':
    img = cv.imread('data/numbers.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    xyxy = get_xyxy(binary)

    for xy in xyxy:
        x1, y1, x2, y2 = xy
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow('img', img)
    cv.waitKey(0)
