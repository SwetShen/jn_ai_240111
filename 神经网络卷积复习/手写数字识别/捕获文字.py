import cv2 as cv

# 有效像素的个数
valid_threshold = 5

img = cv.imread('data/numbers.jpg')
# 灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 二值化
threshold, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# 找非零值
# indices 代表非零值的索引，形状为 (N, C, 2)
# N 代表非零值的数量
# C 代表通道数
# 2 代表坐标: 第一个值是横坐标
indices = cv.findNonZero(binary)

# 统计结果
# key: 横坐标
# value: 代表数量
statistics = {}

# 统计每一列像素点个数
for p in indices[:, 0]:
    x = p[0].item()
    if x not in statistics:
        statistics[x] = 0
    # 统计横坐标对应列上出现了多少个像素点
    statistics[x] += 1

print(statistics)
# 过滤掉较少像素的无效坐标
statistics = {k: v for k, v in statistics.items() if v > valid_threshold}
# 排序
sorted_x = sorted(statistics.keys())
print(sorted_x)

# 像素连续的阈值，若大于该阈值则应该分割
continue_threshold = 5

# xx 用于保存 x1 x2 坐标
xx = []
x1 = sorted_x[0]
x2 = None
# 上一次遍历的 x
last_x = x1
for x in sorted_x:
    # 判断是否不连续，应该分割
    if x - last_x > continue_threshold:
        x2 = last_x
        xx.append([x1, x2])
        x1 = x
    last_x = x
# 追加最后一个坐标
xx.append([x1, sorted_x[-1]])
print(xx)
