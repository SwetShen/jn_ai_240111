import torch

# 生成一个随机的图像数据
image = torch.normal(0, 0.5, (1, 3, 56, 56))


def max_pool_2d(image, kernel=2, stride=2):
    batch_size, channels, h, w = image.shape
    conv_h = (h - kernel) // stride + 1
    conv_w = (w - kernel) // stride + 1
    conv_result = torch.zeros((batch_size, channels, conv_h, conv_w))
    for i in range(conv_h):
        for j in range(conv_w):
            # print(conv_result[:, :, i, j].shape)
            # print(image[:, :, i * stride:i * stride + kernel, j * stride:j * stride + kernel].shape)
            # 最大池化，求区域中的最大值
            val = image[:, :, i * stride:i * stride + kernel, j * stride:j * stride + kernel]
            # torch.max 不可以向高纬度内容求解  torch.amax 根据切片维度求解最大值(降维)
            # max_val = torch.amax(val, dim=-1)
            # max_val = torch.amax(max_val,dim=-1)
            max_val = val.amax(dim=-1).amax(dim=-1)
            conv_result[:, :, i, j] = max_val
    return conv_result


def average_pool_2d(image, kernel=2, stride=2):
    batch_size, channels, h, w = image.shape
    conv_h = (h - kernel) // stride + 1
    conv_w = (w - kernel) // stride + 1
    conv_result = torch.zeros((batch_size, channels, conv_h, conv_w))
    for i in range(conv_h):
        for j in range(conv_w):
            val = image[:, :, i * stride:i * stride + kernel, j * stride:j * stride + kernel]
            mean_val = torch.mean(val.detach(), [2, 3])
            conv_result[:, :, i, j] = mean_val
    return conv_result


max_pool_2d(image, 2, 2)
average_pool_2d(image, 2, 2)
