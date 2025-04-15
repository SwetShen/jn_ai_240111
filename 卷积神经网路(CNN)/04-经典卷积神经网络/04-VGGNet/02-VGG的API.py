from torchvision.models import vgg11,vgg13,vgg16,vgg19
from torchvision.models import VGG11_Weights,VGG13_Weights,VGG16_Weights,VGG19_Weights

# 不加载权重，（从头训练）
# model = vgg11(weights=None)

# 加载权重模型 （可以直接使用该模型预测一些分类（必须包含在1000个分类中））
# model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)

