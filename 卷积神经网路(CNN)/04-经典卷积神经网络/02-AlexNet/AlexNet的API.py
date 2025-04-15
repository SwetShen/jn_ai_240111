# torchvision.models 导入内容：小写为模型，大写一般为权重
from torchvision.models import alexnet, AlexNet_Weights

# 加载模型结构，权重参数随机（从头训练）
# model = alexnet(weights=None)

# 加载带有权重的模型 （微调 --> finetune  站在巨人的肩膀上）
#      微调，基于已经训练好的模型的基础上，再训练自身的模型（加快训练速度，以及提高模型泛化性）
# AlexNet_Weights.IMAGENET1K_V1 ,这个权重是已经在ImageNet（1000个分类的数据集）
# 训练好的权重.
model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
