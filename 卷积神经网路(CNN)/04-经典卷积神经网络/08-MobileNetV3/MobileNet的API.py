from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from torchsummary import summary

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
summary(model, (3, 224, 224), device="cpu")
