from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchsummary import summary

model = mobilenet_v2(weights=None)

summary(model, (3, 224, 224), device="cpu")
