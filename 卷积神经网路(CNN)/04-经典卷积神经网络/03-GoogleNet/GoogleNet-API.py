from torchvision.models import googlenet
from torchsummary import summary

model = googlenet(weights=None)
summary(model, (3, 224, 224), device="cpu")
