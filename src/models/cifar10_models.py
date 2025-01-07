import torchvision.models as models

from src.models.base_model import BaseModel


class CIFAR10ResNet(BaseModel):
    def __init__(self):
        super(CIFAR10ResNet, self).__init__()
        self.resnet = models.resnet18(num_classes=10)

    def forward(self, x):
        return self.resnet(x)

class CIFAR10VGG(BaseModel):
    def __init__(self):
        super(CIFAR10VGG, self).__init__()
        self.vgg = models.vgg16(num_classes=10)

    def forward(self, x):
        return self.vgg(x)
