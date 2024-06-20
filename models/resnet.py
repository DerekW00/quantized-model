import torch.nn as nn
import torch
import torch.quantization as quantization
from torchvision.models.resnet import BasicBlock, ResNet

class ResNet18(nn.Module):
    def __init__(self, layer=None, num_classes=10):
        super(ResNet18, self).__init__()
        if layer is None:
            layer = [2, 2, 2, 2]
        self.model = ResNet(BasicBlock, layers = layer, num_classes=num_classes)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
