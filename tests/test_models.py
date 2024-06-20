# test_models.py
import unittest
import torch
from models.alexnet import AlexNet
from models.resnet import ResNet18

class TestModelOutputs(unittest.TestCase):
    def test_alexnet_output(self):
        model = AlexNet()
        input = torch.randn(1, 3, 224, 224)
        output = model(input)
        self.assertEqual(output.shape, (1, 1000))

    def test_resnet18_output(self):
        model = ResNet18()
        input = torch.randn(1, 3, 224, 224)
        output = model(input)
        self.assertEqual(output.shape, (1, 1000))


if __name__ == '__main__':
    unittest.main()
