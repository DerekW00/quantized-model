import unittest
import torch
import torch.nn as nn
from utilities.hooks import register_hooks, remove_hooks
from models.alexnet import AlexNet
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class TestHooks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = SimpleModel()
        cls.data = torch.randn(2, 10)

    def test_forward_hook(self):
        hooks = register_hooks(self.model)

        # Redirect stdout to capture print statements
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Perform a forward pass
        _ = self.model(self.data)

        # Reset redirect.
        sys.stdout = sys.__stdout__

        # Verify the captured output
        output = captured_output.getvalue()
        self.assertIn("Layer: Linear", output)
        self.assertIn("Input:", output)
        self.assertIn("Output:", output)
        self.assertIn("Weight:", output)
        self.assertIn("Bias:", output)

        remove_hooks(hooks)
class TestAlexNetHooks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = AlexNet()
        cls.data = torch.randn(2, 3, 224, 224)  # Assuming input size for AlexNet

    def test_forward_hook(self):
        hooks = register_hooks(self.model)

        # Redirect stdout to capture print statements
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Perform a forward pass
        _ = self.model(self.data)

        # Reset redirect.
        sys.stdout = sys.__stdout__

        # Verify the captured output
        output = captured_output.getvalue()
        #self.assertIn("Layer: Conv2d", output)
        #self.assertIn("Layer: Linear", output)
        self.assertIn("Input:", output)
        self.assertIn("Output:", output)
        #self.assertIn("Weight:", output)
        #self.assertIn("Bias:", output)

        remove_hooks(hooks)

if __name__ == '__main__':
    unittest.main()
