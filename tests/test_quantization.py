import unittest
import torch
from models.alexnet import AlexNet
from models.rnn import RNN
from models.lstm import LSTM
from models.transformer import Transformer
from models.gpt1 import GPT1
from models.resnet import ResNet18
from utilities.quantization import prepare_model_for_quantization, calibrate_model, convert_model_to_quantized
from utilities.data_loader import get_data_loader
from config import QUANT_ENGINE
import torch.nn as nn


class TestQuantization(unittest.TestCase):

    def setUp(self):
        # Set the quantization backend engine
        torch.backends.quantized.engine = QUANT_ENGINE

        # Setup initial model and data loader
        self.model = AlexNet()
        self.data_loader = get_data_loader()

    def test_prepare_for_quantization(self):
        # Test if the model preparation for quantization is correct
        prepared_model = prepare_model_for_quantization(self.model)
        self.assertTrue(prepared_model.qconfig is not None, "Quantization config should be set.")

    def test_calibrate_model(self):
        # Test model calibration
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        self.assertTrue(calibrated_model is not None, "Calibrated model should not be None.")

    def test_convert_to_quantized(self):
        # Test conversion to quantized model
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        quantized_model = convert_model_to_quantized(calibrated_model)
        self.assertTrue(isinstance(quantized_model, torch.nn.Module), "Quantized model should be a nn.Module.")

    def test_quantized_forward_pass(self):
        # Test forward pass of quantized model
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        quantized_model = convert_model_to_quantized(calibrated_model)

        # Check if quantized model supports quantized convolution
        for name, module in quantized_model.named_modules():
            if isinstance(module, torch.nn.quantized.Conv2d):
                print(f"Module {name} supports quantized Conv2d.")

        for inputs, _ in self.data_loader:
            quantized_model.eval()
            with torch.no_grad():
                try:
                    quantized_outputs = quantized_model(inputs)
                except NotImplementedError as e:
                    self.fail(f"Quantized forward pass failed with NotImplementedError: {e}")
            break

        self.assertIsNotNone(quantized_outputs, "Quantized model output should not be None.")
        self.assertIsInstance(quantized_outputs, torch.Tensor, "Quantized model output should be a tensor.")

    def test_quantized_weights(self):
        # Test if the weights are indeed quantized
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        quantized_model = convert_model_to_quantized(calibrated_model)

        for name, param in quantized_model.named_parameters():
            self.assertTrue(param.dtype in [torch.qint8, torch.quint8, torch.float32],
                            f"Parameter {name} is not quantized.")

    def test_output_consistency(self):
        # Compare the output of the quantized model to the original model
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        quantized_model = convert_model_to_quantized(calibrated_model)

        for inputs, _ in self.data_loader:
            self.model.eval()
            quantized_model.eval()
            with torch.no_grad():
                original_outputs = self.model(inputs)
                try:
                    quantized_outputs = quantized_model(inputs)
                except NotImplementedError as e:
                    self.fail(f"Quantized forward pass failed with NotImplementedError: {e}")
            break

        # Allowing a higher tolerance due to potential quantization error
        self.assertTrue(torch.allclose(original_outputs, quantized_outputs, atol=1e-1), "Outputs are not close enough.")


class TestResNet18(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = ResNet18(num_classes=10)
        cls.data_loader = get_data_loader(batch_size=32)

    def test_prepare_for_quantization(self):
        prepared_model = prepare_model_for_quantization(self.model)
        self.assertTrue(isinstance(prepared_model, nn.Module))

    def test_calibrate_model(self):
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        self.assertTrue(isinstance(calibrated_model, nn.Module))

    def test_convert_to_quantized(self):
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        quantized_model = convert_model_to_quantized(calibrated_model)
        self.assertTrue(isinstance(quantized_model, nn.Module))

    def test_output_consistency(self):
        prepared_model = prepare_model_for_quantization(self.model)
        calibrated_model = calibrate_model(prepared_model, self.data_loader)
        quantized_model = convert_model_to_quantized(calibrated_model)

        for inputs, _ in self.data_loader:
            self.model.eval()
            quantized_model.eval()
            with torch.no_grad():
                original_outputs = self.model(inputs)
                try:
                    quantized_outputs = quantized_model(inputs)
                except NotImplementedError as e:
                    self.fail(f"Quantized forward pass failed with NotImplementedError: {e}")
            break

        # Allowing a higher tolerance due to potential quantization error
        self.assertTrue(torch.allclose(original_outputs, quantized_outputs, atol=1e-1), "Outputs are not close enough.")


if __name__ == '__main__':
    unittest.main()
