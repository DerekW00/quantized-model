import unittest
import torch
from models.alexnet import AlexNet
from models.resnet import ResNet18
from models.transformer import Transformer
from models.rnn import RNN
from models.lstm import LSTM
from models.gpt1 import GPT1

class TestModels(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.num_classes = 10
        self.seq_length = 20
        self.input_dim = 512

    def test_alexnet(self):
        model = AlexNet(num_classes=self.num_classes)
        input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_classes))

    def test_resnet18(self):
        model = ResNet18(num_classes=self.num_classes)
        input_tensor = torch.randn(self.batch_size, 3, 224, 224)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_classes))

    def test_transformer(self):
        model = Transformer(input_dim=self.input_dim, num_classes=self.num_classes)
        input_tensor = torch.randint(0, 30522, (self.batch_size, self.seq_length))
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_classes))

    def test_rnn(self):
        model = RNN(input_dim=self.input_dim, num_classes=self.num_classes)
        input_tensor = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_classes))

    def test_lstm(self):
        model = LSTM(input_dim=self.input_dim, num_classes=self.num_classes)
        input_tensor = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_classes))

    def test_gpt1(self):
        model = GPT1(input_dim=self.input_dim, num_classes=self.num_classes)
        input_tensor = torch.randint(0, 50257, (self.batch_size, self.seq_length))
        output_tensor = model(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_classes))

if __name__ == '__main__':
    unittest.main()
