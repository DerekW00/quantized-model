import torch
import torch.nn as nn
from utilities import quantization_utils, logging_utils

class SimpleRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=1, num_classes=10):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Taking last output
        x = self.dequant(x)
        return x

# Example usage
if __name__ == "__main__":
    model = SimpleRNN()
    model.eval()
    quantized_model = quantization_utils.apply_quantization(model)
    logging_utils.register_hooks(quantized_model)
    dummy_input = torch.randn(1, 5, 10)  # (batch, sequence, feature)
    output = quantized_model(dummy_input)
