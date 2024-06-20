import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = self.dequant(x)
        return x
