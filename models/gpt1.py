import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from utilities import quantization_utils, logging_utils

class SimpleGPT1(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleGPT1, self).__init__()
        config = GPT2Config.from_pretrained('gpt2', return_dict=True)
        self.gpt = GPT2Model(config)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_ids):
        x = self.quant(input_ids)
        outputs = self.gpt(input_ids)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.fc(x)
        x = self.dequant(x)
        return x

# Example usage
if __name__ == "__main__":
    model = SimpleGPT1()
    model.eval()
    quantized_model = quantization_utils.apply_quantization(model)
    logging_utils.register_hooks(quantized_model)
    dummy_input_ids = torch.randint(0, 1000, (1, 10))
    output = quantized_model(dummy_input_ids)
