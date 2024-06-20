import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from utilities import quantization_utils, logging_utils

class SimpleBERT(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleBERT, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased', return_dict=True)
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_ids, attention_mask):
        x = self.quant(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.fc(x)
        x = self.dequant(x)
        return x

# Example usage
if __name__ == "__main__":
    model = SimpleBERT()
    model.eval()
    quantized_model = quantization_utils.apply_quantization(model)
    logging_utils.register_hooks(quantized_model)
    dummy_input_ids = torch.randint(0, 1000, (1, 10))
    dummy_attention_mask = torch.ones((1, 10))
    output = quantized_model(dummy_input_ids, dummy_attention_mask)
