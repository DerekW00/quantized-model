import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class Transformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Transformer, self).__init__()
        config = BertConfig.from_pretrained('bert-base-uncased', return_dict=True)
        self.bert = BertModel(config)
        self.fc = nn.Linear(config.hidden_size, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, input_ids):
        x = self.quant(input_ids)
        outputs = self.bert(input_ids)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.fc(x)
        x = self.dequant(x)
        return x
