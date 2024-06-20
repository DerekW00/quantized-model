# train_transformer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from models.transformer import Transformer
from utilities.quantization import prepare_model_for_quantization, calibrate_model, convert_model_to_quantized

# Hyperparameters
num_epochs = 5
learning_rate = 1e-3
batch_size = 16
input_dim = 100
num_classes = 2

# Load dataset
tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = IMDB()

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
    return vocab(tokenizer(x))

def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(int(_label == 'pos'))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    labels = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return text_list, labels

train_dataset = list(IMDB(split='train'))
test_dataset = list(IMDB(split='test'))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# Initialize model
model = Transformer(input_dim=input_dim, num_classes=num_classes).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for texts, labels in train_loader:
        texts, labels = texts.to('cuda'), labels.to('cuda')
        outputs = model(texts)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Quantization
model.cpu()
prepared_model = prepare_model_for_quantization(model)
calibrated_model = calibrate_model(prepared_model, train_loader)
quantized_model = convert_model_to_quantized(calibrated_model)

# Evaluation
quantized_model.to('cuda')
quantized_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to('cuda'), labels.to('cuda')
        outputs = quantized_model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the quantized model on the test set: {100 * correct / total:.2f}%')

# Benchmark
benchmark_accuracy = 85.0  # Example benchmark for the quantized Transformer model on IMDB
assert (100 * correct / total) >= benchmark_accuracy, "Benchmark not met!"
