# train_gpt1.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.gpt1 import GPT1
from utilities.quantization import prepare_model_for_quantization, calibrate_model, convert_model_to_quantized

# Hyperparameters
num_epochs = 3
learning_rate = 1e-4
batch_size = 16
num_classes = 2

# Load dataset
dataset = load_dataset('imdb')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
train_dataset = dataset['train'].shuffle().select(range(2000))
test_dataset = dataset['test'].shuffle().select(range(500))

def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    return torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True), labels

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Initialize model
model = GPT1(num_classes=num_classes).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for input_ids, labels in train_loader:
        input_ids, labels = input_ids.to('cuda'), labels.to('cuda')
        outputs = model(input_ids)
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
    for input_ids, labels in test_loader:
        input_ids, labels = input_ids.to('cuda'), labels.to('cuda')
        outputs = quantized_model(input_ids)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the quantized model on the test set: {100 * correct / total:.2f}%')

# Benchmark
benchmark_accuracy = 85.0  # Example benchmark for the quantized GPT1 model on IMDB
assert (100 * correct / total) >= benchmark_accuracy, "Benchmark not met!"
