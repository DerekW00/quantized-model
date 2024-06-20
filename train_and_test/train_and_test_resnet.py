# train_resnet18.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from utilities.quantization import prepare_model_for_quantization, calibrate_model, convert_model_to_quantized

# Hyperparameters
num_epochs = 10
learning_rate = 1e-3
batch_size = 32
num_classes = 10

# Load dataset
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = ResNet18(num_classes=num_classes).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
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
    for images, labels in test_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the quantized model on the test set: {100 * correct / total:.2f}%')

# Benchmark
benchmark_accuracy = 90.0  # Example benchmark for the quantized ResNet18 model on CIFAR-10
assert (100 * correct / total) >= benchmark_accuracy, "Benchmark not met!"
