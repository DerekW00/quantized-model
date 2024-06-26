import copy

import torch
import torchvision.transforms as transforms
from PIL import Image


# Helper function to clone models
def clone_model(model):
    return copy.deepcopy(model)


def preprocess_and_infer(image_path, model, device):
    # Define the appropriate transforms for ImageNet-pretrained models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the image
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
    img_t = transform(img)
    batch_t = img_t.unsqueeze(0).to(device)

    # Inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(batch_t)

    return outputs


def preprocess_and_infer_text(text, model, tokenizer, device):
    # Tokenization and encoding the text
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs


# Function to quantize tensors
def quantize_tensor(tensor, bits: int):
    qmin = -2. ** (bits - 1)
    qmax = 2. ** (bits - 1) - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = int((initial_zero_point).round().item())  # Using `.round().item()` to convert to Python int
    zero_point = max(qmin, min(qmax, zero_point))
    quantized = zero_point + tensor / scale
    quantized.clamp_(qmin, qmax).round_()  # In-place clamping and rounding
    return quantized, scale, zero_point


# Function to apply quantization to a model
def apply_quantization(model, weight_bits: int, bias_bits: int):
    cloned_model = clone_model(model)
    for name, param in cloned_model.named_parameters():
        if 'weight' in name:
            quantized_tensor, scale, zero_point = quantize_tensor(param.data, weight_bits)
            param.data = quantized_tensor
        elif 'bias' in name:
            quantized_tensor, scale, zero_point = quantize_tensor(param.data, bias_bits)
            param.data = quantized_tensor
    return cloned_model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No need to track gradients
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy


def compare_models(original_model, quantized_model, dataloader, criterion, device):
    original_loss, original_accuracy = evaluate_model(original_model, dataloader, criterion, device)
    quantized_loss, quantized_accuracy = evaluate_model(quantized_model, dataloader, criterion, device)

    print(f"Original Model - Loss: {original_loss:.4f}, Accuracy: {original_accuracy:.4f}")
    print(f"Quantized Model - Loss: {quantized_loss:.4f}, Accuracy: {quantized_accuracy:.4f}")

