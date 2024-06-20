import torch
import torch.backends.quantized

print("PyTorch version:", torch.__version__)
print("Current quantization engine:", torch.backends.quantized.engine)
print("Available quantization backends:", torch.backends.quantized.supported_engines)

# Set the quantization engine to 'qnnpack' or 'fbgemm'
torch.backends.quantized.engine = 'qnnpack'
print("Set quantization engine to 'qnnpack'")
print("Quantization engine:", torch.backends.quantized.engine)
