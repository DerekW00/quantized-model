from torchvision import models
from quantization import *

# The following code snippet demonstrates how to compare the weights of a quantized model with the original model.


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load models
alexnet_model = models.alexnet(pretrained=True).to(device)
quantized_model = apply_quantization(alexnet_model, 8, 32)

for name, param in alexnet_model.named_parameters():
    if 'weight' in name:
        # Check if the corresponding quantized parameter exists
        quantized_param = dict(quantized_model.named_parameters()).get(name)
        if quantized_param is not None:
            # Print the differences or visualize them
            print(f'Layer: {name}')
            print('Original weights:')
            print(param.data)
            print('Quantized weights:')
            print(quantized_param.data)
            print('Difference:')
            print(param.data - quantized_param.data)
            print('-' * 50)
#
