import torch
import torch.quantization as quant
from config import QUANT_ENGINE

def prepare_model_for_quantization(model):
    torch.backends.quantized.engine = QUANT_ENGINE
    model.qconfig = quant.get_default_qconfig(QUANT_ENGINE)
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    quant.prepare(model, inplace=True)
    return model

def calibrate_model(model, calibration_data):
    model.eval()
    with torch.no_grad():
        for inputs, _ in calibration_data:
            model(inputs)
    return model

def convert_model_to_quantized(model):
    torch.backends.quantized.engine = QUANT_ENGINE
    quantized_model = quant.convert(model, inplace=False)
    return quantized_model
