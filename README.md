# Quantized Model

This repository hosts the code and resources for Quantized Model, a project focused on optimizing machine learning models through quantization techniques. Quantization reduces the computational requirements of models, making them more efficient for deployment on edge devices with limited hardware resources.

## Project Overview

The goal of this project is to experiment with various quantization techniques to compress and optimize machine learning models while maintaining accuracy. This process makes it easier to deploy these models on low-power devices or systems with limited computational capacity.

The project explores:

- Post-training quantization and quantization-aware training.
- Model size reduction through parameter quantization.
- Performance benchmarks comparing quantized and non-quantized models.

## Features

- **Quantization Techniques**: Implements common quantization strategies including 8-bit and dynamic range quantization.
- **Performance Comparison**: Tools to measure and compare the accuracy, inference speed, and size of models before and after quantization.
- **Deployment-Friendly**: Optimized models are suitable for deployment on edge devices like mobile phones, Raspberry Pi, and embedded systems.

## Getting Started

### Prerequisites

To use the project, you’ll need the following installed:

- Python 3.x
- TensorFlow, PyTorch, or the relevant ML framework
- Numpy

You can install the dependencies using:

```sh
pip install -r requirements.txt
```

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/DerekW00/quantized-model.git
    cd quantized-model
    ```

2. Install required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

3. Run the quantization script (replace with the specific script name):

    ```sh
    python quantize_model.py --model model.h5 --output quantized_model.tflite
    ```

## Usage

### Model Quantization

The main functionality of the repository is to quantize models. Use the `quantize_model.py` script to apply quantization:

```sh
python quantize_model.py --model <model_path> --output <output_path> --method <quantization_method>
```

**Parameters:**

- `--model`: Path to the original model.
- `--output`: Output path for the quantized model.
- `--method`: Choose the quantization method (post_training, quantization_aware).

### Benchmarking Quantized Models

To compare the performance of the quantized model to the original:

```sh
python benchmark.py --model <quantized_model_path>
```

This will output the inference time, model size, and accuracy metrics.

## Examples

**Example 1: Post-Training Quantization**

```sh
python quantize_model.py --model ./models/mnist.h5 --output ./models/mnist_quantized.tflite --method post_training
```

**Example 2: Quantization-Aware Training**

```sh
python quantize_model.py --model ./models/cifar10.h5 --output ./models/cifar10_quant_aware.tflite --method quantization_aware
```

## Results

- **Model Size**: Quantized models show a significant reduction in size (up to 4x smaller).
- **Inference Speed**: Quantized models perform inference faster on edge devices.
- **Accuracy Impact**: Quantization introduces minimal accuracy loss while optimizing performance.

## Issues and Limitations

- Some models may experience more accuracy degradation than others, especially when using aggressive quantization strategies.
- Compatibility varies depending on the model architecture and framework.

## Future Work

- Explore further optimizations for convolutional neural networks (CNNs).
- Integrate additional frameworks (e.g., ONNX).
- Experiment with mixed precision and hardware-specific quantization optimizations.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes or improvements.

## License

This project is licensed under the MIT License.

## References

- [TensorFlow Quantization Documentation](https://www.tensorflow.org/model_optimization/guide/quantization)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
