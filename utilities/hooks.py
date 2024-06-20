def register_hooks(model):
    def hook_fn(module, input, output):
        print(f"Layer: {module}")
        print(f"Input: {input}")
        print(f"Output: {output}")
        if hasattr(module, 'weight'):
            print(f"Weight: {module.weight}")
        if hasattr(module, 'bias'):
            print(f"Bias: {module.bias}")

    for layer in model.children():
        layer.register_forward_hook(hook_fn)
