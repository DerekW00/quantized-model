import torch

def forward_hook(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Input: {input}")
    print(f"Output: {output}")
    if hasattr(module, 'weight'):
        print(f"Weight: {module.weight.data}")
    if hasattr(module, 'bias') and module.bias is not None:
        print(f"Bias: {module.bias.data}")

def register_hooks(model):
    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
