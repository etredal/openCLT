from transformers import GPT2Model
import torch

def check_mlp_activations():
    """Check the actual MLP activations in the GPT-2 model."""
    model = GPT2Model.from_pretrained('gpt2')
    
    # Create a simple input
    input_ids = torch.tensor([[464, 3290, 318, 617, 836]])  # Random token IDs
    
    # Register a hook to capture MLP activations
    activations = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output
            print(f"Layer {layer_idx} MLP output shape: {output.shape}")
            print(f"Layer {layer_idx} MLP input shape: {input[0].shape}")
            return output
        return hook
    
    # Register hooks for each layer's MLP
    hooks = []
    for i in range(model.config.n_layer):
        hook = model.h[i].mlp.register_forward_hook(hook_fn(i))
        hooks.append(hook)
    
    # Forward pass
    outputs = model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Check the first layer's MLP structure
    first_layer_mlp = model.h[0].mlp
    print("\nMLP Structure:")
    print(f"c_fc weight shape: {first_layer_mlp.c_fc.weight.shape}")
    print(f"c_proj weight shape: {first_layer_mlp.c_proj.weight.shape}")
    
    return activations

if __name__ == "__main__":
    activations = check_mlp_activations()
