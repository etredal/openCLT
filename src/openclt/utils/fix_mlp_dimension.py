from transformers import GPT2Model

def check_mlp_dimensions():
    """Check the actual MLP dimensions in the GPT-2 model."""
    model = GPT2Model.from_pretrained('gpt2')
    
    # Check the actual MLP dimensions in the model
    print("Checking MLP dimensions in GPT-2 model:")
    
    # Check the first layer's MLP
    first_layer = model.h[0].mlp
    
    # GPT-2 uses Conv1D instead of Linear layers
    # Conv1D in transformers is a 1x1 convolution that's equivalent to a linear layer
    # with weight of shape (out_features, in_features)
    
    # Check c_fc (first part of MLP)
    c_fc_weight_shape = first_layer.c_fc.weight.shape
    print(f"c_fc weight shape: {c_fc_weight_shape}")
    
    # Check c_proj (second part of MLP)
    c_proj_weight_shape = first_layer.c_proj.weight.shape
    print(f"c_proj weight shape: {c_proj_weight_shape}")
    
    # The intermediate dimension is the output of c_fc (first dimension of weight)
    intermediate_dim = c_fc_weight_shape[0]
    
    # The input/output dimension is the input to c_fc / output of c_proj
    # (second dimension of weight)
    input_dim = c_fc_weight_shape[1]
    output_dim = c_proj_weight_shape[1]
    
    return {
        'input_dim': input_dim,
        'intermediate_dim': intermediate_dim,
        'output_dim': output_dim
    }

if __name__ == "__main__":
    dimensions = check_mlp_dimensions()
    print(f"\nFor GPT-2 Small, the MLP intermediate dimension is {dimensions['intermediate_dim']}")
    print(f"This is {dimensions['intermediate_dim'] / dimensions['input_dim']}x the hidden size")
