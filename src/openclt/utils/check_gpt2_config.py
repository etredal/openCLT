from transformers import GPT2Model

def main():
    print("Loading GPT-2 model...")
    model = GPT2Model.from_pretrained('gpt2')
    
    print("\nConfig attributes:")
    print(dir(model.config))
    
    print("\nConfig values:")
    for attr in dir(model.config):
        if not attr.startswith('_') and not callable(getattr(model.config, attr)):
            print(f'{attr}: {getattr(model.config, attr)}')
    
    # Check MLP dimensions specifically
    print("\nChecking MLP dimensions:")
    print(f"hidden_size (n_embd): {model.config.n_embd}")
    print(f"n_inner: {getattr(model.config, 'n_inner', None)}")
    
    # Check the actual MLP dimensions in the model
    print("\nActual MLP dimensions from model:")
    for i in range(model.config.n_layer):
        mlp_dim = model.h[i].mlp.c_fc.weight.shape[0]
        print(f"Layer {i} MLP dimension: {mlp_dim}")

if __name__ == "__main__":
    main()
