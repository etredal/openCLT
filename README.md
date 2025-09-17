# Open Cross-Layer Transcoder

Note: This project is not fully functional yet.  With small~medium edits it should be, depending on your use case.

This project implements a cross-layer transcoder originally based on the paper [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).  This is an open source project free to use.  Cross Layer Transcoders (CLT) have become very useful for Mechanistic Interpretability research -- that is, the study of how an AI model works.  We are hoping in the long run that we can model the architecture of AI in a human-understandable way, and CLTs are a good first step.  This would have major implications in AI alignment, safety, and understanding, which would even allow humans to create more efficient and useful AI. 

## Overview

The cross-layer transcoder replaces MLP neurons with more interpretable features, allowing us to visualize and analyze how features interact across different layers of the model. This implementation provides tools for:

1. Training the cross-layer transcoder on GPT-2 Small, with configurations to change the model
2. Feature activations across different layers including visualizations
3. Creating attribution graphs showing feature influences
4. Comparing original model outputs with the replacement model

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cross-layer-transcoder.git
cd cross-layer-transcoder
```

This project uses Poetry for **dependency management** and **packaging**.
You can install poetry [here](https://python-poetry.org/docs/).

```bash
# Install dependencies
$ poetry install
Creating virtualenv openclt-E4zm1Hic-py3.12 in /home/user/.cache/pypoetry/virtualenvs
...
```
This will generate a **venv** folder where all the dependencies are stored.
You can find and activate the virtual enviornment with:
```bash
poetry env activate
> /some/path/pypoetry/virtualenvs/openclt-{uid}
# Linux / MacOS
source `~/.cache/pypoetry/virtualenvs/openclt-{uid}/bin/activate`
# Windows
./some/path/pypoetry/virtualenvs/openclt-{uid}/bin/activate
```

***Now you're set!***

Explore and play around with CrossLayerTranscoders, explore and add new features, share your discoveries!

#### VSCode users:
You will have to activate the virtual enviornment in your IDE. Follow [this](https://code.visualstudio.com/docs/python/environments) tutorial for more info.

## Usage

### Basic Example

```python
from open_cross_layer_transcoder import OpenCrossLayerTranscoder, ReplacementModel
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the cross-layer transcoder
transcoder = OpenCrossLayerTranscoder(
    model_name="gpt2",  # GPT-2 Small
    num_features=100,   # Number of interpretable features
    device=device
)

# Train the transcoder on sample texts
train_texts = [
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "New York City is the largest city in the United States.",
    # Add more training texts...
]

metrics = transcoder.train_transcoder(
    texts=train_texts,
    batch_size=2,
    num_epochs=3,
    learning_rate=1e-4
)

# Visualize feature activations for a test text
test_text = "The president of the United States lives in the White House."
transcoder.visualize_feature_activations(
    text=test_text,
    top_n=5,
    save_path='feature_activations.png'
)

# Create an attribution graph
transcoder.create_attribution_graph(
    text=test_text,
    threshold=0.1,
    save_path='attribution_graph.png'
)

# Create a replacement model
replacement_model = ReplacementModel(
    base_model_name="gpt2",
    transcoder=transcoder
)

# Generate text with the replacement model
generated_text = replacement_model.generate(
    text="Artificial intelligence",
    max_length=50
)
print(generated_text)

# Save the trained transcoder
transcoder.save_model('cross_layer_transcoder_gpt2.pt')
```

### Running the Example Script

```bash
python practice_run.py
```

This will:
1. Initialize the cross-layer transcoder with GPT-2 Small
2. Train it on sample texts
3. Visualize feature activations and create attribution graphs
4. Compare the original model with the replacement model
5. Save all visualizations to the 'visualizations' directory

### Advanced Visualizations

For more advanced visualizations, use the `visualization_utils.py` module:

```python
from visualization_utils import (
    visualize_feature_heatmap,
    visualize_feature_embedding,
    visualize_cross_layer_correlations,
    visualize_feature_importance,
    run_visualization_suite
)

# Run a complete suite of visualizations
run_visualization_suite(
    transcoder=transcoder,
    texts=test_texts,
    output_dir="visualizations"
)
```

## Implementation Details

### OpenCrossLayerTranscoder

The `OpenCrossLayerTranscoder` class implements the core functionality:

- **Initialization**: Sets up encoder/decoder networks for each layer of GPT-2
- **Training**: Trains the transcoder to reconstruct MLP activations from interpretable features
- **Feature Extraction**: Extracts feature activations for any input text
- **Visualization**: Creates visualizations of feature activations and attribution graphs

### ReplacementModel

The `ReplacementModel` class creates a modified version of GPT-2 where MLP outputs are replaced with reconstructions from the cross-layer transcoder:

- **Hooks**: Uses PyTorch hooks to replace MLP outputs during forward passes
- **Generation**: Supports text generation using the modified model
- **Comparison**: Allows comparison between original and replacement model outputs

### Visualization Utilities

The `visualization_utils.py` module provides additional visualization tools:

- **Feature Heatmaps**: Visualize feature activations as heatmaps
- **Feature Embeddings**: Project features into 2D space using t-SNE or PCA
- **Cross-Layer Correlations**: Analyze correlations between features across layers
- **Feature Importance**: Visualize the importance of different features

## Attribution Graphs

Attribution graphs show how features influence each other across different layers of the model. They help reveal the computational mechanisms underlying model behavior by tracing individual computational steps.

In these graphs:
- Nodes represent features in different layers
- Edges represent influence between features
- Edge thickness indicates strength of influence

## Special Shout Out To
CathK1

## References

This implementation is based on the paper:
- "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025)
- URL: https://transformer-circuits.pub/2025/attribution-graphs/methods.html

## Contributing

Feel free to raise issues or make changes and create a merge request.  Please contact etredal with supporting this project.

## License

MIT License
