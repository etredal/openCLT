"""
Cross-Layer Transcoder Implementation for GPT-2 Small

This module implements a cross-layer transcoder for GPT-2 Small based on the paper:
"Circuit Tracing: Revealing Computational Graphs in Language Models"
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

The cross-layer transcoder replaces MLP neurons with more interpretable features,
allowing us to visualize and analyze how features interact across different layers.
"""

# === Std Lib ===
from pathlib import Path

# === Local Lib ===
from openclt.types.metrics import TrainingMetric

# === Packages === 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any


DEBUG = True

def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)

class jumprelu(torch.autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: Tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
        ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        # Gradient is 1 where x > threshold, scaled by bandwidth
        x_grad = (x >threshold) * grad_output
        threshold_grad = torch.sum(
            -(threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0
        )
        return x_grad, threshold_grad, None

class JumpReLU(torch.nn.Module):
    def __init__(self, threshold: float, bandwidth: float) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return jumprelu.apply(x, self.threshold, self.bandwidth)
    
    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, bandwidth={self.bandwidth}"

class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(x, k=self.k, dim=-1)
        gate = torch.zeros_like(x)
        gate.scatter_(dim=-1, index=indices, value=1)
        return x * gate.to(x.dtype)
    
class OpenCrossLayerTranscoder(nn.Module):
    
    def __init__(self, model_name: str = "gpt2", num_features: int = 100, 
                 device: str = "cpu", activation_type: str = "jumprelu", topk_features: int = None):
        """
        Initialize the cross-layer transcoder.
        
        Args:
            model_name: Name of the GPT-2 model to use
            num_features: Number of interpretable features to extract
            device: Device to run the model on ("cpu" or "cuda")
            activation_type: Type of activation function to use in the transcoder
                             (e.g., "jumprelu", "relu", "topk")
            topk_features: If using "topk" activation, specify the number of top features to keep
        """
        super().__init__()
        self.device = torch.device(device=device)
        self.model_name = model_name
        self.num_features = num_features
        self.activation_type = activation_type
        
        # Load the base GPT-2 model
        self.base_model = GPT2Model.from_pretrained(model_name).to(device)
        self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Get model dimensions
        self.config = self.base_model.config
        self.num_layers = self.config.n_layer
        self.hidden_size = self.config.n_embd
        
        # Create encoder for each layer
        self.encoders = nn.ModuleList([
            nn.Linear(self.hidden_size, num_features) 
            for _ in range(self.num_layers)
        ]).to(device)

        # Create encoder for each layer
        self.decoders = nn.ModuleDict()
        for encoder_layer in range (self.num_layers):
            for decoder_layer in range (encoder_layer, self.num_layers): # current layer and all subsequent layers only (no backwards)
                key = f"{encoder_layer}_{decoder_layer}"
                self.decoders[key] = nn.Linear(num_features, self.hidden_size).to(device)

        self._initialize_weights()

        # Activation functions for each layer
        if activation_type == "jumprelu":
            self.activation_functions = nn.ModuleList([
                JumpReLU(threshold=0.03, bandwidth=1.0)  # Example threshold and bandwidth from the paper
                for _ in range(self.num_layers)
            ]).to(device)
        elif activation_type == "topk":
            if topk_features is None:
                topk_features = max(1, num_features // 20)  # Default to 5% of num_features
            self.activation_functions = nn.ModuleList([
                TopK(k=topk_features) 
                for _ in range(self.num_layers)
                ]).to(device)
        else:
            self.activation_functions = nn.ModuleList([
                nn.ReLU()  # Default to ReLU if no specific activation is chosen
                for _ in range(self.num_layers)
            ]).to(device)

        # Initialize the hooks and activation storage
        self.hooks = []
        self.mlp_activations = {}
        self.mlp_inputs_captured = {}
        self._register_hooks()
        
        # Feature importance tracking
        self.feature_importance = torch.zeros(self.num_layers, num_features).to(device)

    def _initialize_weights(self):
        for encoder_layer in range(self.num_layers):
            # Initialize encoder weights
            std_encoder = 1.0 / np.sqrt(self.num_features) #specified in the paper
            nn.init.uniform_(self.encoders[encoder_layer].weight, -std_encoder, std_encoder)
            nn.init.zeros_(self.encoders[encoder_layer].bias)
            
            # Initialize all decoders for this encoder layer
            for decoder_layer in range(encoder_layer, self.num_layers):
                key = f"{encoder_layer}_{decoder_layer}"
                # Initialize decoder weights
                std_decoder = 1.0 / np.sqrt(self.num_layers * self.hidden_size) # specified in the paper   
                nn.init.uniform_(self.decoders[key].weight, -std_decoder, std_decoder)
                nn.init.zeros_(self.decoders[key].bias)
        
    def print_activations(self, layer_idx, features_relu, num_features):
        if features_relu.numel() > 0: # Check if tensor is not empty
            print(f"\n--- Debug Sample: Layer {layer_idx} Feature Activations (features_relu) ---")
            print(f"Shape of features_relu: {features_relu.shape} (Batch, SeqLen, NumFeatures)")
            
            # Print all feature activations for the first token of the first item in the batch
            if features_relu.shape[0] > 0 and features_relu.shape[1] > 0:
                print(f"Activations for B[0], T[0] (all {num_features} features):")
                print(features_relu[0, 0, :])
            
            print(f"--- End Debug Sample ---")

    def _register_hooks(self):
        """Register forward hooks to capture MLP activations from each layer."""
        def hook_fn(layer_idx):
            def hook(module, input_args, output_tensor):
                self.mlp_inputs_captured[layer_idx] = input_args[0]
                self.mlp_activations[layer_idx] = output_tensor
                return output_tensor
            return hook
        
        # Remove any existing hooks
        self.remove_hooks()
        
        # Register new hooks for each MLP layer
        for i in range(self.num_layers):
            # Access the MLP module in each transformer block
            mlp = self.base_model.h[i].mlp
            hook = mlp.register_forward_hook(hook_fn(i))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model and cross-layer transcoder.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Dictionary containing:
                - 'last_hidden_state': The base model's output
                - 'feature_activations': Feature activations for each layer
                - 'reconstructed_activations': Reconstructed MLP activations
        """
        # Clear previous activations
        self.mlp_inputs_captured.clear()
        self.mlp_activations.clear()
        
        # Forward pass through base model to collect MLP activations via hooks
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Process each layer's activations through the transcoder
        feature_activations = {}
        reconstructed_activations = {}
        
        for layer_idx in range(self.num_layers):
            if layer_idx in self.mlp_inputs_captured:
                # MLP input
                mlp_input = self.mlp_inputs_captured[layer_idx] # this is what gets read
                mlp_output = self.mlp_activations[layer_idx] # this is what gets written

                mlp_input_normalized = F.layer_norm(mlp_input, (mlp_input.shape[-1],)) # Normalize input to MLP
                mlp_output_normalized = F.layer_norm(mlp_output, (mlp_output.shape[-1],)) # Normalize output of MLP
                # Encode to features
                features = self.encoders[layer_idx](mlp_input_normalized) # linear encoder
                features_activated = self.activation_functions[layer_idx](features) # non-linear activation
                feature_activations[layer_idx] = features_activated

                # Sum contributions from all previous layers to this one 
                reconstructed = torch.zeros_like(self.mlp_activations[layer_idx])
                for encoder_layer in range(layer_idx + 1): # all layer up to and including current one 
                    if encoder_layer in feature_activations:
                        key = f"{encoder_layer}_{layer_idx}"
                        reconstructed += self.decoders[key](feature_activations[encoder_layer])

                reconstructed_activations[layer_idx] = reconstructed
                
                # Update feature importance based on activation magnitude
                with torch.no_grad():
                    importance = torch.mean(torch.abs(features_activated), dim=(0, 1))
                    self.feature_importance[layer_idx] += importance
        
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'feature_activations': feature_activations,
            'reconstructed_activations': reconstructed_activations
        }
    
    def train_transcoder(self, 
                         texts: List[str], 
                         batch_size: int = 4, 
                         num_epochs: int = 5,
                         learning_rate: float = 1e-4,
                         l1_sparsity_coefficient: float = 0.05,
                         lr_scheduler_factor: float = 0.1,
                         lr_scheduler_patience: int = 100,
                         ) -> TrainingMetric:
        """
        Train the cross-layer transcoder on a corpus of texts.
        
        Args:
            texts: List of text samples to train on
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            l1_sparsity_coefficient: Coefficient for L1 sparsity loss on features
            lr_scheduler_factor: Factor by which the learning rate will be reduced by scheduler.
            lr_scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced.
            
        Returns:
            Dictionary of training metrics
        """
        # Set model to training mode
        self.train()
        
        all_params = (list(self.encoders.parameters()) 
                      + list(self.decoders.parameters()) 
                      + list(self.activation_functions.parameters()))

        # Create optimizer
        optimizer = torch.optim.Adam(
            all_params, lr=learning_rate
        )

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min', # We want the loss to decrease
                                      factor=lr_scheduler_factor,
                                      patience=lr_scheduler_patience
        )        
        # Training metrics
        metrics = TrainingMetric()
        
        # Tokenize all texts
        encoded_texts = [self.tokenizer.encode(text, return_tensors="pt").to(self.device) for text in texts]
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_sparsity_loss = 0
            epoch_l0_metric = 0
            
            # Process in batches
            num_batches = (len(encoded_texts) + batch_size - 1) // batch_size
            
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(encoded_texts))
                batch_texts = encoded_texts[start_idx:end_idx]
                
                # Pad to same length within batch
                max_len = max(text.size(1) for text in batch_texts)
                padded_texts = []
                attention_masks = []
                
                for text in batch_texts:
                    pad_len = max_len - text.size(1)
                    padded_text = F.pad(text, (0, pad_len), value=self.tokenizer.pad_token_id)
                    mask = torch.ones_like(padded_text, dtype=torch.float)
                    mask[:, -pad_len:] = 0 if pad_len > 0 else 1
                    
                    padded_texts.append(padded_text)
                    attention_masks.append(mask)
                
                input_ids = torch.cat(padded_texts, dim=0)
                attention_mask = torch.cat(attention_masks, dim=0)
                
                # Clear gradients
                optimizer.zero_grad()

                self.mlp_inputs_captured.clear()
                self.mlp_activations.clear()
                
                # Forward pass to collect MLP activations
                self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Compute loss for each layer
                total_loss = 0
                reconstruction_loss = 0
                sparsity_loss = 0
                

                all_features = {}  # Store features for reconstruction

                for layer_idx in range(self.num_layers):
                    if layer_idx in self.mlp_inputs_captured:
                        mlp_input = self.mlp_inputs_captured[layer_idx]
                        mlp_output = self.mlp_activations[layer_idx]

                        # Normalize input to MLP
                        mlp_input_normalized = F.layer_norm(mlp_input, mlp_input.shape[-1:])  # Normalize input to MLP
                        mlp_output_normalized = F.layer_norm(mlp_output, mlp_output.shape[-1:])  # Normalize output of MLP

                        # Encode to features
                        features = self.encoders[layer_idx](mlp_input_normalized)
                        
                        # Apply activation function
                        features_activated = self.activation_functions[layer_idx](features)

                        all_features[layer_idx] = features_activated

                        # Sum cintributions from all previous layers (same as in forward)
                        reconstructed = torch.zeros_like(mlp_output)
                        for encoder_layer in range(layer_idx + 1):
                            if encoder_layer in all_features:
                                key = f"{encoder_layer}_{layer_idx}"
                                reconstructed += self.decoders[key](all_features[encoder_layer])

                        
                        if DEBUG == True and batch_idx == 0 and epoch == 0: 
                            self.print_activations(layer_idx, features_activated, self.num_features)

                        # L0 metric (sparsity)
                        l0_metric = torch.mean((features_activated > 1e-6).float())

                        # Reconstruction loss (MSE)
                        recon_loss = F.mse_loss(reconstructed, mlp_output_normalized)
                        reconstruction_loss += recon_loss
                        
                        # Sparsity loss (L1 regularization on features)
                        if self.activation_type != "topk":
                            decoder_norms = []
                            for decoder_layer in range(layer_idx, self.num_layers):
                                key = f"{layer_idx}_{decoder_layer}"
                                decoder_norm = torch.norm(self.decoders[key].weight, dim=1)
                                decoder_norms.append(decoder_norm)
                            
                            total_decoder_norm = torch.stack(decoder_norms, dim=0).sum(dim=0)
                            c = 1.0
                            feature_sparsity = torch.tanh(c * total_decoder_norm * features_activated.abs())
                            l1_loss = l1_sparsity_coefficient * torch.mean(feature_sparsity)
                            sparsity_loss += l1_loss
                        else:
                            l1_loss = 0
                        # Add to total loss
                        total_loss += recon_loss + l1_loss
                
                # Backward pass and optimization
                total_loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                

                # Update metrics
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += reconstruction_loss.item()
                if isinstance(sparsity_loss, torch.Tensor):
                    epoch_sparsity_loss += sparsity_loss.item()
                else:
                    epoch_sparsity_loss += sparsity_loss
                epoch_l0_metric += l0_metric.item()
            
            # Record epoch metrics
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_sparsity_loss = epoch_sparsity_loss / num_batches
            avg_l0_metric = epoch_l0_metric / num_batches
            
            metrics.total_loss.append(avg_total_loss)
            metrics.reconstruction_loss.append(avg_recon_loss)
            metrics.sparsity_loss.append(avg_sparsity_loss)
            metrics.l0_metric.append(avg_l0_metric)
            metrics.learning_rate.append(optimizer.param_groups[0]['lr'])

            # Step the scheduler
            scheduler.step(avg_total_loss) # Step with the monitored metric
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Loss = {avg_total_loss:.4f}, "
                  f"Recon = {avg_recon_loss:.4f}, "
                  f"Sparsity = {avg_sparsity_loss:.4f}, "
                  f"L0 Metric = {avg_l0_metric:.4f}")

            if self.activation_type == "jumprelu":
                thresholds = [act.threshold.item() for act in self.activation_functions[:3]]
                print(f"JumpReLU thresholds first 3 layers: {[f'{t:.3f}' for t in thresholds]}")
            
        return metrics
    
    def get_feature_activations(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Get feature activations for all layers on a given text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping layer indices to feature activations
        """
        # Set to evaluation mode
        self.eval()
        
        # Tokenize input
        input_ids: torch.Tensor = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids)
        print([k, v] for k, v in outputs.items())

        return outputs['feature_activations']
    
    def get_top_features(self, n: int = 10) -> Dict[int, List[int]]:
        """
        Get the indices of the top N most important features for each layer.
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary mapping layer indices to lists of top feature indices
        """
        top_features = {}
        
        for layer_idx in range(self.num_layers):
            # Get importance scores for this layer
            importance = self.feature_importance[layer_idx]
            
            # Get indices of top N features
            _, indices = torch.topk(importance, min(n, self.num_features))
            top_features[layer_idx] = indices.cpu().numpy().tolist()
        
        return top_features
    
    def visualize_feature_activations(self, 
                                     text: str, 
                                     top_n: int = 5,
                                     save_path: Optional[Path] = None) -> plt.Figure:
        """
        Visualize feature activations across layers for a given text.
        
        Args:
            text: Input text
            top_n: Number of top features to visualize
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        # Get feature activations
        feature_acts = self.get_feature_activations(text)
        
        # Get top features
        top_features = self.get_top_features(top_n)
        
        # Create figure
        fig, axes = plt.subplots(self.num_layers, 1, figsize=(12, 2*self.num_layers))
        
        # Tokenize for token labels
        tokens = self.tokenizer.tokenize(text)
        
        for layer_idx in range(self.num_layers):
            ax = axes[layer_idx] if self.num_layers > 1 else axes
            
            if layer_idx in feature_acts:
                # Get activations for this layer
                acts = feature_acts[layer_idx].cpu().numpy()[0]  # First batch item
                
                # Get top features for this layer
                top_feat_idx = top_features[layer_idx]
                
                # Plot activations for top features
                for i, feat_idx in enumerate(top_feat_idx):
                    ax.plot(acts[:, feat_idx], label=f"Feature {feat_idx}")
                
                # Set x-ticks to token positions
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                
                # Set title and labels
                ax.set_title(f"Layer {layer_idx+1} Feature Activations")
                ax.set_ylabel("Activation")
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def create_attribution_graph(self, 
                                text: str,
                                threshold: float = 0.8,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create an attribution graph showing feature influences across layers.
        
        Args:
            text: Input text
            threshold: Threshold for including connections in the graph
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        # Set to evaluation mode
        self.eval()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Get feature activations
        with torch.no_grad():
            outputs = self(input_ids)
            feature_acts = outputs['feature_activations']
        
        # Get top features for each layer
        top_features = self.get_top_features(5)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Track node positions
        node_positions = {}
        
        # Draw nodes for each layer
        for layer_idx in range(self.num_layers):
            if layer_idx in feature_acts:
                # Get top features for this layer
                top_feat_idx = top_features[layer_idx]
                
                # Draw nodes for top features
                for i, feat_idx in enumerate(top_feat_idx):
                    x = layer_idx
                    y = i * 0.5
                    
                    # Draw node
                    circle = plt.Circle((x, y), 0.1, fill=True, color='skyblue', alpha=0.8)
                    ax.add_patch(circle)
                    
                    # Add label
                    ax.text(x, y, f"L{layer_idx}F{feat_idx}", ha='center', va='center', fontsize=8)
                    
                    # Store position
                    node_positions[(layer_idx, feat_idx)] = (x, y)
        
        # Draw edges between layers
        for src_layer in range(self.num_layers - 1):
            if src_layer in feature_acts:
                src_feats = top_features[src_layer]
                
                for tgt_layer in range(src_layer + 1, self.num_layers):
                    if tgt_layer in feature_acts:
                        tgt_feats = top_features[tgt_layer]
                        
                        # Get activations
                        src_acts = feature_acts[src_layer][0].cpu().numpy()  # First batch item
                        tgt_acts = feature_acts[tgt_layer][0].cpu().numpy()
                        
                        # Calculate correlations between features
                        for src_feat in src_feats:
                            for tgt_feat in tgt_feats:
                                # Simple correlation measure
                                src_vec = src_acts[:, src_feat]
                                tgt_vec = tgt_acts[:, tgt_feat]
                                
                                # Normalize vectors
                                src_vec = (src_vec - np.mean(src_vec)) / (np.std(src_vec) + 1e-8)
                                tgt_vec = (tgt_vec - np.mean(tgt_vec)) / (np.std(tgt_vec) + 1e-8)
                                
                                # Calculate correlation
                                corr = np.abs(np.mean(src_vec * tgt_vec))
                                
                                # Draw edge if correlation is above threshold
                                if corr > threshold:
                                    src_pos = node_positions[(src_layer, src_feat)]
                                    tgt_pos = node_positions[(tgt_layer, tgt_feat)]
                                    
                                    # Draw line with width proportional to correlation
                                    line_width = corr * 3
                                    ax.plot([src_pos[0], tgt_pos[0]], [src_pos[1], tgt_pos[1]], 
                                           'k-', alpha=min(corr, 0.8), linewidth=line_width)
        
        # Set axis limits and labels
        ax.set_xlim(-0.5, self.num_layers - 0.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Feature")
        ax.set_title(f"Attribution Graph for: {text[:30]}...")
        
        # Set x-ticks to layer numbers
        ax.set_xticks(range(self.num_layers))
        ax.set_xticklabels([f"Layer {i+1}" for i in range(self.num_layers)])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def save_model(self, path: str):
        """Save the cross-layer transcoder model."""
        torch.save({
            'encoders': self.encoders.state_dict(),
            'decoders': self.decoders.state_dict(),
            'feature_importance': self.feature_importance,
            'config': {
                'model_name': self.model_name,
                'num_features': self.num_features,
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size
            }
        }, path)
        
    @classmethod
    def load_model(cls, path: str, device: str = "cpu"):
        """Load a saved cross-layer transcoder model."""
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved config
        config = checkpoint['config']
        model = cls(
            model_name=config['model_name'],
            num_features=config['num_features'],
            device=device
        )
        
        # Load state dictionaries
        model.encoders.load_state_dict(checkpoint['encoders'])
        model.decoders.load_state_dict(checkpoint['decoders'])
        model.feature_importance = checkpoint['feature_importance']
        
        return model


class ReplacementModel(nn.Module):
    """
    Replacement model that substitutes MLP neurons with cross-layer transcoder features.
    
    This model replaces the MLP outputs in each layer with the reconstructed outputs
    from the cross-layer transcoder, allowing us to analyze how features interact
    across different layers.
    """
    
    def __init__(self, base_model_name: str = "gpt2", transcoder: OpenCrossLayerTranscoder = None):
        """
        Initialize the replacement model.
        
        Args:
            base_model_name: Name of the base GPT-2 model
            transcoder: Trained cross-layer transcoder
        """
        super().__init__()
        self.device = transcoder.device if transcoder else "cpu"
        
        # Load the base model
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store the transcoder
        self.transcoder = transcoder
        
        self.stored_features = {}

        # Register hooks to replace MLP outputs
        self.hooks = []
        self._register_replacement_hooks()
    
    def _register_replacement_hooks(self):
        """Register hooks to replace MLP outputs with transcoder reconstructions."""
        def hook_fn(layer_idx):
            def hook(module, input_args, output_tensor):
                # If we have a transcoder and it has processed this layer
                if self.transcoder and layer_idx < self.transcoder.num_layers: # Check transcoder and layer index validity
                    # Get the LIVE input to the current MLP block (r_k for this layer in ReplacementModel)

                    self.transcoder.eval()
                    with torch.no_grad():
                        current_mlp_input = input_args[0]  # This is the input to the MLP block (r_k)
                        current_mlp_input_rk = F.layer_norm(current_mlp_input, [current_mlp_input.shape[-1]])

                    # Encode using the transcoder's encoder, which is now trained to expect r_k
                    features = self.transcoder.encoders[layer_idx](current_mlp_input_rk)
                    features_activated = self.transcoder.activation_functions[layer_idx](features)

                    # store features for this layer
                    self.stored_features[layer_idx] = features_activated

                    # Decode back to MLP output space. The decoder was trained to produce MLP_k(r_k).
                    reconstructed = torch.zeros_like(output_tensor)
                    for encoder_layer in range(layer_idx + 1): # all layers up to and including current one
                        if encoder_layer in self.stored_features:
                            key = f"{encoder_layer}_{layer_idx}"
                            reconstructed += self.transcoder.decoders[key](self.stored_features[encoder_layer])

                    return reconstructed
                else:
                    return output_tensor
            return hook
        
        # Remove any existing hooks
        self.remove_hooks()
        
        # Register new hooks for each MLP layer
        for i in range(self.transcoder.num_layers):
            # Access the MLP module in each transformer block
            mlp = self.base_model.transformer.h[i].mlp
            hook = mlp.register_forward_hook(hook_fn(i))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the replacement model.
        
        """
        self.stored_features.clear()  # Clear stored features from previous runs

        # First run through transcoder to collect activations
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs
    
    def generate(self, text: str, max_length: int = 50) -> str:
        """
        Generate text using the replacement model.
        
        Args:
            text: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        print(f"Shape of input_ids before generate: {input_ids.shape}")
        
        # Generate
        output_ids = self.base_model.generate(
            input_ids, 
            max_length=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            attention_mask=None,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return output_text
    
    def save_model(self, path: str):
        """Save the replacement model."""
        torch.save({
            'base_model': self.base_model.state_dict(),
            'replacement_model': self.transcoder.state_dict()
        }, path)