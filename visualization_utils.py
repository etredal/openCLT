"""
Visualization Utilities for Cross-Layer Transcoder

This module provides additional visualization tools for the cross-layer transcoder
to help analyze and understand features across different layers of the model.
"""

import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from open_cross_layer_transcoder import OpenCrossLayerTranscoder
import os

def visualize_feature_heatmap(transcoder: OpenCrossLayerTranscoder, 
                             text: str,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap of feature activations across all layers.
    
    Args:
        transcoder: Trained cross-layer transcoder
        text: Input text
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get feature activations
    feature_acts = transcoder.get_feature_activations(text)
    
    # Create figure
    fig, axes = plt.subplots(len(feature_acts), 1, figsize=(12, 3*len(feature_acts)))
    
    # Tokenize for token labels
    tokens = transcoder.tokenizer.tokenize(text)
    
    for i, layer_idx in enumerate(sorted(feature_acts.keys())):
        ax = axes[i] if len(feature_acts) > 1 else axes
        
        # Get activations for this layer
        acts = feature_acts[layer_idx].cpu().numpy()[0]  # First batch item
        
        # Create heatmap
        sns.heatmap(acts.T, ax=ax, cmap='viridis', 
                   xticklabels=tokens if acts.shape[0] == len(tokens) else False,
                   yticklabels=False)
        
        # Set title and labels
        ax.set_title(f"Layer {layer_idx+1} Feature Activations")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Feature Index")
        
        # Rotate x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def visualize_feature_embedding(transcoder: OpenCrossLayerTranscoder,
                               texts: List[str],
                               layer_idx: int = 0,
                               method: str = 'tsne',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize feature embeddings for multiple texts using dimensionality reduction.
    
    Args:
        transcoder: Trained cross-layer transcoder
        texts: List of input texts
        layer_idx: Layer index to visualize
        method: Dimensionality reduction method ('tsne' or 'pca')
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get feature activations for all texts
    all_features = []
    
    for text in texts:
        feature_acts = transcoder.get_feature_activations(text)
        if layer_idx in feature_acts:
            # Get mean activation across tokens
            mean_acts = torch.mean(feature_acts[layer_idx][0], dim=0).cpu().numpy()
            all_features.append(mean_acts)
    
    if not all_features:
        raise ValueError(f"No feature activations found for layer {layer_idx}")
    
    # Convert to numpy array
    features_array = np.array(all_features)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2)
    
    reduced_features = reducer.fit_transform(features_array)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                        c=range(len(texts)), cmap='viridis', alpha=0.8)
    
    # Add labels
    for i, text in enumerate(texts):
        # Truncate long texts
        short_text = text[:20] + "..." if len(text) > 20 else text
        ax.annotate(short_text, (reduced_features[i, 0], reduced_features[i, 1]),
                   fontsize=8, alpha=0.7)
    
    # Set title and labels
    ax.set_title(f"Feature Embedding Visualization (Layer {layer_idx+1}, {method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def visualize_cross_layer_correlations(transcoder: OpenCrossLayerTranscoder,
                                      text: str,
                                      top_n: int = 10,
                                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize correlations between top features across different layers.
    
    Args:
        transcoder: Trained cross-layer transcoder
        text: Input text
        top_n: Number of top features to include
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get feature activations
    feature_acts = transcoder.get_feature_activations(text)
    
    # Get top features
    top_features = transcoder.get_top_features(top_n)
    
    # Create correlation matrix
    layer_indices = sorted(feature_acts.keys())
    num_layers = len(layer_indices)
    
    # Calculate total number of top features across all layers
    total_features = sum(len(top_features[layer_idx]) for layer_idx in layer_indices)
    
    # Create correlation matrix
    corr_matrix = np.zeros((total_features, total_features))
    
    # Create labels for features
    feature_labels = []
    feature_idx_map = {}  # Maps (layer_idx, feature_idx) to index in the correlation matrix
    
    # Fill feature labels and index map
    idx = 0
    for layer_idx in layer_indices:
        for feat_idx in top_features[layer_idx]:
            feature_labels.append(f"L{layer_idx+1}F{feat_idx}")
            feature_idx_map[(layer_idx, feat_idx)] = idx
            idx += 1
    
    # Calculate correlations
    for i, src_layer in enumerate(layer_indices):
        src_acts = feature_acts[src_layer][0].cpu().numpy()  # First batch item
        
        for src_feat in top_features[src_layer]:
            src_vec = src_acts[:, src_feat]
            src_idx = feature_idx_map[(src_layer, src_feat)]
            
            for j, tgt_layer in enumerate(layer_indices):
                tgt_acts = feature_acts[tgt_layer][0].cpu().numpy()
                
                for tgt_feat in top_features[tgt_layer]:
                    tgt_vec = tgt_acts[:, tgt_feat]
                    tgt_idx = feature_idx_map[(tgt_layer, tgt_feat)]
                    
                    # Normalize vectors
                    src_vec_norm = (src_vec - np.mean(src_vec)) / (np.std(src_vec) + 1e-8)
                    tgt_vec_norm = (tgt_vec - np.mean(tgt_vec)) / (np.std(tgt_vec) + 1e-8)
                    
                    # Calculate correlation
                    corr = np.abs(np.mean(src_vec_norm * tgt_vec_norm))
                    
                    # Fill correlation matrix
                    corr_matrix[src_idx, tgt_idx] = corr
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(corr_matrix, ax=ax, cmap='viridis', 
               xticklabels=feature_labels, yticklabels=feature_labels)
    
    # Set title and labels
    ax.set_title(f"Cross-Layer Feature Correlations\n{text[:30]}...")
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def visualize_feature_importance(transcoder: OpenCrossLayerTranscoder,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize feature importance across all layers.
    
    Args:
        transcoder: Trained cross-layer transcoder
        save_path: Path to save the visualization (optional)
        
    Returns:
        Matplotlib figure
    """
    # Get feature importance
    importance = transcoder.feature_importance.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(importance, ax=ax, cmap='viridis')
    
    # Set title and labels
    ax.set_title("Feature Importance Across Layers")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Layer")
    
    # Set y-ticks to layer numbers
    ax.set_yticks(range(importance.shape[0]))
    ax.set_yticklabels([f"Layer {i+1}" for i in range(importance.shape[0])])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    return fig

def run_visualization_suite(transcoder: OpenCrossLayerTranscoder, texts: List[str], output_dir: str = "visualizations"):
    """
    Run a complete suite of visualizations for the cross-layer transcoder.
    
    Args:
        transcoder: Trained cross-layer transcoder
        texts: List of input texts
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Running visualization suite...")
    
    # 1. Feature heatmaps
    for i, text in enumerate(texts):
        print(f"Creating feature heatmap for text {i+1}...")
        fig = visualize_feature_heatmap(
            transcoder=transcoder,
            text=text,
            save_path=f"{output_dir}/feature_heatmap_{i+1}.png"
        )
        plt.close(fig)
    
    # 2. Feature embeddings
    print("Creating feature embeddings visualization...")
    for layer_idx in range(transcoder.num_layers):
        for method in ['tsne', 'pca']:
            fig = visualize_feature_embedding(
                transcoder=transcoder,
                texts=texts,
                layer_idx=layer_idx,
                method=method,
                save_path=f"{output_dir}/feature_embedding_layer{layer_idx+1}_{method}.png"
            )
            plt.close(fig)
    
    # 3. Cross-layer correlations
    for i, text in enumerate(texts):
        print(f"Creating cross-layer correlations for text {i+1}...")
        fig = visualize_cross_layer_correlations(
            transcoder=transcoder,
            text=text,
            top_n=5,
            save_path=f"{output_dir}/cross_layer_correlations_{i+1}.png"
        )
        plt.close(fig)
    
    # 4. Feature importance
    print("Creating feature importance visualization...")
    fig = visualize_feature_importance(
        transcoder=transcoder,
        save_path=f"{output_dir}/feature_importance.png"
    )
    plt.close(fig)
    
    # 5. Attribution graphs (already in the main module)
    for i, text in enumerate(texts):
        print(f"Creating attribution graph for text {i+1}...")
        fig = transcoder.create_attribution_graph(
            text=text,
            threshold=0.99,
            save_path=f"{output_dir}/attribution_graph_{i+1}{datetime.now()}.png"
        )
        plt.close(fig)
    
    print(f"All visualizations saved to {output_dir}/")
