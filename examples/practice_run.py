"""
Basic Example of Cross-Layer Transcoder with GPT-2 Small

This script demonstrates how to use the cross-layer transcoder with GPT-2 Small
to visualize features across different layers of the model.
"""
# === Std lib ===
import os
import uuid
import gc
import time
from pathlib import Path

# === Local Lib ===
from openclt import OpenCrossLayerTranscoder, ReplacementModel
from openclt.types import TrainingMetric

# === Packages
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

import argparse


# Output directory
OUT_DIR: Path = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(exist_ok=True)

run_uuid = uuid.uuid4()
RUN_DIR: Path = OUT_DIR / f"run_{str(run_uuid)}" 
DATASET_DIR: Path = Path(__file__).resolve().parent / "datasets"

os.makedirs(RUN_DIR, exist_ok=True)

def main(args):
    print("Starting: " + str(RUN_DIR))
    print("Initializing Cross-Layer Transcoder with GPT-2 Small...")
    
    # Configure device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using Device: {device}")

    # Params
    model_name = "gpt2"  # GPT-2 Small
    num_features = 10
    batch_size = 1
    num_epochs = 2
    activation_type = "topk"
    topk_features = int(num_features * 0.02) 
    
    SAMPLES = f"{args.num_samples}" if args.num_samples else f'{args.percent_samples}%' if args.percent_samples else '5%'
    print(f"Using {SAMPLES} of dataset")

    # Initialize the cross-layer transcoder
    transcoder = OpenCrossLayerTranscoder(
        model_name=model_name,
        num_features=num_features,
        device=device,
        activation_type=activation_type,
        topk_features=topk_features
    )

    dataset = load_dataset("rojagtap/bookcorpus", split=f"train[:{SAMPLES}]", keep_in_memory=True)
    dataset.take(100)
    sentences = [entry["text"] for entry in dataset]
    train_texts = sentences[:800000]
    print(f"Loaded {len(train_texts)} training texts: {train_texts[:5]}")
    

    print("Cleaning memory before training...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    print("Training the Cross-Layer Transcoder...")
    
    # Train the transcoder
    metrics: TrainingMetric = transcoder.train_transcoder(
        texts=train_texts,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        l1_sparsity_coefficient=0.0,
        lr_scheduler_factor=0.5,      # Reduce LR to 10% of current
        lr_scheduler_patience=50     # After 7 epochs of no improvement in total_loss
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/3600:.2f} hours")


    # Plot training metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics.total_loss, label='Total Loss')
    plt.plot(metrics.reconstruction_loss, label='Reconstruction Loss')
    plt.plot(metrics.sparsity_loss, label='Sparsity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{RUN_DIR}/training_metrics.png')
    print(f"Training metrics saved to {RUN_DIR}/training_metrics.png")

    # Test texts for visualization
    test_texts = [
        "The president of the United States lives in the White House.",
        "Artificial intelligence systems can learn from data.",
        "The Sahara Desert is the largest hot desert in the world."
    ]
    
    # Visualize feature activations for each test text
    print("Visualizing feature activations across layers...")
    for i, text in enumerate(test_texts):
        fig = transcoder.visualize_feature_activations(
            text=text,
            top_n=5,
            save_path=RUN_DIR / f'feature_activations_{i+1}.png'
        )
        plt.close(fig)
        print(f"Feature activations for text {i+1} saved to {RUN_DIR}/feature_activations_{i+1}.png")

    # Create attribution graphs
    print("Creating attribution graphs...")
    for i, text in enumerate(test_texts):
        fig = transcoder.create_attribution_graph(
            text=text,
            threshold=0.8,
            save_path=f'{RUN_DIR}/attribution_graph_{i+1}.png'
        )
        plt.close(fig)
        print(f"Attribution graph for text {i+1} saved to {RUN_DIR}/attribution_graph_{i+1}.png")
    
    # Create a replacement model
    print("Creating replacement model...")
    replacement_model = ReplacementModel(
        base_model_name=model_name,
        transcoder=transcoder
    )
    
    # Compare original model vs replacement model
    print("Comparing original model vs replacement model outputs...")
    
    # Initialize the original model
    original_model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Compare outputs
    comparison_results = []
    
    for i, text in enumerate(test_texts):
        print(f"\nTest text {i+1}: {text}")
        
        # Tokenize input
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        
        # Get original model output
        with torch.no_grad():
            original_output = original_model(input_ids)
            original_logits = original_output.logits
        
        # Get replacement model output
        with torch.no_grad():
            replacement_output = replacement_model(input_ids)
            replacement_logits = replacement_output.logits
        
        # Calculate similarity between outputs
        # Fix: Use dim=0 for cosine similarity between flattened tensors
        similarity = torch.nn.functional.cosine_similarity(
            original_logits.view(-1, 1), 
            replacement_logits.view(-1, 1),
            dim=0
        ).item()
        
        # Calculate mean squared error
        mse = torch.nn.functional.mse_loss(
            original_logits, 
            replacement_logits
        ).item()
        
        comparison_results.append({
            'text': text,
            'similarity': similarity,
            'mse': mse
        })
        
        print(f"  Cosine similarity: {similarity:.4f}")
        print(f"  Mean squared error: {mse:.4f}")
        
        # Generate text with both models
        original_generated = original_model.generate(
            input_ids, 
            max_length=50,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        
        replacement_generated = replacement_model.generate(text, max_length=50)
        
        print(f"\n  Original model output: {tokenizer.decode(original_generated[0], skip_special_tokens=True)}")
        print(f"\n  Replacement model output: {replacement_generated}")
    
    # Visualize comparison results
    plt.figure(figsize=(10, 6))
    
    # Plot similarity
    plt.subplot(1, 2, 1)
    plt.bar(range(len(comparison_results)), [r['similarity'] for r in comparison_results])
    plt.xlabel('Text Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Output Similarity')
    plt.xticks(range(len(comparison_results)), [f"Text {i+1}" for i in range(len(comparison_results))])
    plt.grid(True, alpha=0.3)
    
    # Plot MSE
    plt.subplot(1, 2, 2)
    plt.bar(range(len(comparison_results)), [r['mse'] for r in comparison_results])
    plt.xlabel('Text Index')
    plt.ylabel('Mean Squared Error')
    plt.title('Output MSE')
    plt.xticks(range(len(comparison_results)), [f"Text {i+1}" for i in range(len(comparison_results))])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RUN_DIR}/model_comparison.png')
    print(f"Model comparison results saved to {RUN_DIR}/model_comparison.png")

    # Save the trained transcoder
    transcoder.save_model(f'{RUN_DIR}/cross_layer_transcoder_gpt2.pt')
    print(f"Trained cross-layer transcoder saved to {RUN_DIR}/cross_layer_transcoder_gpt2.pt")

    # Save the replacement model
    replacement_model.save_model(f'{RUN_DIR}/replacement_model_gpt2.pt')
    print(f"Replacement model saved to {RUN_DIR}/replacement_model_gpt2.pt")

    # Save the training metrics in json format
    print("Saving training metrics...")
    save_metrics = {
        'total_loss': metrics.total_loss,
        'reconstruction_loss': metrics.reconstruction_loss,
        'sparsity_loss': metrics.sparsity_loss
    }

    with open(f'{RUN_DIR}/training_metrics.json', 'w') as f:
        import json
        json.dump(save_metrics, f)
    print(f"Training metrics saved to {RUN_DIR}/training_metrics.json")

    # Save params
    params = {
        'model_name': model_name,
        'num_features': num_features,
        'device': device,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'activation_type': activation_type,
        'topk_features': topk_features
    }
    # Save params to a text file
    with open(f'{RUN_DIR}/params.txt', 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"Parameters saved to {RUN_DIR}/params.txt")

    print("\nExample completed successfully!")
    print(f"All visualizations and data are saved in the '{RUN_DIR}' directory.")


def validate_percentile_input(value: int) -> int:
    ivalue = int(value)
    if ivalue <= 0 or ivalue > 100:
        raise argparse.ArgumentTypeError(f"{value} is an invalid percentile")
    return ivalue

def validate_num_samples_input(value: int) -> int:
    ivalue = int(value)
    if ivalue < 100:
        raise argparse.ArgumentError(f"Must take in at least 100 samples")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        ===== openCLT =====\n
        """)
    parser.add_argument("--cpu", action='store_true', help="Enables usage on CPU. Omit if running on a GPU")
    parser.add_argument("--num_samples", type=int, help="Fixed number of samples to use")
    parser.add_argument("--percent_samples", type=validate_percentile_input, help="Percent of samples to use from the dataset. Must be > 0, <= 100")

    args = parser.parse_args()

    main(args)