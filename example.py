"""
Basic Example of Cross-Layer Transcoder with GPT-2 Small

This script demonstrates how to use the cross-layer transcoder with GPT-2 Small
to visualize features across different layers of the model.
"""

import torch
import matplotlib.pyplot as plt
from open_cross_layer_transcoder import OpenCrossLayerTranscoder, ReplacementModel
import os
import numpy as np
from tqdm import tqdm

# Set device
#device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(f"Using device: {device}")

# Create output directory for visualizations
os.makedirs("visualizations", exist_ok=True)

def main():
    print("Initializing Cross-Layer Transcoder with GPT-2 Small...")
    
    # Initialize the cross-layer transcoder
    transcoder = OpenCrossLayerTranscoder(
        model_name="gpt2",  # GPT-2 Small
        num_features=4000,   # Number of interpretable features
        device=device
    )
    
    # Sample texts for training and visualization
    train_texts = [
        "The old house creaked in the wind, filling me with unease.",
        "A cat sat on the windowsill, basking in the warm sunlight.",
        "The stock market experienced a sharp decline yesterday.",
        "Baking bread fills the kitchen with a comforting aroma.",
        "The scientist made a groundbreaking discovery in genetics.",
        "The children laughed as they chased bubbles in the park.",
        "A lone tree stood on the hilltop, silhouetted against the sunset.",
        "The detective carefully examined the evidence at the crime scene.",
        "The ancient ruins whispered stories of a forgotten civilization.",
        "The hiker reached the summit, breathless but exhilarated.",
        "The artist blended colors on the canvas, creating a vibrant scene.",
        "Heavy rain poured down, blurring the city lights.",
        "The musician played a soulful melody on the saxophone.",
        "The chef prepared a delicious meal with fresh ingredients.",
        "The astronaut floated weightlessly in the vastness of space.",
        "A sense of peace washed over her as she meditated.",
        "The car sped down the highway, leaving a trail of dust.",
        "The librarian quietly shelved books in the silent library.",
        "The programmer worked late into the night, debugging the code.",
        "The athlete crossed the finish line, exhausted but triumphant.",
        "The waves crashed against the shore, creating a rhythmic sound.",
        "A thick fog rolled in, obscuring the view of the harbor.",
        "The comedian told jokes that had the audience roaring with laughter.",
        "The teacher patiently explained the concept to the students.",
        "The train chugged along the tracks, carrying passengers to their destinations.",
        "A sense of nostalgia filled him as he looked through old photographs.",
        "The construction workers built a towering skyscraper in the city.",
        "The river flowed gently through the valley, carving a path through the rocks.",
        "The stars twinkled in the night sky, creating a breathtaking spectacle.",
        "A feeling of anticipation grew as the concert began.",
        "The clock ticked slowly, marking the passage of time.",
        "The old book smelled of paper and ink, hinting at its history.",
        "The gardener carefully pruned the roses in the garden.",
        "A wave of sadness washed over her as she said goodbye.",
        "The airplane took off into the sky, soaring above the clouds.",
        "The factory produced goods at a rapid pace, contributing to the economy.",
        "The dancer gracefully moved across the stage, captivating the audience.",
        "A feeling of curiosity sparked within him as he explored the unknown.",
        "The fire crackled in the fireplace, providing warmth and comfort.",
        "The wind howled outside, rattling the windows and doors.",
        "The children built a sandcastle on the beach, decorating it with shells.",
        "A sense of determination filled her as she faced the challenge.",
        "The computer processed information quickly and efficiently.",
        "The farmer harvested crops from the fields, ensuring a bountiful supply.",
        "The detective followed a trail of clues, hoping to solve the mystery.",
        "The mountain climber ascended the steep slope, driven by a sense of adventure.",
        "The sculptor shaped clay into a beautiful work of art.",
        "A feeling of gratitude filled her heart as she received the gift.",
        "The motorcycle roared down the road, its engine echoing through the air.",
        "The scientist conducted experiments in the laboratory, seeking answers to complex questions.",
        "The astronaut walked on the moon, leaving footprints in the lunar dust.",
        "A sense of wonder filled him as he gazed at the stars through the telescope.",
        "The clock chimed midnight, signaling the start of a new day.",
        "The ancient tree stood tall and strong, a symbol of resilience.",
        "The gardener planted seeds in the soil, nurturing new life.",
        "A wave of disappointment washed over her as her plans fell through.",
        "The rocket launched into space, carrying a satellite into orbit.",
        "The factory produced cars on an assembly line, contributing to mass transportation.",
        "The singer performed a powerful ballad, moving the audience to tears.",
        "A feeling of excitement grew as the roller coaster climbed the steep hill.",
        "The oven baked cookies, filling the house with a sweet fragrance.",
        "The river flowed through the forest, providing a habitat for diverse wildlife.",
        "A thick blanket of snow covered the ground, creating a winter wonderland.",
        "The comedian delivered a hilarious stand-up routine, eliciting laughter from the crowd.",
        "The teacher inspired students to learn and grow, shaping future generations.",
        "The train traveled across the country, connecting cities and people.",
        "A sense of longing filled him as he remembered past memories.",
        "The construction workers built a bridge across the river, facilitating transportation.",
        "The waves gently lapped against the shore, creating a soothing sound.",
        "The stars shone brightly in the clear night sky, offering a sense of peace.",
        "A feeling of anticipation built as the curtain rose on the stage.",
        "The clock ticked steadily, marking the passage of time.",
        "The antique furniture held stories of generations past.",
        "The gardener tended to the flowers, creating a vibrant display of colors.",
        "A wave of frustration washed over her as she encountered obstacles.",
        "The plane flew through the clouds, offering a breathtaking view of the landscape.",
        "The factory manufactured electronics, contributing to technological advancements.",
        "The dancer performed a passionate tango, expressing a range of emotions.",
        "A feeling of curiosity drove him to explore new places and ideas.",
        "The campfire crackled under the stars, providing warmth and companionship.",
        "The wind howled through the trees, creating an eerie atmosphere.",
        "The children played in the park, enjoying the freedom and joy of childhood.",
        "A sense of courage filled her as she faced her fears.",
        "The computer processed data with lightning speed, enabling complex calculations.",
        "The farmer harvested wheat from the fields, ensuring food security.",
        "The detective investigated a complex case, piecing together the puzzle.",
        "The explorer ventured into uncharted territories, seeking new discoveries.",
        "The painter created a masterpiece, capturing the beauty of the world.",
        "A feeling of love and affection filled her heart as she embraced her loved ones.",
        "The race car zoomed around the track, competing for the championship.",
        "The biologist studied microorganisms in the laboratory, uncovering the secrets of life.",
        "The pilot flew the airplane through turbulent weather, ensuring the safety of passengers.",
        "A sense of awe inspired him as he witnessed the power of nature.",
        "The bell tolled in the distance, marking the hour.",
        "The old photograph captured a moment in time, preserving a memory.",
        "The florist arranged flowers into a stunning bouquet, celebrating beauty and life.",
        "A wave of anger surged through her as she experienced injustice.",
        "The helicopter hovered above the city, providing aerial views of the landscape.",
        "The factory produced textiles, contributing to the fashion industry.",
        "The actor delivered a moving performance, portraying a complex character.",
        "A feeling of excitement bubbled up as she anticipated a special event.",
        "The oven roasted vegetables, filling the kitchen with a savory aroma.",
        "The river wound its way through the mountains, shaping the terrain.",
        "A thick layer of ice covered the lake, reflecting the winter sky.",
        "The comedian told jokes about everyday life, finding humor in the mundane.",
        "The mentor guided a young person, sharing wisdom and experience.",
        "The train sped across the countryside, offering glimpses of rural landscapes.",
        "A sense of loneliness washed over him as he found himself alone.",
        "The construction crew built a dam, controlling the flow of water.",
        "The waves rolled onto the shore, leaving behind seashells and seaweed.",
        "The stars twinkled in the clear night, inspiring dreams and aspirations.",
        "A feeling of joy radiated as she celebrated a special occasion.",
        "The clock chimed the hour, marking the passage of time.",
        "The antique jewelry sparkled with elegance, hinting at a rich history.",
        "The gardener pruned the bushes, shaping them into artistic forms.",
        "A wave of sadness overwhelmed her as she mourned a loss.",
        "The jet plane soared through the sky, connecting distant continents.",
        "The factory manufactured furniture, providing comfort and style to homes.",
        "The musician played a lively melody, encouraging dancing and celebration.",
        "A feeling of curiosity propelled him to explore the unknown.",
        "The bonfire burned brightly, casting shadows and creating a sense of community.",
        "The wind whispered through the leaves, creating a soothing sound.",
        "The children splashed in the puddles, enjoying the simple pleasures of rain.",
        "A sense of determination empowered her to overcome challenges.",
        "The computer executed complex calculations, enabling scientific discoveries.",
        "The farmer harvested corn from the fields, ensuring a plentiful harvest.",
        "The detective solved a puzzling case, bringing justice to the victims.",
        "The adventurer trekked through the jungle, encountering exotic wildlife.",
        "The artist painted a portrait, capturing the essence of a person's soul.",
        "A feeling of deep connection filled her heart as she shared a moment with a friend.",
        "The race car sped around the track, its engine roaring with power.",
        "The scientist studied cells under a microscope, revealing the intricacies of life.",
        "The pilot navigated the aircraft through a storm, ensuring the safety of everyone on board.",
        "A sense of wonder captivated him as he witnessed the beauty of the aurora borealis.",
        "The church bells rang, calling the community to worship.",
        "The old map guided explorers to hidden treasures, fueling their dreams of discovery.",
        "The baker prepared delicious pastries, tempting everyone with their sweet aroma.",
        "A wave of fear washed over her as she faced a dangerous situation.",
        "The hot air balloon drifted gently across the sky, offering panoramic views.",
        "The factory produced paper, contributing to communication and knowledge sharing.",
        "The dancer performed a graceful ballet, expressing emotions through movement.",
        "A feeling of anticipation heightened as she awaited a long-awaited reunion.",
        "The oven baked bread, filling the house with a comforting scent.",
        "The river carved a canyon through the landscape, showcasing the power of erosion.",
        "A thick fog blanketed the city, creating an atmosphere of mystery and intrigue.",
        "The comedian entertained the crowd with witty observations, sparking laughter and joy.",
        "The teacher inspired students to pursue their passions, shaping their futures.",
        "The train journeyed through tunnels and over bridges, connecting distant places.",
        "A sense of nostalgia enveloped him as he revisited his childhood home.",
        "The construction crew built a tunnel through the mountain, facilitating transportation and trade.",
        "The waves crashed against the cliffs, showcasing the raw power of the ocean.",
        "The stars glittered in the clear night sky, inspiring contemplation and wonder.",
        "A feeling of excitement surged as the fireworks exploded in a dazzling display.",
        "The clock ticked steadily onward, marking the relentless passage of time.",
        "The ancient artifact held secrets of a bygone era, sparking curiosity and intrigue.",
        "The gardener nurtured the garden, creating a haven of tranquility and beauty.",
        "A wave of grief overwhelmed her as she mourned the loss of a loved one.",
        "The spacecraft launched into the cosmos, embarking on a journey to explore distant galaxies.",
        "The factory manufactured textiles, contributing to the vibrant world of fashion.",
        "The musician played a moving symphony, evoking a range of emotions in the listeners.",
        "A feeling of curiosity and wonder propelled him to explore the mysteries of the universe.",
        "The campfire crackled under the starry sky, fostering camaraderie and warmth.",
        "The wind rustled through the leaves, creating a soothing and meditative atmosphere.",
        "The children splashed and played in the rain puddles, finding joy in the simple moments.",
        "A sense of resilience and determination empowered her to overcome obstacles and achieve her goals.",
        "The computer processed information with incredible speed and accuracy, revolutionizing various industries.",
        "The farmer harvested the ripe crops, ensuring a bountiful harvest and nourishing communities.",
        "The detective meticulously gathered evidence, piecing together the fragments of a complex crime.",
        "The explorer ventured into uncharted territories, driven by a thirst for discovery and adventure.",
        "The painter skillfully wielded brushes and colors, creating breathtaking works of art that captured the essence of life.",
        "A feeling of profound connection and love filled her heart as she shared intimate moments with her partner.",
        "The race car zoomed around the track, its powerful engine roaring as it competed for victory.",
        "The scientist peered through the microscope, unraveling the intricate details of the microscopic world.",
        "The pilot skillfully navigated the aircraft through challenging weather conditions, ensuring the safety and comfort of the passengers.",
        "A sense of awe and wonder enveloped him as he witnessed the majestic beauty of the natural world.",
        "The church bells chimed, resonating through the town and inviting people to gather in worship and community.",
        "The ancient map guided adventurers to hidden treasures, sparking their imagination and fueling their quest for the unknown.",
        "The baker crafted delectable pastries, filling the air with the enticing aroma of sweetness and warmth.",
        "A wave of intense fear washed over her as she found herself in a perilous and threatening situation.",
        "The hot air balloon gracefully ascended into the sky, offering breathtaking panoramic views of the landscape below.",
        "The factory efficiently produced paper, playing a vital role in communication, education, and knowledge dissemination.",
        "The dancer captivated the audience with their graceful and expressive movements, conveying a powerful story without words.",
        "A feeling of eager anticipation heightened as she eagerly awaited the arrival of a long-awaited guest.",
        "The oven gently baked a warm and comforting loaf of bread, filling the kitchen with its irresistible fragrance.",
        "The river carved its way through the rugged terrain, shaping the landscape and providing sustenance to the surrounding ecosystem.",
        "A thick and mysterious fog enveloped the city, creating an atmosphere of intrigue and suspense.",
        "The comedian delivered a hilarious performance, eliciting laughter and joy from the audience.",
        "The mentor generously shared their wisdom and guidance, empowering others to reach their full potential.",
        "The train traversed vast distances, connecting people and cultures across the country.",
        "A sense of profound loneliness lingered in the air as he found himself isolated and longing for connection.",
        "The construction crew worked diligently to build a massive dam, controlling the flow of water and providing vital resources.",
        "The waves crashed against the rocky cliffs, showcasing the untamed power and beauty of the ocean.",
        "The stars twinkled brilliantly in the clear night sky, inspiring dreams and aspirations for a brighter future.",
        "A feeling of pure joy and elation radiated as she celebrated a momentous achievement with loved ones.",
        "The clock ticked steadily onward, marking the relentless passage of time and the fleeting nature of moments.",
        "The ancient artifact whispered tales of a rich and vibrant history, sparking curiosity and a desire to learn more.",
        "The gardener lovingly tended to the garden, creating a sanctuary of tranquility and natural beauty.",
        "A wave of overwhelming grief washed over her as she mourned the profound loss of a cherished companion.",
        "The spacecraft embarked on a daring mission to explore the uncharted depths of the cosmos, pushing the boundaries of human knowledge.",
        "The factory efficiently produced a wide array of textiles, contributing to the ever-evolving world of fashion and design.",
        "The musician poured their heart and soul into a captivating symphony, evoking a powerful emotional response from the listeners.",
        "A sense of insatiable curiosity and boundless wonder propelled him to unravel the mysteries of the universe and explore the vast realms of knowledge.",
        "The campfire crackled under the expansive starry sky, fostering a sense of camaraderie and shared experience among the gathered companions.",
        "The gentle breeze rustled through the leaves of the trees, creating a soothing and meditative ambiance in the tranquil forest.",
        "The children gleefully splashed and played in the rain-soaked puddles, embracing the simple joys and carefree spirit of childhood.",
        "A sense of unwavering resilience and unwavering determination empowered her to conquer challenges and achieve her loftiest ambitions.",
        "The computer processed immense amounts of data with remarkable speed and precision, revolutionizing industries and shaping the course of technological advancement.",
        "The farmer diligently harvested the abundant crops, ensuring a plentiful supply of food and sustaining communities with their hard work and dedication.",
        "The detective meticulously pieced together the fragments of a complex crime, unraveling the truth and seeking justice for the victims.",
        "The intrepid explorer ventured into uncharted territories, driven by an insatiable thirst for discovery and a spirit of adventure.",
        "The artist skillfully wielded their brushes and paints, creating breathtaking masterpieces that captured the beauty and essence of the world around them.",
        "A feeling of profound love and deep connection filled her heart as she shared intimate moments and built lasting memories with her beloved partner.",
        "The race car roared around the track, its powerful engine pulsating as it competed for victory and pushed the limits of speed.",
        "The scientist meticulously studied cells and microorganisms under the microscope, meticulously unraveling the intricate secrets of life's building blocks.",
        "The skilled pilot masterfully navigated the aircraft through turbulent skies, ensuring the safety and well-being of the passengers and crew.",
        "A sense of awe and wonder enveloped him as he witnessed the breathtaking spectacle of the aurora borealis, a celestial dance of light and color.",
        "The church bells resounded through the town, calling the community together in worship, reflection, and shared fellowship.",
        "The ancient map held the key to unlocking hidden treasures and long-lost civilizations, igniting the imagination and fueling the spirit of adventure.",
        "The talented baker crafted delectable pastries and treats, filling the air with the irresistible aroma of sweetness and warmth.",
        "A wave of overwhelming fear washed over her as she found herself in a perilous and threatening situation, testing her courage and resilience.",
        "The hot air balloon gracefully ascended into the sky, offering unparalleled panoramic views of the sprawling landscape below.",
        "The factory played a crucial role in efficiently producing paper, contributing to the dissemination of knowledge, communication, and creative expression.",
        "The dancer captivated the audience with their mesmerizing performance, conveying a powerful narrative and evoking a range of emotions through graceful movements.",
        "The capital of France is Paris, which is known for the Eiffel Tower.",
        "New York City is the largest city in the United States.",
        "Machine learning models can process natural language.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "Quantum computers use qubits instead of classical bits.",
        "The Great Wall of China is visible from space.",
        "Neural networks have revolutionized artificial intelligence.",
        "Climate change is affecting ecosystems around the world.",
        "The human genome contains approximately 3 billion base pairs.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second."
    ]
    
    # Train the transcoder
    print("Training the Cross-Layer Transcoder...")
    metrics = transcoder.train_transcoder(
        texts=train_texts,
        batch_size=2,
        num_epochs=5,
        learning_rate=1e-4
    )
    
    # Plot training metrics
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['total_loss'], label='Total Loss')
    plt.plot(metrics['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(metrics['sparsity_loss'], label='Sparsity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('visualizations/training_metrics.png')
    print("Training metrics saved to visualizations/training_metrics.png")
    
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
            save_path=f'visualizations/feature_activations_{i+1}.png'
        )
        plt.close(fig)
        print(f"Feature activations for text {i+1} saved to visualizations/feature_activations_{i+1}.png")
    
    # Create attribution graphs
    print("Creating attribution graphs...")
    for i, text in enumerate(test_texts):
        fig = transcoder.create_attribution_graph(
            text=text,
            threshold=0.9,
            save_path=f'visualizations/attribution_graph_{i+1}.png'
        )
        plt.close(fig)
        print(f"Attribution graph for text {i+1} saved to visualizations/attribution_graph_{i+1}.png")
    
    # Create a replacement model
    print("Creating replacement model...")
    replacement_model = ReplacementModel(
        base_model_name="gpt2",
        transcoder=transcoder
    )
    
    # Compare original model vs replacement model
    print("Comparing original model vs replacement model outputs...")
    
    # Initialize the original model
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    original_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
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
    plt.savefig('visualizations/model_comparison.png')
    print("Model comparison results saved to visualizations/model_comparison.png")
    
    # Save the trained transcoder
    transcoder.save_model('cross_layer_transcoder_gpt2.pt')
    print("Trained cross-layer transcoder saved to cross_layer_transcoder_gpt2.pt")
    
    print("\nExample completed successfully!")
    print("All visualizations are saved in the 'visualizations' directory.")

if __name__ == "__main__":
    main()
