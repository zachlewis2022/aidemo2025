import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from bible_pytorch_classifier import PyTorchBibleClassifier, TextToTensor
import numpy as np
from typing import List, Tuple, Dict
import argparse
import logging

class TensorClient:
    def __init__(self):
        self.classifier = PyTorchBibleClassifier()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('TensorClient')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def visualize_embeddings(self, text: str) -> None:
        """Visualize word embeddings for a given text"""
        # Get word embeddings
        text_tensor = self.classifier.text_to_tensor.text_to_tensor(text)
        embeddings = self.classifier.model.embedding(text_tensor)
        
        # Reduce dimensionality for visualization (using PCA)
        U, S, V = torch.pca_lowrank(embeddings.detach(), q=2)
        reduced_embeddings = torch.matmul(embeddings.detach(), V[:, :2])
        
        # Plot
        plt.figure(figsize=(10, 6))
        words = text.split()
        x = reduced_embeddings[:len(words), 0].numpy()
        y = reduced_embeddings[:len(words), 1].numpy()
        
        plt.scatter(x, y)
        for i, word in enumerate(words):
            plt.annotate(word, (x[i], y[i]))
            
        plt.title('Word Embeddings Visualization (2D PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def visualize_attention(self, text: str) -> None:
        """Visualize attention weights for a given text"""
        _, attention = self.classifier.predict(text, return_attention=True)
        
        # Create heatmap
        plt.figure(figsize=(12, 3))
        words = text.split()
        attention = attention[:len(words)]
        
        sns.heatmap([attention], xticklabels=words, yticklabels=['Attention'],
                    cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'})
        plt.title('Attention Weights Visualization')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def analyze_tensor_operations(self, text: str) -> Dict:
        """Analyze various tensor operations in the model"""
        # Get initial tensor representation
        text_tensor = self.classifier.text_to_tensor.text_to_tensor(text)
        
        # Get embeddings
        embedded = self.classifier.model.embedding(text_tensor.unsqueeze(0))
        
        # Get LSTM outputs
        lstm_out, _ = self.classifier.model.lstm(embedded)
        
        # Get attention weights
        attention_weights = F.softmax(self.classifier.model.attention(lstm_out), dim=1)
        
        return {
            'input_shape': text_tensor.shape,
            'embedding_shape': embedded.shape,
            'lstm_output_shape': lstm_out.shape,
            'attention_shape': attention_weights.shape,
            'vocab_size': len(self.classifier.text_to_tensor.word2idx),
            'embedding_dim': embedded.shape[-1],
            'hidden_dim': lstm_out.shape[-1] // 2  # divide by 2 because bidirectional
        }

    def interactive_session(self):
        """Run an interactive tensor analysis session"""
        self.logger.info("Starting interactive tensor analysis session...")
        
        while True:
            print("\nTensor Analysis Options:")
            print("1. Analyze text and show tensor operations")
            print("2. Visualize word embeddings")
            print("3. Visualize attention weights")
            print("4. Train on new data")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                text = input("Enter text to analyze: ")
                analysis = self.analyze_tensor_operations(text)
                print("\nTensor Analysis Results:")
                for key, value in analysis.items():
                    print(f"{key}: {value}")
                    
            elif choice == '2':
                text = input("Enter text for embedding visualization: ")
                self.visualize_embeddings(text)
                
            elif choice == '3':
                text = input("Enter text for attention visualization: ")
                self.visualize_attention(text)
                
            elif choice == '4':
                texts = []
                labels = []
                print("Enter training examples (enter 'done' when finished)")
                while True:
                    text = input("Enter text (or 'done'): ")
                    if text.lower() == 'done':
                        break
                    label = input("Enter label: ")
                    texts.append(text)
                    labels.append(label)
                
                self.classifier.train(texts, labels)
                print("Training completed!")
                
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description='Tensor Operations Client')
    parser.add_argument('--interactive', action='store_true', 
                        help='Run in interactive mode')
    parser.add_argument('--analyze', type=str, help='Text to analyze')
    parser.add_argument('--visualize', type=str, 
                        help='Text to visualize embeddings and attention')
    
    args = parser.parse_args()
    client = TensorClient()
    
    if args.interactive:
        client.interactive_session()
    elif args.analyze:
        analysis = client.analyze_tensor_operations(args.analyze)
        print("\nTensor Analysis Results:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
    elif args.visualize:
        print("Generating visualizations...")
        client.visualize_embeddings(args.visualize)
        client.visualize_attention(args.visualize)
    else:
        print("No action specified. Use --help for usage information.")

if __name__ == "__main__":
    main()