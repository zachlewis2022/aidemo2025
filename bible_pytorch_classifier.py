import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class TextToTensor:
    def __init__(self, min_freq=2):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Create vocabulary (only include words that appear min_freq times)
        vocab = ['<PAD>', '<UNK>']  # Special tokens
        vocab.extend([word for word, count in word_counts.items() 
                     if count >= self.min_freq])
        
        # Create word-to-index mappings
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
    def text_to_tensor(self, text, max_len=50):
        """Convert text to tensor of indices"""
        words = text.lower().split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                  for word in words[:max_len]]
        
        # Pad sequence if necessary
        if len(indices) < max_len:
            indices.extend([self.word2idx['<PAD>']] * (max_len - len(indices)))
            
        return torch.tensor(indices, dtype=torch.long)

class BibleDataset(Dataset):
    def __init__(self, texts, labels, text_to_tensor):
        self.texts = texts
        self.labels = labels
        self.text_to_tensor = text_to_tensor
        
        # Convert labels to indices
        unique_labels = sorted(set(labels))
        self.label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx2label = {idx: label for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text and label to tensors
        text_tensor = self.text_to_tensor.text_to_tensor(text)
        label_tensor = torch.tensor(self.label2idx[label], dtype=torch.long)
        
        return text_tensor, label_tensor

class BibleClassifierNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        
        # Word embeddings layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                           bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Get embeddings
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  # Shape: (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply dropout and get final predictions
        dropped = self.dropout(context_vector)
        output = self.fc(dropped)
        
        return output, attention_weights

class PyTorchBibleClassifier:
    def __init__(self, embed_dim=100, hidden_dim=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.text_to_tensor = TextToTensor()
        self.model = None
        
    def train(self, texts, labels, epochs=10, batch_size=32):
        # Build vocabulary
        self.text_to_tensor.fit(texts)
        
        # Create dataset and dataloader
        dataset = BibleDataset(texts, labels, self.text_to_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        vocab_size = len(self.text_to_tensor.word2idx)
        num_classes = len(set(labels))
        self.model = BibleClassifierNN(vocab_size, self.embed_dim, 
                                     self.hidden_dim, num_classes)
        self.model.to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    def predict(self, text, return_attention=False):
        """Predict topic and return attention weights if requested"""
        self.model.eval()
        with torch.no_grad():
            # Convert text to tensor
            text_tensor = self.text_to_tensor.text_to_tensor(text).unsqueeze(0)
            text_tensor = text_tensor.to(self.device)
            
            # Get prediction and attention weights
            outputs, attention = self.model(text_tensor)
            predicted_idx = outputs.argmax(dim=1).item()
            
            # Convert prediction to label
            dataset = self.model.dataset if hasattr(self.model, 'dataset') else None
            if dataset:
                predicted_label = dataset.idx2label[predicted_idx]
            else:
                predicted_label = str(predicted_idx)
                
            if return_attention:
                return predicted_label, attention.squeeze().cpu().numpy()
            return predicted_label

# Example usage and demonstration
def demonstrate_tensor_operations():
    # Example text
    text = "And I saw a beast rise up out of the sea"
    
    # Initialize classifier
    classifier = PyTorchBibleClassifier()
    
    # Example training data
    texts = [
        "And I saw a beast rise up out of the sea having seven heads",
        "The ram which you saw having two horns are the kings",
        "I saw in the night visions, and behold, one like the Son of man",
    ]
    labels = ["Beasts", "Interpretation", "Prophecy"]
    
    # Train model
    classifier.train(texts, labels)
    
    # Make prediction
    prediction, attention = classifier.predict(text, return_attention=True)
    
    # Show how tensors are used
    print("\nTensor Operations Example:")
    print("-" * 50)
    print(f"Input text: {text}")
    
    # Show text to tensor conversion
    text_tensor = classifier.text_to_tensor.text_to_tensor(text)
    print(f"\nText as tensor (word indices):\n{text_tensor}")
    
    # Show embedding lookup
    embedded = classifier.model.embedding(text_tensor.unsqueeze(0))
    print(f"\nEmbedding shape: {embedded.shape}")
    print("Each word is now represented by a {}-dimensional vector".format(
        classifier.embed_dim))
    
    # Show attention weights
    print(f"\nAttention weights shape: {attention.shape}")
    print("Words with highest attention:")
    words = text.split()
    for word, weight in zip(words, attention[:len(words)]):
        print(f"{word}: {weight:.4f}")

if __name__ == "__main__":
    demonstrate_tensor_operations()