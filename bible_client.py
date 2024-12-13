import argparse
import sys
from pathlib import Path
import json
import pandas as pd
from bible_classifier import BibleTextClassifier  # This imports our previous model

class BibleClassifierClient:
    def __init__(self):
        self.classifier = BibleTextClassifier()
        self.model_trained = False
        
    def load_training_data(self, file_path):
        """Load training data from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading training data: {e}")
            sys.exit(1)
            
    def train_model(self, data_path):
        """Train the model with provided data."""
        print("Loading training data...")
        df = self.load_training_data(data_path)
        
        print(f"Training model with {len(df)} examples...")
        self.classifier.train(df['text'].tolist(), df['topic'].tolist())
        self.model_trained = True
        print("Model training complete!")
        
    def classify_text(self, text):
        """Classify a single piece of text."""
        if not self.model_trained:
            print("Error: Model needs to be trained first!")
            return
            
        prediction = self.classifier.predict(text)
        probabilities = self.classifier.get_topic_probabilities(text)
        
        print("\nClassification Results:")
        print("-" * 50)
        print(f"Text: {text[:100]}...")
        print(f"Predicted Topic: {prediction}")
        print("\nTopic Probabilities:")
        for topic, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"{topic}: {prob:.2%}")
            
    def interactive_mode(self):
        """Run an interactive classification session."""
        if not self.model_trained:
            print("Error: Model needs to be trained first!")
            return
            
        print("\nEntering interactive mode (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text to classify: ")
            if text.lower() == 'quit':
                break
            self.classify_text(text)

def main():
    parser = argparse.ArgumentParser(description='Bible Text Classification Client')
    parser.add_argument('--train', type=str, help='Path to training data JSON file')
    parser.add_argument('--classify', type=str, help='Text to classify')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    client = BibleClassifierClient()
    
    # Example training data structure if no file is provided
    example_data = {
        'text': [
            "In the first year of Belshazzar king of Babylon, Daniel had a dream and visions.",
            "The ram which you saw, having the two horns—they are the kings of Media and Persia.",
            "Then I saw another beast coming up out of the earth.",
        ],
        'topic': [
            'Prophecy',
            'Interpretation',
            'Beasts'
        ]
    }
    
    # Train the model
    if args.train:
        client.train_model(args.train)
    else:
        print("Using example training data...")
        print("python bible_client.py --train data.json --classify \"And I saw a beast rise up out of the sea\"")
        print("python bible_client.py --train data.json --interactive")
        print("python bible_client.py --interactive")
        with open('example_training_data.json', 'w') as f:
            json.dump(example_data, f)
        client.train_model('example_training_data.json')
    
    # Handle classification
    if args.classify:
        client.classify_text(args.classify)
    elif args.interactive:
        client.interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()