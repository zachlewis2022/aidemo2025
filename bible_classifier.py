import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

class BibleTextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = MultinomialNB()
        
    def train(self, texts, labels):
        """
        Train the classifier on biblical texts and their topics
        
        Args:
            texts (list): List of text passages
            labels (list): List of corresponding topic labels
        """
        # Convert texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Train the classifier
        self.classifier.fit(X, labels)
    
    def predict(self, text):
        """
        Predict the topic of a given text
        
        Args:
            text (str): Biblical text passage
            
        Returns:
            str: Predicted topic
        """
        # Transform text using the same vectorizer
        X = self.vectorizer.transform([text])
        
        # Make prediction
        return self.classifier.predict(X)[0]
    
    def get_topic_probabilities(self, text):
        """
        Get probability distribution across all topics
        
        Args:
            text (str): Biblical text passage
            
        Returns:
            dict: Topic probabilities
        """
        X = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X)[0]
        return dict(zip(self.classifier.classes_, probs))

# Example training data
example_data = {
    'text': [
        "In the first year of Belshazzar king of Babylon, Daniel had a dream and visions of his head while on his bed.",
        "The ram which you saw, having the two horns—they are the kings of Media and Persia.",
        "Then I saw another beast coming up out of the earth, and he had two horns like a lamb and spoke like a dragon.",
        "And the dragon was enraged with the woman, and he went to make war with the rest of her offspring.",
        "I saw in the night visions, and behold, one like the Son of man came with the clouds of heaven.",
        "Then I heard one saint speaking, and another saint said unto that certain saint which spake.",
        "And there appeared a great wonder in heaven; a woman clothed with the sun, and the moon under her feet.",
        "And the fourth kingdom shall be strong as iron: forasmuch as iron breaketh in pieces."
    ],
    'topic': [
        'Prophecy',
        'Interpretation',
        'Beasts',
        'Spiritual Warfare',
        'Second Coming',
        'Heavenly Communication',
        'Symbols',
        'Kingdoms'
    ]
}

def train_and_evaluate_model():
    # Convert example data to DataFrame
    df = pd.DataFrame(example_data)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['topic'], test_size=0.2, random_state=42
    )
    
    # Initialize and train classifier
    classifier = BibleTextClassifier()
    classifier.train(X_train, y_train)
    
    # Example prediction
    test_text = "And I saw a beast rise up out of the sea, having seven heads"
    prediction = classifier.predict(test_text)
    probabilities = classifier.get_topic_probabilities(test_text)
    
    return classifier, prediction, probabilities