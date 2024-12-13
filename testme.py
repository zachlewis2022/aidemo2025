import pytest
import pandas as pd
import numpy as np
from bible_classifier import BibleTextClassifier
from bible_client import BibleClassifierClient
import json
import tempfile
from pathlib import Path

# Test data fixtures
@pytest.fixture
def sample_training_data():
    return {
        'text': [
            "In the first year of Belshazzar king of Babylon, Daniel had a dream",
            "The ram which you saw, having the two horns—they are the kings",
            "Then I saw another beast coming up out of the earth",
            "And the dragon was enraged with the woman",
            "I saw in the night visions, and behold, one like the Son of man"
        ],
        'topic': [
            'Prophecy',
            'Interpretation',
            'Beasts',
            'Spiritual Warfare',
            'Second Coming'
        ]
    }

@pytest.fixture
def trained_classifier(sample_training_data):
    classifier = BibleTextClassifier()
    classifier.train(sample_training_data['text'], sample_training_data['topic'])
    return classifier

@pytest.fixture
def temp_training_file(sample_training_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_training_data, f)
        return Path(f.name)

# Test BibleTextClassifier
class TestBibleTextClassifier:
    def test_initialization(self):
        classifier = BibleTextClassifier()
        assert hasattr(classifier, 'vectorizer')
        assert hasattr(classifier, 'classifier')
    
    def test_training(self, sample_training_data):
        classifier = BibleTextClassifier()
        classifier.train(sample_training_data['text'], sample_training_data['topic'])
        assert classifier.vectorizer is not None
        assert classifier.classifier is not None
    
    def test_prediction(self, trained_classifier):
        test_text = "And I saw a beast rise up out of the sea"
        prediction = trained_classifier.predict(test_text)
        assert isinstance(prediction, str)
        assert prediction in ['Prophecy', 'Interpretation', 'Beasts', 
                            'Spiritual Warfare', 'Second Coming']
    
    def test_topic_probabilities(self, trained_classifier):
        test_text = "And I saw a beast rise up out of the sea"
        probs = trained_classifier.get_topic_probabilities(test_text)
        assert isinstance(probs, dict)
        assert all(isinstance(v, float) for v in probs.values())
        assert pytest.approx(sum(probs.values())) == 1.0
    
    def test_empty_text_handling(self, trained_classifier):
        with pytest.raises(Exception):
            trained_classifier.predict("")
    
    def test_vectorizer_consistency(self, trained_classifier):
        text1 = "And I saw a beast"
        text2 = "And I saw a beast"
        assert trained_classifier.predict(text1) == trained_classifier.predict(text2)

# Test BibleClassifierClient
class TestBibleClassifierClient:
    def test_initialization(self):
        client = BibleClassifierClient()
        assert hasattr(client, 'classifier')
        assert not client.model_trained
    
    def test_load_training_data(self, temp_training_file):
        client = BibleClassifierClient()
        df = client.load_training_data(temp_training_file)
        assert isinstance(df, pd.DataFrame)
        assert 'text' in df.columns
        assert 'topic' in df.columns
    
    def test_train_model(self, temp_training_file):
        client = BibleClassifierClient()
        client.train_model(temp_training_file)
        assert client.model_trained
    
    def test_invalid_training_file(self):
        client = BibleClassifierClient()
        with pytest.raises(SystemExit):
            client.train_model('nonexistent_file.json')
    
    def test_classify_without_training(self):
        client = BibleClassifierClient()
        # Should print error message and return None
        assert client.classify_text("test text") is None
    
    @pytest.mark.parametrize("test_text", [
        "And I saw a beast rise up out of the sea",
        "The king saw a vision in the night",
        "The dragon pursued the woman"
    ])
    def test_multiple_classifications(self, test_text, temp_training_file):
        client = BibleClassifierClient()
        client.train_model(temp_training_file)
        result = client.classify_text(test_text)
        assert result is None  # classify_text prints results but doesn't return

# Test data validation
def test_data_validation(sample_training_data):
    assert len(sample_training_data['text']) == len(sample_training_data['topic'])
    assert all(isinstance(text, str) for text in sample_training_data['text'])
    assert all(isinstance(topic, str) for topic in sample_training_data['topic'])

# Test edge cases
@pytest.mark.parametrize("invalid_text", [
    "",
    " ",
    "   ",
    None
])
def test_invalid_inputs(trained_classifier, invalid_text):
    with pytest.raises(Exception):
        trained_classifier.predict(invalid_text)