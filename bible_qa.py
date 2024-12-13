import pandas as pd
from bible_classifier import BibleTextClassifier
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BibleQASystem:
    def __init__(self):
        self.classifier = BibleTextClassifier()
        self.knowledge_base = self._create_knowledge_base()
        
    def _create_knowledge_base(self):
        """Create a knowledge base of biblical passages and their explanations"""
        return {
            'prophecy': {
                'Daniel 7:13-14': {
                    'text': "I saw in the night visions, and behold, one like the Son of man came with the clouds of heaven...",
                    'explanation': "This prophecy refers to Jesus Christ receiving His kingdom and authority from God the Father.",
                    'keywords': ['son of man', 'clouds', 'kingdom', 'authority', 'vision']
                },
                'Revelation 13:1': {
                    'text': "And I stood upon the sand of the sea, and saw a beast rise up out of the sea...",
                    'explanation': "This represents a political power rising from populated areas, as waters represent peoples and nations.",
                    'keywords': ['beast', 'sea', 'power', 'nations']
                }
            },
            'symbols': {
                'Daniel 8:3-4': {
                    'text': "Then I lifted up mine eyes, and saw, and, behold, there stood before the river a ram which had two horns...",
                    'explanation': "The ram represents the Medo-Persian empire, with its two horns symbolizing the two powers united.",
                    'keywords': ['ram', 'horns', 'empire', 'power']
                },
                'Revelation 12:1': {
                    'text': "And there appeared a great wonder in heaven; a woman clothed with the sun...",
                    'explanation': "The pure woman represents God's true church, with the sun symbolizing the gospel light.",
                    'keywords': ['woman', 'sun', 'church', 'pure']
                }
            }
        }
    
    def train(self, texts, topics):
        """Train the classifier with provided texts and topics"""
        self.classifier.train(texts, topics)
    
    def answer_question(self, question):
        """
        Answer a question about Daniel or Revelation
        
        Args:
            question (str): The question to answer
            
        Returns:
            dict: Answer information including relevant passages and explanations
        """
        # Classify the question topic
        topic = self.classifier.predict(question)
        topic_probs = self.classifier.get_topic_probabilities(question)
        
        # Find relevant passages from knowledge base
        relevant_info = self._find_relevant_passages(question, topic)
        
        # Construct the answer
        answer = {
            'topic': topic,
            'confidence': max(topic_probs.values()),
            'relevant_passages': relevant_info,
            'suggested_topics': self._suggest_related_topics(topic)
        }
        
        return answer
    
    def _find_relevant_passages(self, question, topic):
        """Find passages relevant to the question"""
        if topic.lower() in self.knowledge_base:
            passages = self.knowledge_base[topic.lower()]
            
            # Simple keyword matching (could be enhanced with better NLP)
            relevant_passages = []
            for passage_id, info in passages.items():
                if any(keyword in question.lower() for keyword in info['keywords']):
                    relevant_passages.append({
                        'reference': passage_id,
                        'text': info['text'],
                        'explanation': info['explanation']
                    })
            
            return relevant_passages
        return []
    
    def _suggest_related_topics(self, topic):
        """Suggest related topics for further study"""
        topic_relations = {
            'prophecy': ['symbols', 'interpretation', 'time prophecies'],
            'symbols': ['prophecy', 'beasts', 'interpretation'],
            'beasts': ['prophecy', 'kingdoms', 'symbols'],
            'interpretation': ['prophecy', 'symbols', 'application']
        }
        return topic_relations.get(topic.lower(), [])

def demonstrate_qa():
    # Create example training data
    training_data = {
        'text': [
            "What do the beasts in Daniel represent?",
            "Can you explain the symbols in Revelation?",
            "What is the meaning of the time prophecies?",
            "Who is the ram in Daniel 8?",
            "What does the woman in Revelation 12 represent?"
        ],
        'topic': [
            'beasts',
            'symbols',
            'prophecy',
            'interpretation',
            'symbols'
        ]
    }
    
    # Initialize and train the QA system
    qa_system = BibleQASystem()
    qa_system.train(training_data['text'], training_data['topic'])
    
    # Example questions
    questions = [
        "What does the ram with two horns represent in Daniel?",
        "Who is the woman clothed with the sun in Revelation 12?",
        "What is the meaning of the beast from the sea?"
    ]
    
    # Answer questions
    for question in questions:
        print("\nQuestion:", question)
        answer = qa_system.answer_question(question)
        print("\nTopic:", answer['topic'])
        print("Confidence: {:.2%}".format(answer['confidence']))
        print("\nRelevant Passages:")
        for passage in answer['relevant_passages']:
            print(f"\nReference: {passage['reference']}")
            print(f"Text: {passage['text']}")
            print(f"Explanation: {passage['explanation']}")
        print("\nSuggested Related Topics:", ', '.join(answer['suggested_topics']))
        print("-" * 80)

if __name__ == "__main__":
    demonstrate_qa()