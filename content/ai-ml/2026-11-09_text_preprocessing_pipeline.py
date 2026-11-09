"""
NLP Text Preprocessing Pipeline
Common text cleaning and feature extraction steps.
"""
import re
from collections import Counter

class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    @staticmethod
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def tokenize(text):
        return text.split()

    @staticmethod
    def remove_stopwords(tokens):
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'we', 'they', 'and', 'or', 'but', 'not', 'no', 'as'
        }
        return [t for t in tokens if t not in stopwords]

    def build_vocab(self, documents):
        all_tokens = []
        for doc in documents:
            tokens = self.tokenize(self.clean_text(doc))
            all_tokens.extend(tokens)
        freq = Counter(all_tokens)
        self.vocab = {word: idx for idx, (word, _) in enumerate(freq.most_common())}
        return self.vocab

    def bag_of_words(self, text):
        tokens = self.tokenize(self.clean_text(text))
        vector = [0] * len(self.vocab)
        for token in tokens:
            if token in self.vocab:
                vector[self.vocab[token]] += 1
        return vector

    def ngrams(self, text, n=2):
        tokens = self.tokenize(self.clean_text(text))
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


if __name__ == "__main__":
    docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "NLP helps computers understand human language.",
    ]
    preprocessor = TextPreprocessor()
    vocab = preprocessor.build_vocab(docs)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Bigrams: {preprocessor.ngrams(docs[0])}")
