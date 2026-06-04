"""
Word2Vec Skip-Gram (Simplified)
Learning word embeddings by predicting context words.
"""
import numpy as np
from collections import Counter

class SkipGram:
    def __init__(self, vocab_size, embedding_dim=50, lr=0.01):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.01

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def train_pair(self, center_idx, context_idx):
        """Train on a single (center, context) pair."""
        h = self.W_in[center_idx]  # (embedding_dim,)
        scores = h @ self.W_out    # (vocab_size,)
        probs = self._softmax(scores)

        # Gradient
        grad_out = probs.copy()
        grad_out[context_idx] -= 1  # (vocab_size,)

        # Update output weights
        self.W_out -= self.lr * np.outer(h, grad_out)

        # Update input weights
        grad_h = self.W_out @ grad_out  # (embedding_dim,)
        self.W_in[center_idx] -= self.lr * grad_h

        loss = -np.log(probs[context_idx] + 1e-8)
        return loss

    def get_embedding(self, word_idx):
        return self.W_in[word_idx]

    def most_similar(self, word_idx, top_k=5):
        target = self.W_in[word_idx]
        norms = np.linalg.norm(self.W_in, axis=1)
        similarities = (self.W_in @ target) / (norms * np.linalg.norm(target) + 1e-8)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        return [(idx, similarities[idx]) for idx in top_indices]


def build_training_pairs(tokens, word2idx, window=2):
    pairs = []
    for i, token in enumerate(tokens):
        center = word2idx.get(token)
        if center is None:
            continue
        for j in range(max(0, i-window), min(len(tokens), i+window+1)):
            if j != i:
                context = word2idx.get(tokens[j])
                if context is not None:
                    pairs.append((center, context))
    return pairs
