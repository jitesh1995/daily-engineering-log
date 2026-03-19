#!/usr/bin/env python3
"""
Helper script to organize daily engineering notes into the right folders.
"""

from datetime import datetime as dt_class
import os

CONTENT_DIR = "content"

# ============================================================
# AI/ML SNIPPETS
# ============================================================
AI_ML_SNIPPETS = [
    {
        "filename": "logistic_regression_from_scratch.py",
        "content": '''"""
Logistic Regression from Scratch
Implementing binary classification with gradient descent.
"""
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = X @ self.weights + self.bias
            predictions = self._sigmoid(linear_pred)

            dw = (1 / n_samples) * (X.T @ (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = X @ self.weights + self.bias
        y_pred = self._sigmoid(linear_pred)
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, X):
        linear_pred = X @ self.weights + self.bias
        return self._sigmoid(linear_pred)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Accuracy: {accuracy:.4f}")
'''
    },
    {
        "filename": "knn_classifier.py",
        "content": '''"""
K-Nearest Neighbors Classifier
Distance-based classification with support for different metrics.
"""
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric

    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        elif self.metric == "cosine":
            return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        raise ValueError(f"Unknown metric: {self.metric}")

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = [self._distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    model = KNNClassifier(k=5, metric="euclidean")
    model.fit(X_train, y_train)

    X_test = np.random.randn(20, 2)
    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")
'''
    },
    {
        "filename": "neural_network_basics.py",
        "content": '''"""
Simple Feed-Forward Neural Network
Two-layer network with backpropagation for binary classification.
"""
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim=16, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = (1/m) * (self.a1.T @ dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_deriv(self.z1)
        dW1 = (1/m) * (X.T @ dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=500):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y.reshape(-1,1)*np.log(output+1e-8) +
                           (1-y.reshape(-1,1))*np.log(1-output+1e-8))
            self.backward(X, y)
            if epoch % 100 == 0:
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses
'''
    },
    {
        "filename": "text_preprocessing_pipeline.py",
        "content": '''"""
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
        text = re.sub(r'http\\S+|www\\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        text = re.sub(r'\\s+', ' ', text).strip()
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
'''
    },
    {
        "filename": "decision_tree_implementation.py",
        "content": '''"""
Decision Tree Classifier (CART Algorithm)
Binary splits using Gini impurity.
"""
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf prediction

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((c/n)**2 for c in counts.values())

    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return X[left_mask], X[~left_mask], y[left_mask], y[~left_mask]

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)
        n = len(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = parent_gini - (len(y_left)/n)*self._gini(y_left) - (len(y_right)/n)*self._gini(y_right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feature, threshold

        return best_feat, best_thresh

    def _build(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Node(value=Counter(y).most_common(1)[0][0])

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        X_left, X_right, y_left, y_right = self._split(X, y, feat, thresh)
        return Node(
            feature=feat, threshold=thresh,
            left=self._build(X_left, y_left, depth+1),
            right=self._build(X_right, y_right, depth+1),
        )

    def fit(self, X, y):
        self.root = self._build(np.array(X), np.array(y))

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.root) for x in np.array(X)]
'''
    },
    {
        "filename": "gradient_boosting_intuition.md",
        "content": '''# Gradient Boosting: Intuition & Key Concepts

## Core Idea
Gradient Boosting builds an ensemble of weak learners (usually decision trees)
**sequentially**, where each new tree corrects the errors of the combined ensemble so far.

## Algorithm Steps

1. **Initialize** with a constant prediction (e.g., mean of target)
2. **For each boosting round:**
   - Compute **pseudo-residuals** (negative gradient of loss)
   - Fit a new tree to these residuals
   - Add the new tree's predictions (scaled by learning rate) to the ensemble
3. **Final prediction** = sum of all trees' contributions

## Key Hyperparameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `n_estimators` | More trees = more capacity (risk overfit) | 100-1000 |
| `learning_rate` | Shrinks each tree's contribution | 0.01-0.3 |
| `max_depth` | Controls individual tree complexity | 3-8 |
| `subsample` | Row sampling per tree (reduces variance) | 0.7-1.0 |
| `colsample_bytree` | Feature sampling per tree | 0.7-1.0 |

## Why It Works

```
F_0(x) = argmin_c Σ L(y_i, c)           # initial constant
F_m(x) = F_{m-1}(x) + η * h_m(x)       # add correction tree

where h_m is fit to: r_im = -∂L(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)
```

## XGBoost vs LightGBM vs CatBoost

- **XGBoost**: Level-wise growth, L1/L2 regularization on leaf weights
- **LightGBM**: Leaf-wise growth (faster), histogram-based splitting
- **CatBoost**: Ordered boosting, native categorical feature handling

## Practical Tips

- Always tune `learning_rate` and `n_estimators` together
- Use early stopping on a validation set
- Feature importance from gain > from split count
- Monitor train vs val loss to detect overfitting
'''
    },
    {
        "filename": "cosine_similarity_search.py",
        "content": '''"""
Vector Similarity Search
Efficient cosine similarity for embedding-based retrieval (RAG pattern).
"""
import numpy as np

class VectorStore:
    """Simple in-memory vector store for similarity search."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []

    def add(self, vector, meta=None):
        vec = np.array(vector, dtype=np.float32)
        assert vec.shape == (self.dimension,), f"Expected dim {self.dimension}"
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm  # pre-normalize for cosine similarity
        self.vectors.append(vec)
        self.metadata.append(meta or {})

    def search(self, query_vector, top_k=5):
        query = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if not self.vectors:
            return []

        matrix = np.stack(self.vectors)
        similarities = matrix @ query  # cosine sim (pre-normalized)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx],
            })
        return results

    def batch_add(self, vectors, metadata_list=None):
        for i, vec in enumerate(vectors):
            meta = metadata_list[i] if metadata_list else None
            self.add(vec, meta)


if __name__ == "__main__":
    store = VectorStore(dimension=128)

    # Simulate adding document embeddings
    np.random.seed(42)
    for i in range(100):
        embedding = np.random.randn(128)
        store.add(embedding, {"doc_id": f"doc_{i}", "title": f"Document {i}"})

    # Search
    query = np.random.randn(128)
    results = store.search(query, top_k=3)
    for r in results:
        print(f"Score: {r[\'score\']:.4f} | {r[\'metadata\']}")
'''
    },
    {
        "filename": "cross_validation_strategies.py",
        "content": '''"""
Cross-Validation Strategies
Different CV approaches for model evaluation.
"""
import numpy as np

def k_fold_split(n_samples, k=5, shuffle=True, random_state=42):
    """Generate k-fold train/test indices."""
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    fold_size = n_samples // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        test_idx = indices[start:end]
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def stratified_k_fold(y, k=5, random_state=42):
    """Stratified K-Fold preserving class distribution."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}

    for c in classes:
        rng.shuffle(class_indices[c])

    folds = [[] for _ in range(k)]
    for c in classes:
        idxs = class_indices[c]
        fold_size = len(idxs) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(idxs)
            folds[i].extend(idxs[start:end])

    all_indices = np.arange(len(y))
    result = []
    for i in range(k):
        test_idx = np.array(folds[i])
        train_idx = np.setdiff1d(all_indices, test_idx)
        result.append((train_idx, test_idx))
    return result


def time_series_split(n_samples, n_splits=5, min_train_size=None):
    """Expanding window time series cross-validation."""
    min_train = min_train_size or (n_samples // (n_splits + 1))
    test_size = (n_samples - min_train) // n_splits

    folds = []
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_end = train_end + test_size
        if test_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        folds.append((train_idx, test_idx))
    return folds


if __name__ == "__main__":
    n = 100
    y = np.array([0]*60 + [1]*40)

    print("=== K-Fold ===")
    for i, (train, test) in enumerate(k_fold_split(n)):
        print(f"Fold {i+1}: train={len(train)}, test={len(test)}")

    print("\\n=== Stratified K-Fold ===")
    for i, (train, test) in enumerate(stratified_k_fold(y)):
        print(f"Fold {i+1}: train={len(train)}, test={len(test)}, "
              f"test_class_1_ratio={np.mean(y[test]):.2f}")

    print("\\n=== Time Series Split ===")
    for i, (train, test) in enumerate(time_series_split(n)):
        print(f"Split {i+1}: train=[0:{len(train)}], test=[{len(train)}:{len(train)+len(test)}]")
'''
    },
    {
        "filename": "attention_mechanism.py",
        "content": '''"""
Scaled Dot-Product Attention
Core building block of the Transformer architecture.
"""
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query matrix (batch, seq_len, d_k)
        K: Key matrix (batch, seq_len, d_k)
        V: Value matrix (batch, seq_len, d_v)
        mask: Optional mask (batch, seq_len, seq_len)
    Returns:
        output: Weighted values (batch, seq_len, d_v)
        weights: Attention weights (batch, seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax along last dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    output = np.matmul(weights, V)
    return output, weights


def multi_head_attention(Q, K, V, n_heads=4):
    """Split into multiple heads, apply attention, concatenate."""
    batch, seq_len, d_model = Q.shape
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    def split_heads(x):
        return x.reshape(batch, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

    Q_split = split_heads(Q)  # (batch, n_heads, seq_len, d_k)
    K_split = split_heads(K)
    V_split = split_heads(V)

    # Apply attention per head
    heads = []
    for h in range(n_heads):
        out, _ = scaled_dot_product_attention(
            Q_split[:, h], K_split[:, h], V_split[:, h]
        )
        heads.append(out)

    # Concatenate heads
    concat = np.concatenate(heads, axis=-1)  # (batch, seq_len, d_model)
    return concat


if __name__ == "__main__":
    batch, seq_len, d_model = 2, 8, 64
    Q = np.random.randn(batch, seq_len, d_model)
    K = np.random.randn(batch, seq_len, d_model)
    V = np.random.randn(batch, seq_len, d_model)

    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Attention output shape: {out.shape}")
    print(f"Attention weights shape: {weights.shape}")

    multi_out = multi_head_attention(Q, K, V, n_heads=4)
    print(f"Multi-head output shape: {multi_out.shape}")
'''
    },
    {
        "filename": "confusion_matrix_metrics.py",
        "content": '''"""
Classification Metrics from Scratch
Precision, Recall, F1, ROC-AUC without sklearn.
"""
import numpy as np

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true) | set(y_pred))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true]][class_to_idx[pred]] += 1
    return cm, classes

def precision_recall_f1(y_true, y_pred, positive_label=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

def roc_auc(y_true, y_scores):
    """Compute ROC-AUC using the trapezoidal rule."""
    pairs = sorted(zip(y_scores, y_true), reverse=True)
    tp, fp = 0, 0
    total_pos = sum(1 for _, y in pairs if y == 1)
    total_neg = len(pairs) - total_pos
    points = [(0.0, 0.0)]

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos if total_pos > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        points.append((fpr, tpr))

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(points)):
        x_diff = points[i][0] - points[i-1][0]
        y_avg = (points[i][1] + points[i-1][1]) / 2
        auc += x_diff * y_avg
    return auc


if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    y_scores = [0.9, 0.1, 0.8, 0.4, 0.3, 0.7, 0.6, 0.2, 0.85, 0.15]

    cm, classes = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\\n{cm}")

    metrics = precision_recall_f1(y_true, y_pred)
    print(f"Precision: {metrics[\'precision\']:.4f}")
    print(f"Recall:    {metrics[\'recall\']:.4f}")
    print(f"F1 Score:  {metrics[\'f1\']:.4f}")
    print(f"ROC-AUC:   {roc_auc(y_true, y_scores):.4f}")
'''
    },
    {
        "filename": "word_embeddings_word2vec.py",
        "content": '''"""
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
'''
    },
    {
        "filename": "kmeans_clustering.py",
        "content": '''"""
K-Means Clustering Implementation
Partitioning data into K clusters by minimizing within-cluster variance.
"""
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.RandomState(random_state)
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _init_centroids(self, X):
        """K-Means++ initialization for better convergence."""
        n_samples = X.shape[0]
        centroids = [X[self.rng.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            dists = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            probs = dists / dists.sum()
            idx = self.rng.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            # Assign clusters
            distances = np.array([
                np.sum((X - c)**2, axis=1) for c in self.centroids
            ]).T
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[self.labels == k].mean(axis=0)
                if np.sum(self.labels == k) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            shift = np.sum((new_centroids - self.centroids)**2)
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.inertia = sum(
            np.sum((X[self.labels == k] - self.centroids[k])**2)
            for k in range(self.n_clusters)
        )
        return self

    def predict(self, X):
        distances = np.array([
            np.sum((X - c)**2, axis=1) for c in self.centroids
        ]).T
        return np.argmin(distances, axis=1)


def elbow_method(X, max_k=10):
    """Find optimal K using the elbow method."""
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k)
        km.fit(X)
        inertias.append(km.inertia)
    return inertias


if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.randn(50, 2) + [2, 2],
        np.random.randn(50, 2) + [-2, -2],
        np.random.randn(50, 2) + [2, -2],
    ])

    km = KMeans(n_clusters=3)
    km.fit(X)
    print(f"Centroids:\\n{km.centroids}")
    print(f"Inertia: {km.inertia:.2f}")
    print(f"Cluster sizes: {[np.sum(km.labels == k) for k in range(3)]}")
'''
    },
    {
        "filename": "learning_rate_schedulers.py",
        "content": '''"""
Learning Rate Schedulers
Common scheduling strategies for training neural networks.
"""
import numpy as np

class StepDecay:
    def __init__(self, initial_lr=0.01, drop_factor=0.5, drop_every=10):
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        return self.initial_lr * (self.drop_factor ** (epoch // self.drop_every))

class CosineAnnealing:
    def __init__(self, initial_lr=0.01, T_max=100, eta_min=1e-6):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min

    def __call__(self, epoch):
        return self.eta_min + (self.initial_lr - self.eta_min) * \
               (1 + np.cos(np.pi * epoch / self.T_max)) / 2

class WarmupCosine:
    def __init__(self, initial_lr=0.01, warmup_epochs=10, total_epochs=100):
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        return self.initial_lr * (1 + np.cos(np.pi * progress)) / 2

class ExponentialDecay:
    def __init__(self, initial_lr=0.01, decay_rate=0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        return self.initial_lr * (self.decay_rate ** epoch)


if __name__ == "__main__":
    schedulers = {
        "Step Decay": StepDecay(),
        "Cosine Annealing": CosineAnnealing(),
        "Warmup + Cosine": WarmupCosine(),
        "Exponential Decay": ExponentialDecay(),
    }

    for name, scheduler in schedulers.items():
        lrs = [scheduler(e) for e in range(100)]
        print(f"{name}: start={lrs[0]:.6f}, mid={lrs[50]:.6f}, end={lrs[99]:.6f}")
'''
    },
    {
        "filename": "pca_dimensionality_reduction.py",
        "content": '''"""
Principal Component Analysis (PCA) from Scratch
Dimensionality reduction via eigendecomposition.
"""
import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_var
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components.T + self.mean


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(200, 10)
    X[:, 1] = X[:, 0] * 2 + np.random.randn(200) * 0.1  # correlated features

    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X)

    print(f"Original shape: {X.shape}")
    print(f"Reduced shape:  {X_reduced.shape}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio}")
    print(f"Total explained: {sum(pca.explained_variance_ratio):.4f}")
'''
    },
    {
        "filename": "rag_chunking_strategies.md",
        "content": '''# RAG Chunking Strategies

## Why Chunking Matters
In Retrieval-Augmented Generation, how you split documents into chunks
directly impacts retrieval quality. Bad chunks = bad context = bad answers.

## Common Strategies

### 1. Fixed-Size Chunking
```python
def fixed_size_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```
**Pros**: Simple, predictable size
**Cons**: Cuts mid-sentence, loses semantic boundaries

### 2. Sentence-Based Chunking
Split on sentence boundaries, group N sentences per chunk.
**Pros**: Preserves complete thoughts
**Cons**: Variable chunk sizes, may split related paragraphs

### 3. Recursive Character Splitting
Try splitting by paragraphs first, then sentences, then words.
```python
separators = ["\\n\\n", "\\n", ". ", " ", ""]
# Try each separator in order until chunks are small enough
```

### 4. Semantic Chunking
Use embeddings to detect topic boundaries.
- Embed each sentence
- Compute cosine similarity between consecutive sentences
- Split where similarity drops below threshold

### 5. Document-Aware Chunking
Leverage document structure (headers, sections, lists).
Best for structured docs like documentation, papers, legal text.

## Best Practices

- **Overlap**: 10-20% overlap between chunks prevents losing context at boundaries
- **Metadata**: Attach source, page number, section title to each chunk
- **Size**: 200-500 tokens is a good starting range for most models
- **Evaluation**: Always measure retrieval recall on your actual queries
- **Hybrid**: Combine structural + semantic splitting for best results
'''
    },
    {
        "filename": "model_evaluation_bias_variance.md",
        "content": '''# Bias-Variance Tradeoff

## The Core Concept

Total Error = Bias² + Variance + Irreducible Noise

| | Low Bias | High Bias |
|---|---|---|
| **Low Variance** | Sweet spot | Underfitting |
| **High Variance** | Overfitting | Worst case |

## Diagnosing the Problem

### High Bias (Underfitting)
- Training error is high
- Training and validation errors are close
- Model is too simple for the data

**Fix**: More features, more complex model, less regularization

### High Variance (Overfitting)
- Training error is low
- Large gap between training and validation error
- Model memorizes training data

**Fix**: More data, regularization, simpler model, dropout, early stopping

## Learning Curves

```
Error
  |   \\                    ← Validation error
  |    \\___________
  |                        ← Gap = variance
  |    ___________
  |   /                    ← Training error
  |_________________________
        Training set size
```

Large gap → high variance → need more data or regularization
Both high → high bias → need more complex model

## Regularization Techniques

- **L1 (Lasso)**: Drives weights to zero → feature selection
- **L2 (Ridge)**: Shrinks weights → prevents any single feature from dominating
- **Elastic Net**: Combines L1 + L2
- **Dropout**: Randomly zero neurons during training (neural nets)
- **Early Stopping**: Stop training when validation error increases
- **Data Augmentation**: Artificially increase training set diversity
'''
    },
]

# ============================================================
# DEVOPS SNIPPETS
# ============================================================
DEVOPS_SNIPPETS = [
    {
        "filename": "multistage_docker_python.Dockerfile",
        "content": '''# Multi-Stage Docker Build for Python Application
# Reduces final image size by separating build and runtime stages

# === Stage 1: Builder ===
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# === Stage 2: Runtime ===
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Set PATH for user-installed packages
ENV PATH=/home/appuser/.local/bin:$PATH \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen(\'http://localhost:8000/health\')" || exit 1

CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:8000", "--workers", "4"]
'''
    },
    {
        "filename": "kubernetes_deployment.yaml",
        "content": '''# Production-grade Kubernetes Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: api-server-sa
      securityContext:
        runAsNonRoot: true
        fsGroup: 1000
      containers:
      - name: api
        image: registry.example.com/api-server:1.2.3
        ports:
        - containerPort: 8000
          protocol: TCP
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: LOG_LEVEL
          value: "info"
        resources:
          requests:
            cpu: 250m
            memory: 256Mi
          limits:
            cpu: "1"
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 20
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          failureThreshold: 30
          periodSeconds: 5
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfied: DoNotSchedule
        labelSelector:
          matchLabels:
            app: api-server
---
apiVersion: v1
kind: Service
metadata:
  name: api-server
  namespace: production
spec:
  type: ClusterIP
  selector:
    app: api-server
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
    },
    {
        "filename": "terraform_aws_vpc.tf",
        "content": '''# AWS VPC Module with Public/Private Subnets
# Production-ready networking foundation

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "project_name" {
  type    = string
  default = "myapp"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

variable "azs" {
  type    = list(string)
  default = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

locals {
  public_subnets  = [for i, az in var.azs : cidrsubnet(var.vpc_cidr, 8, i)]
  private_subnets = [for i, az in var.azs : cidrsubnet(var.vpc_cidr, 8, i + 10)]
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public" {
  count                   = length(var.azs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnets[count.index]
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-public-${var.azs[count.index]}" }
}

resource "aws_subnet" "private" {
  count             = length(var.azs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = local.private_subnets[count.index]
  availability_zone = var.azs[count.index]

  tags = { Name = "${var.project_name}-private-${var.azs[count.index]}" }
}

resource "aws_eip" "nat" {
  count  = 1
  domain = "vpc"
  tags   = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${var.project_name}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(var.azs)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(var.azs)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}
'''
    },
    {
        "filename": "prometheus_alerting_rules.yaml",
        "content": '''# Prometheus Alerting Rules
# Production monitoring and SLO-based alerts

groups:
  - name: application_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          /
          sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High 5xx error rate ({{ $value | humanizePercentage }})"
          description: "Error rate exceeds 5% for the last 5 minutes."
          runbook_url: "https://wiki.internal/runbooks/high-error-rate"

      - alert: HighLatencyP99
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 2.0
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "P99 latency above 2s ({{ $value | humanizeDuration }})"

      - alert: PodCrashLooping
        expr: |
          rate(kube_pod_container_status_restarts_total[15m]) * 60 * 15 > 3
        for: 5m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping"

  - name: infrastructure_alerts
    rules:
      - alert: NodeHighCPU
        expr: |
          100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Node {{ $labels.instance }} CPU > 85% for 15m"

      - alert: DiskSpaceLow
        expr: |
          (node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"}
          / node_filesystem_size_bytes) * 100 < 15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Disk space < 15% on {{ $labels.instance }}:{{ $labels.mountpoint }}"

      - alert: NodeMemoryPressure
        expr: |
          (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Memory usage > 90% on {{ $labels.instance }}"

  - name: slo_alerts
    rules:
      - alert: SLOBudgetBurning
        expr: |
          1 - (
            sum(rate(http_requests_total{status!~"5.."}[1h]))
            /
            sum(rate(http_requests_total[1h]))
          ) > (1 - 0.999) * 14.4
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "SLO error budget burning too fast (14.4x normal rate)"
'''
    },
    {
        "filename": "github_actions_ci_cd.yaml",
        "content": '''# GitHub Actions CI/CD Pipeline
# Build, test, scan, and deploy

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint with ruff
        run: ruff check .

      - name: Type check with mypy
        run: mypy src/ --strict

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml --cov-fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: fs
          severity: CRITICAL,HIGH

      - name: Check for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified

  build-and-push:
    runs-on: ubuntu-latest
    needs: [lint-and-test, security-scan]
    if: github.ref == \'refs/heads/main\'
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == \'refs/heads/main\'
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/api-server \\
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \\
            --namespace production
          kubectl rollout status deployment/api-server --namespace production --timeout=300s
'''
    },
    {
        "filename": "nginx_reverse_proxy.conf",
        "content": '''# Nginx Reverse Proxy Configuration
# Production-ready with security headers, rate limiting, caching

upstream api_backend {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    keepalive 32;
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/s;
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=5r/m;

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    # TLS Configuration
    ssl_certificate     /etc/ssl/certs/api.example.com.pem;
    ssl_certificate_key /etc/ssl/private/api.example.com.key;
    ssl_protocols       TLSv1.3 TLSv1.2;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;

    # Security Headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src \'self\'" always;

    # Logging
    access_log /var/log/nginx/api_access.log combined buffer=16k flush=5s;
    error_log  /var/log/nginx/api_error.log warn;

    # Gzip
    gzip on;
    gzip_types application/json text/plain application/javascript;
    gzip_min_length 256;

    # API routes
    location /api/ {
        limit_req zone=api_limit burst=50 nodelay;
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
        proxy_send_timeout 10s;
    }

    location /api/auth/login {
        limit_req zone=login_limit burst=3 nodelay;
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Health check (no rate limit)
    location /health {
        proxy_pass http://api_backend;
        access_log off;
    }

    location / {
        return 404;
    }
}
'''
    },
    {
        "filename": "docker_compose_dev_stack.yaml",
        "content": '''# Docker Compose Development Stack
# Full local development environment with hot reload

version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./tests:/app/tests
    environment:
      - DATABASE_URL=postgresql://dev:devpass@postgres:5432/appdb
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - LOG_LEVEL=debug
      - RELOAD=true
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: devpass
      POSTGRES_DB: appdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dev -d appdb"]
      interval: 5s
      timeout: 3s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: celery -A tasks worker -l info -c 2
    volumes:
      - ./src:/app/src
    environment:
      - DATABASE_URL=postgresql://dev:devpass@postgres:5432/appdb
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
    depends_on:
      - postgres
      - redis

  mailhog:
    image: mailhog/mailhog
    ports:
      - "1025:1025"
      - "8025:8025"

volumes:
  postgres_data:
'''
    },
    {
        "filename": "helm_chart_values.yaml",
        "content": '''# Helm Chart Values - Production Configuration
# Comprehensive values for a production API deployment

replicaCount: 3

image:
  repository: registry.example.com/api-server
  tag: "1.2.3"
  pullPolicy: IfNotPresent

imagePullSecrets:
  - name: registry-credentials

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789:role/api-server

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  vault.hashicorp.com/agent-inject: "true"
  vault.hashicorp.com/role: "api-server"

podSecurityContext:
  runAsNonRoot: true
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault

containerSecurityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL

resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: "1"
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilization: 70
  targetMemoryUtilization: 80

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: api-tls
      hosts:
        - api.example.com

livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 15
  periodSeconds: 20

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

env:
  - name: LOG_LEVEL
    value: "info"
  - name: DATABASE_URL
    valueFrom:
      secretKeyRef:
        name: db-credentials
        key: url

nodeSelector:
  workload-type: api

tolerations:
  - key: "dedicated"
    operator: "Equal"
    value: "api"
    effect: "NoSchedule"

topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfied: DoNotSchedule
'''
    },
    {
        "filename": "terraform_s3_cloudfront.tf",
        "content": '''# Static Site Hosting with S3 + CloudFront
# Includes OAC, custom domain, and security headers

variable "domain_name" {
  type    = string
  default = "app.example.com"
}

variable "zone_id" {
  type = string
}

resource "aws_s3_bucket" "site" {
  bucket = var.domain_name
}

resource "aws_s3_bucket_versioning" "site" {
  bucket = aws_s3_bucket.site.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "site" {
  bucket = aws_s3_bucket.site.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.domain_name}-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_response_headers_policy" "security" {
  name = "${replace(var.domain_name, ".", "-")}-security"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000
      include_subdomains         = true
      override                   = true
    }
    content_type_options {
      override = true
    }
    frame_options {
      frame_option = "DENY"
      override     = true
    }
    xss_protection {
      mode_block = true
      protection = true
      override   = true
    }
  }
}

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  default_root_object = "index.html"
  aliases             = [var.domain_name]
  price_class         = "PriceClass_100"

  origin {
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_id                = "s3-origin"
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  default_cache_behavior {
    allowed_methods            = ["GET", "HEAD"]
    cached_methods             = ["GET", "HEAD"]
    target_origin_id           = "s3-origin"
    viewer_protocol_policy     = "redirect-to-https"
    compress                   = true
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security.id

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # SPA fallback
  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.site.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
}

resource "aws_acm_certificate" "site" {
  domain_name       = var.domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "site" {
  zone_id = var.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.site.domain_name
    zone_id                = aws_cloudfront_distribution.site.hosted_zone_id
    evaluate_target_health = false
  }
}
'''
    },
    {
        "filename": "security_hardening_checklist.md",
        "content": '''# Production Security Hardening Checklist

## Container Security
- [ ] Use minimal base images (distroless/alpine)
- [ ] Run as non-root user
- [ ] Read-only root filesystem
- [ ] Drop all capabilities, add only needed
- [ ] Scan images for CVEs (Trivy, Snyk)
- [ ] Pin image versions (no `latest` tag)
- [ ] Use multi-stage builds
- [ ] No secrets in image layers

## Kubernetes Security
- [ ] Pod Security Standards (restricted profile)
- [ ] Network Policies (default deny, explicit allow)
- [ ] RBAC with least privilege
- [ ] Service mesh mTLS (Istio/Linkerd)
- [ ] Secrets encrypted at rest (KMS)
- [ ] Audit logging enabled
- [ ] Resource quotas and limits
- [ ] Pod disruption budgets

## Application Security
- [ ] Input validation on all endpoints
- [ ] Parameterized queries (no SQL injection)
- [ ] CORS properly configured
- [ ] Rate limiting on auth endpoints
- [ ] JWT with short expiry + refresh tokens
- [ ] HTTPS only (HSTS enabled)
- [ ] Security headers (CSP, X-Frame-Options, etc.)
- [ ] Dependency scanning (Dependabot, Renovate)

## Infrastructure Security
- [ ] VPC with private subnets for workloads
- [ ] Security groups: least privilege
- [ ] WAF in front of public endpoints
- [ ] CloudTrail / audit logging
- [ ] Automated backups with encryption
- [ ] SSH key rotation
- [ ] MFA on all admin accounts

## Monitoring & Response
- [ ] Centralized logging (ELK/Loki)
- [ ] Alert on anomalous patterns
- [ ] Incident response runbooks
- [ ] Regular penetration testing
- [ ] Automated compliance checks (OPA/Kyverno)
'''
    },
    {
        "filename": "makefile_project_automation",
        "content": '''# Project Automation Makefile
# Common development tasks automated

.PHONY: help install dev test lint format build deploy clean

PYTHON := python3
PIP := pip
DOCKER := docker
COMPOSE := docker compose

# Default
help: ## Show this help message
\t@grep -E \'^[a-zA-Z_-]+:.*?## .*$$\' $(MAKEFILE_LIST) | \\
\t\tawk \'BEGIN {FS = ":.*?## "}; {printf "\\033[36m%-20s\\033[0m %s\\n", $$1, $$2}\'

# === Development ===

install: ## Install production dependencies
\t$(PIP) install -r requirements.txt

dev: ## Install dev dependencies and pre-commit hooks
\t$(PIP) install -r requirements.txt -r requirements-dev.txt
\tpre-commit install

test: ## Run tests with coverage
\tpytest tests/ -v --cov=src --cov-report=html --cov-fail-under=80
\t@echo "Coverage report: htmlcov/index.html"

test-watch: ## Run tests in watch mode
\tptw -- tests/ -v

lint: ## Run all linters
\truff check src/ tests/
\tmypy src/ --strict
\t@echo "All linting passed!"

format: ## Format code
\truff format src/ tests/
\truff check --fix src/ tests/

# === Docker ===

build: ## Build Docker image
\t$(DOCKER) build -t myapp:latest .
\t$(DOCKER) build -t myapp:$$(git rev-parse --short HEAD) .

up: ## Start development stack
\t$(COMPOSE) up -d
\t@echo "Stack is up. API: http://localhost:8000"

down: ## Stop development stack
\t$(COMPOSE) down

logs: ## Tail logs from all services
\t$(COMPOSE) logs -f --tail=50

# === Database ===

db-migrate: ## Run database migrations
\talembic upgrade head

db-rollback: ## Rollback last migration
\talembic downgrade -1

db-seed: ## Seed database with test data
\t$(PYTHON) scripts/seed_db.py

db-reset: ## Reset database (destructive!)
\t@echo "WARNING: This will destroy all data!"
\t@read -p "Continue? [y/N] " confirm && [ $$confirm = y ]
\t$(COMPOSE) exec postgres psql -U dev -d appdb -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
\talembic upgrade head
\t$(PYTHON) scripts/seed_db.py

# === Deploy ===

deploy-staging: ## Deploy to staging
\t@echo "Deploying to staging..."
\tkubectl config use-context staging
\thelm upgrade --install myapp charts/myapp -f values/staging.yaml

deploy-prod: ## Deploy to production
\t@echo "Deploying to production..."
\t@read -p "Confirm production deploy? [y/N] " confirm && [ $$confirm = y ]
\tkubectl config use-context production
\thelm upgrade --install myapp charts/myapp -f values/production.yaml

# === Cleanup ===

clean: ## Clean generated files
\trm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
\tfind . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
\t@echo "Cleaned!"
'''
    },
    {
        "filename": "terraform_rds_postgres.tf",
        "content": '''# AWS RDS PostgreSQL with Read Replicas
# Production database with Multi-AZ, encryption, and monitoring

variable "db_name" {
  type    = string
  default = "appdb"
}

variable "db_username" {
  type      = string
  sensitive = true
}

variable "vpc_id" {
  type = string
}

variable "private_subnet_ids" {
  type = list(string)
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.db_name}-subnet-group"
  subnet_ids = var.private_subnet_ids
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.db_name}-rds-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]
    description     = "PostgreSQL access from application"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_rds_cluster" "main" {
  cluster_identifier     = var.db_name
  engine                 = "aurora-postgresql"
  engine_version         = "15.4"
  database_name          = var.db_name
  master_username        = var.db_username
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  storage_encrypted = true
  kms_key_id        = aws_kms_key.rds.arn

  backup_retention_period      = 30
  preferred_backup_window      = "03:00-04:00"
  preferred_maintenance_window = "sun:04:00-sun:05:00"

  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.db_name}-final-snapshot"

  enabled_cloudwatch_logs_exports = ["postgresql"]

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 16
  }
}

resource "aws_rds_cluster_instance" "writer" {
  identifier         = "${var.db_name}-writer"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.main.engine

  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.rds.arn

  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
}

resource "aws_rds_cluster_instance" "reader" {
  count              = 2
  identifier         = "${var.db_name}-reader-${count.index}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.main.engine

  performance_insights_enabled = true
}

resource "aws_kms_key" "rds" {
  description         = "KMS key for RDS encryption"
  enable_key_rotation = true
}

output "cluster_endpoint" {
  value     = aws_rds_cluster.main.endpoint
  sensitive = true
}

output "reader_endpoint" {
  value     = aws_rds_cluster.main.reader_endpoint
  sensitive = true
}
'''
    },
    {
        "filename": "github_actions_docker_build.yaml",
        "content": '''# Optimized Docker Build with GitHub Actions
# Multi-platform builds with caching

name: Docker Build & Publish

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

permissions:
  contents: read
  packages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        platform: [linux/amd64, linux/arm64]

    steps:
      - uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: ${{ matrix.platform }}
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          provenance: true
          sbom: true

      - name: Scan image for vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ghcr.io/${{ github.repository }}:sha-${{ github.sha }}
          format: sarif
          output: trivy-results.sarif
          severity: CRITICAL,HIGH

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
'''
    },
    {
        "filename": "blue_green_deployment.md",
        "content": '''# Blue-Green Deployment Strategy

## Overview
Blue-Green deployment maintains two identical production environments.
Only one (say "Blue") serves live traffic at any time. You deploy to
the idle environment ("Green"), test it, then switch traffic instantly.

## Architecture

```
                     ┌─────────────┐
                     │  Load       │
                     │  Balancer   │
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐            ┌─────▼─────┐
        │   BLUE    │            │   GREEN   │
        │  (live)   │            │  (idle)   │
        │  v1.2.3   │            │  v1.2.4   │
        └───────────┘            └───────────┘
```

## Steps

1. **Deploy** new version to idle environment (Green)
2. **Smoke test** Green environment via internal URL
3. **Run integration tests** against Green
4. **Switch** load balancer to point to Green
5. **Monitor** error rates, latency, logs for 15 min
6. **Rollback** (if needed): switch back to Blue instantly

## Kubernetes Implementation

```yaml
# Blue deployment (currently live)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
  labels:
    app: myapp
    slot: blue
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: myapp:1.2.3

---
# Service targeting active slot
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    slot: blue  # ← Change to "green" to switch
```

## Rollback

Rollback is instant — just switch the selector back:
```bash
kubectl patch svc myapp -p \'{"spec":{"selector":{"slot":"blue"}}}\'
```

## Tradeoffs

| Pros | Cons |
|------|------|
| Instant rollback | 2x infrastructure cost |
| Zero downtime | Database migrations need care |
| Full production testing | More complex networking |
| Simple mental model | Stateful services are harder |
'''
    },
    {
        "filename": "observability_stack.md",
        "content": '''# Observability Stack Design

## Three Pillars

### 1. Metrics (Prometheus + Grafana)
Quantitative measurements over time.
- **RED Method** (request-scoped): Rate, Errors, Duration
- **USE Method** (resource-scoped): Utilization, Saturation, Errors

Key metrics to always have:
- Request rate (rpm)
- Error rate (5xx / total)
- Latency percentiles (p50, p95, p99)
- CPU/Memory utilization
- Queue depth / processing lag
- Database connection pool usage

### 2. Logs (Loki / ELK / CloudWatch)
Structured events with context.

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "error",
  "service": "api-server",
  "trace_id": "abc123",
  "message": "Failed to process payment",
  "error": "timeout after 5s",
  "user_id": "usr_456",
  "amount": 99.99
}
```

Best practices:
- Always structured (JSON)
- Include trace_id for correlation
- Log at appropriate levels
- Never log secrets or PII

### 3. Traces (Jaeger / Tempo / X-Ray)
Distributed request flow across services.

```
[API Gateway] ──► [Auth Service] ──► [User DB]
      │
      └──► [Order Service] ──► [Payment Service] ──► [Stripe]
                  │
                  └──► [Inventory DB]
```

## SLI/SLO Framework

- **SLI**: What you measure (e.g., "% of requests < 200ms")
- **SLO**: Target for the SLI (e.g., "99.9% of requests < 200ms")
- **Error Budget**: 100% - SLO = budget for risky changes

## Alerting Philosophy

1. Alert on symptoms, not causes
2. Every alert must be actionable
3. Link to runbook in annotation
4. Page only for customer-facing impact
5. Use multi-window burn rate for SLO alerts
'''
    },
]

# ============================================================
# DATA ENGINEERING SNIPPETS
# ============================================================
DATA_ENGINEERING_SNIPPETS = [
    {
        "filename": "sql_window_functions.sql",
        "content": '''-- SQL Window Functions Reference
-- Common patterns for analytics queries

-- 1. Running Total
SELECT
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date) AS running_total,
    SUM(revenue) OVER (
        PARTITION BY EXTRACT(YEAR FROM date)
        ORDER BY date
    ) AS ytd_revenue
FROM daily_revenue;

-- 2. Ranking with Ties
SELECT
    product_name,
    category,
    sales,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY sales DESC) AS row_num,
    RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS rank,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY sales DESC) AS dense_rank
FROM product_sales;

-- 3. Moving Average (7-day)
SELECT
    date,
    daily_active_users,
    AVG(daily_active_users) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM user_metrics;

-- 4. Lead/Lag - Period over Period
SELECT
    month,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY month) AS prev_month_revenue,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY month))
        / LAG(revenue, 1) OVER (ORDER BY month) * 100,
        2
    ) AS mom_growth_pct
FROM monthly_revenue;

-- 5. First/Last Value
SELECT
    user_id,
    event_date,
    event_type,
    FIRST_VALUE(event_type) OVER (
        PARTITION BY user_id ORDER BY event_date
    ) AS first_action,
    LAST_VALUE(event_type) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_action
FROM user_events;

-- 6. Percentile / Distribution
SELECT
    department,
    employee_name,
    salary,
    PERCENT_RANK() OVER (PARTITION BY department ORDER BY salary) AS percentile,
    NTILE(4) OVER (PARTITION BY department ORDER BY salary) AS salary_quartile
FROM employees;

-- 7. Gap Detection (session boundaries)
SELECT
    user_id,
    event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event,
    EXTRACT(EPOCH FROM (
        event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time)
    )) / 60 AS minutes_since_last,
    CASE
        WHEN EXTRACT(EPOCH FROM (
            event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time)
        )) / 60 > 30 THEN 1
        ELSE 0
    END AS new_session_flag
FROM clickstream;
'''
    },
    {
        "filename": "airflow_dag_etl.py",
        "content": '''"""
Airflow DAG: Daily ETL Pipeline
Extract from API, transform, load to warehouse.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.task_group import TaskGroup

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-alerts@company.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


def extract_api_data(**context):
    """Extract data from source API."""
    import requests

    execution_date = context["ds"]
    url = f"https://api.source.com/events?date={execution_date}"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Push to XCom for downstream tasks
    context["ti"].xcom_push(key="raw_data", value=data)
    context["ti"].xcom_push(key="record_count", value=len(data))
    return len(data)


def validate_data(**context):
    """Data quality checks on extracted data."""
    data = context["ti"].xcom_pull(key="raw_data", task_ids="extract.extract_api")
    record_count = len(data)

    # Assertions
    assert record_count > 0, "No records extracted!"
    assert all("id" in record for record in data), "Missing 'id' field"
    assert all("timestamp" in record for record in data), "Missing 'timestamp'"

    # Check for duplicates
    ids = [r["id"] for r in data]
    duplicates = len(ids) - len(set(ids))
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate IDs found")

    return {"records": record_count, "duplicates": duplicates}


def transform_data(**context):
    """Clean and transform raw data."""
    data = context["ti"].xcom_pull(key="raw_data", task_ids="extract.extract_api")

    transformed = []
    for record in data:
        transformed.append({
            "event_id": record["id"],
            "event_type": record.get("type", "unknown").lower(),
            "user_id": record.get("user_id"),
            "amount": float(record.get("amount", 0)),
            "event_timestamp": record["timestamp"],
            "processed_at": dt_class.utcnow().isoformat(),
        })

    # Deduplicate
    seen = set()
    deduplicated = []
    for row in transformed:
        if row["event_id"] not in seen:
            seen.add(row["event_id"])
            deduplicated.append(row)

    context["ti"].xcom_push(key="transformed_data", value=deduplicated)
    return len(deduplicated)


def load_to_warehouse(**context):
    """Load transformed data to PostgreSQL warehouse."""
    data = context["ti"].xcom_pull(
        key="transformed_data", task_ids="transform.transform_data"
    )

    hook = PostgresHook(postgres_conn_id="warehouse")
    conn = hook.get_conn()
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO events (event_id, event_type, user_id, amount, event_timestamp, processed_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (event_id) DO NOTHING
    """

    rows = [
        (r["event_id"], r["event_type"], r["user_id"],
         r["amount"], r["event_timestamp"], r["processed_at"])
        for r in data
    ]

    cursor.executemany(insert_sql, rows)
    conn.commit()
    cursor.close()
    conn.close()

    return f"Loaded {len(rows)} records"


with DAG(
    dag_id="daily_etl_pipeline",
    default_args=default_args,
    description="Daily ETL: API -> Transform -> Warehouse",
    schedule_interval="0 6 * * *",  # 6 AM UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["etl", "daily", "production"],
) as dag:

    start = EmptyOperator(task_id="start")

    with TaskGroup("extract") as extract_group:
        extract_task = PythonOperator(
            task_id="extract_api",
            python_callable=extract_api_data,
        )
        validate_task = PythonOperator(
            task_id="validate",
            python_callable=validate_data,
        )
        extract_task >> validate_task

    with TaskGroup("transform") as transform_group:
        transform_task = PythonOperator(
            task_id="transform_data",
            python_callable=transform_data,
        )

    with TaskGroup("load") as load_group:
        load_task = PythonOperator(
            task_id="load_warehouse",
            python_callable=load_to_warehouse,
        )

    end = EmptyOperator(task_id="end")

    start >> extract_group >> transform_group >> load_group >> end
'''
    },
    {
        "filename": "spark_data_transformations.py",
        "content": '''"""
PySpark Data Transformation Patterns
Common transformations for data engineering pipelines.
"""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType


def create_spark_session(app_name="DataTransformations"):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def deduplicate_events(df, id_col="event_id", timestamp_col="event_time"):
    """Keep the latest version of each event (SCD Type 1)."""
    window = Window.partitionBy(id_col).orderBy(F.col(timestamp_col).desc())
    return (
        df
        .withColumn("row_num", F.row_number().over(window))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )


def sessionize_events(df, user_col="user_id", time_col="event_time", gap_minutes=30):
    """Assign session IDs based on inactivity gaps."""
    window = Window.partitionBy(user_col).orderBy(time_col)

    return (
        df
        .withColumn("prev_time", F.lag(time_col).over(window))
        .withColumn(
            "gap_minutes",
            (F.unix_timestamp(time_col) - F.unix_timestamp("prev_time")) / 60,
        )
        .withColumn(
            "new_session",
            F.when(
                (F.col("gap_minutes") > gap_minutes) | F.col("prev_time").isNull(), 1
            ).otherwise(0),
        )
        .withColumn(
            "session_id",
            F.concat(
                F.col(user_col),
                F.lit("_"),
                F.sum("new_session").over(window),
            ),
        )
        .drop("prev_time", "gap_minutes", "new_session")
    )


def compute_rfm(df, user_col="user_id", date_col="order_date", amount_col="amount"):
    """Compute RFM (Recency, Frequency, Monetary) scores."""
    max_date = df.agg(F.max(date_col)).collect()[0][0]

    rfm = (
        df.groupBy(user_col)
        .agg(
            F.datediff(F.lit(max_date), F.max(date_col)).alias("recency"),
            F.countDistinct("order_id").alias("frequency"),
            F.sum(amount_col).alias("monetary"),
        )
    )

    # Score each metric 1-5
    for col_name in ["recency", "frequency", "monetary"]:
        ascending = col_name == "recency"  # lower recency = better
        rfm = rfm.withColumn(
            f"{col_name}_score",
            F.ntile(5).over(
                Window.orderBy(
                    F.col(col_name).asc() if ascending else F.col(col_name).desc()
                )
            ),
        )

    return rfm.withColumn(
        "rfm_segment",
        F.concat(
            F.col("recency_score"), F.col("frequency_score"), F.col("monetary_score")
        ),
    )


def pivot_metrics(df, group_col, pivot_col, value_col):
    """Dynamic pivot table for metrics."""
    pivot_values = [row[0] for row in df.select(pivot_col).distinct().collect()]

    return (
        df.groupBy(group_col)
        .pivot(pivot_col, pivot_values)
        .agg(F.sum(value_col))
        .fillna(0)
    )


def scd_type2_merge(current_df, new_df, key_cols, value_cols):
    """Slowly Changing Dimension Type 2 merge logic."""
    # Find changed records
    join_condition = [current_df[k] == new_df[k] for k in key_cols]
    value_condition = F.lit(False)
    for v in value_cols:
        value_condition = value_condition | (current_df[v] != new_df[v])

    changed = (
        current_df
        .join(new_df, join_condition, "inner")
        .filter(value_condition & (current_df["is_current"] == True))
        .select(current_df["*"])
    )

    # Close old records
    closed = changed.withColumn("is_current", F.lit(False)).withColumn(
        "valid_to", F.current_timestamp()
    )

    # New records from changes
    new_records = (
        new_df
        .join(changed.select(key_cols), key_cols, "inner")
        .withColumn("is_current", F.lit(True))
        .withColumn("valid_from", F.current_timestamp())
        .withColumn("valid_to", F.lit(None).cast(TimestampType()))
    )

    # Unchanged records
    unchanged = current_df.join(changed.select(key_cols), key_cols, "left_anti")

    # Truly new records (not in current at all)
    truly_new = (
        new_df
        .join(current_df.select(key_cols), key_cols, "left_anti")
        .withColumn("is_current", F.lit(True))
        .withColumn("valid_from", F.current_timestamp())
        .withColumn("valid_to", F.lit(None).cast(TimestampType()))
    )

    return unchanged.unionByName(closed).unionByName(new_records).unionByName(truly_new)
'''
    },
    {
        "filename": "dbt_model_patterns.sql",
        "content": '''-- dbt Model Patterns
-- Common data modeling patterns for analytics engineering

-- ============================================================
-- 1. STAGING MODEL (stg_orders.sql)
-- Light transformations, renaming, type casting
-- ============================================================

/*
{{ config(
    materialized=\'view\',
    schema=\'staging\'
) }}
*/

WITH source AS (
    SELECT * FROM /* {{ source(\'raw\', \'orders\') }} */ raw.orders
),

renamed AS (
    SELECT
        id AS order_id,
        user_id,
        LOWER(status) AS order_status,
        CAST(amount AS DECIMAL(10, 2)) AS order_amount,
        CAST(created_at AS TIMESTAMP) AS ordered_at,
        CAST(updated_at AS TIMESTAMP) AS updated_at,
        _loaded_at AS _etl_loaded_at
    FROM source
    WHERE id IS NOT NULL
)

SELECT * FROM renamed;


-- ============================================================
-- 2. INTERMEDIATE MODEL (int_order_items_enriched.sql)
-- Join and enrich, one clear purpose
-- ============================================================

/*
{{ config(materialized=\'ephemeral\') }}
*/

WITH orders AS (
    SELECT * FROM /* {{ ref(\'stg_orders\') }} */ staging.stg_orders
),

items AS (
    SELECT * FROM /* {{ ref(\'stg_order_items\') }} */ staging.stg_order_items
),

products AS (
    SELECT * FROM /* {{ ref(\'stg_products\') }} */ staging.stg_products
)

SELECT
    orders.order_id,
    orders.user_id,
    orders.ordered_at,
    items.product_id,
    products.product_name,
    products.category,
    items.quantity,
    items.unit_price,
    items.quantity * items.unit_price AS line_total
FROM orders
INNER JOIN items ON orders.order_id = items.order_id
INNER JOIN products ON items.product_id = products.product_id;


-- ============================================================
-- 3. MART MODEL (fct_daily_revenue.sql)
-- Business-level fact table
-- ============================================================

/*
{{ config(
    materialized=\'incremental\',
    unique_key=\'revenue_date\',
    schema=\'marts\'
) }}
*/

WITH order_items AS (
    SELECT * FROM /* {{ ref(\'int_order_items_enriched\') }} */ intermediate.int_order_items_enriched
    /*
    {% if is_incremental() %}
    WHERE ordered_at > (SELECT MAX(revenue_date) FROM {{ this }})
    {% endif %}
    */
)

SELECT
    DATE(ordered_at) AS revenue_date,
    category,
    COUNT(DISTINCT order_id) AS total_orders,
    SUM(quantity) AS total_units,
    SUM(line_total) AS gross_revenue,
    AVG(line_total) AS avg_order_value,
    COUNT(DISTINCT user_id) AS unique_customers
FROM order_items
GROUP BY DATE(ordered_at), category;


-- ============================================================
-- 4. DATA QUALITY TESTS (schema.yml style, as SQL)
-- ============================================================

-- Test: No null order_ids
SELECT order_id
FROM staging.stg_orders
WHERE order_id IS NULL;
-- Expected: 0 rows

-- Test: Referential integrity
SELECT oi.order_id
FROM staging.stg_order_items oi
LEFT JOIN staging.stg_orders o ON oi.order_id = o.order_id
WHERE o.order_id IS NULL;
-- Expected: 0 rows

-- Test: Accepted values
SELECT order_status
FROM staging.stg_orders
WHERE order_status NOT IN (\'pending\', \'processing\', \'shipped\', \'delivered\', \'cancelled\');
-- Expected: 0 rows
'''
    },
    {
        "filename": "data_quality_framework.py",
        "content": '''"""
Data Quality Framework
Reusable checks for data pipeline validation.
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional
from datetime import datetime
import json

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    metric_value: Any
    threshold: Any
    severity: str  # "critical", "warning", "info"
    message: str
    timestamp: str = ""

    def __post_init__(self):
        self.timestamp = dt_class.utcnow().isoformat()


class DataQualityChecker:
    """Framework for running data quality checks."""

    def __init__(self):
        self.results: list[CheckResult] = []

    def check_not_null(self, df, columns, severity="critical"):
        """Check that specified columns have no nulls."""
        for col in columns:
            null_count = df[col].isna().sum()
            total = len(df)
            self.results.append(CheckResult(
                check_name=f"not_null_{col}",
                passed=null_count == 0,
                metric_value=null_count,
                threshold=0,
                severity=severity,
                message=f"{col}: {null_count}/{total} nulls found",
            ))

    def check_unique(self, df, columns, severity="critical"):
        """Check that columns form a unique key."""
        key = columns if isinstance(columns, list) else [columns]
        duplicates = df.duplicated(subset=key).sum()
        self.results.append(CheckResult(
            check_name=f"unique_{'_'.join(key)}",
            passed=duplicates == 0,
            metric_value=duplicates,
            threshold=0,
            severity=severity,
            message=f"{'_'.join(key)}: {duplicates} duplicate rows",
        ))

    def check_accepted_values(self, df, column, accepted, severity="warning"):
        """Check that column values are within accepted set."""
        invalid = df[~df[column].isin(accepted)][column].unique()
        self.results.append(CheckResult(
            check_name=f"accepted_values_{column}",
            passed=len(invalid) == 0,
            metric_value=list(invalid),
            threshold=accepted,
            severity=severity,
            message=f"{column}: invalid values {list(invalid)[:5]}",
        ))

    def check_freshness(self, df, timestamp_col, max_hours=24, severity="critical"):
        """Check that data is recent enough."""
        max_ts = df[timestamp_col].max()
        age_hours = (dt_class.utcnow() - max_ts).total_seconds() / 3600
        self.results.append(CheckResult(
            check_name=f"freshness_{timestamp_col}",
            passed=age_hours <= max_hours,
            metric_value=round(age_hours, 2),
            threshold=max_hours,
            severity=severity,
            message=f"Data is {age_hours:.1f}h old (max: {max_hours}h)",
        ))

    def check_row_count(self, df, min_rows=1, max_rows=None, severity="critical"):
        """Check that row count is within expected range."""
        count = len(df)
        passed = count >= min_rows and (max_rows is None or count <= max_rows)
        self.results.append(CheckResult(
            check_name="row_count",
            passed=passed,
            metric_value=count,
            threshold={"min": min_rows, "max": max_rows},
            severity=severity,
            message=f"Row count: {count} (expected: {min_rows}-{max_rows or 'inf'})",
        ))

    def check_referential_integrity(self, df, column, reference_df, ref_column, severity="critical"):
        """Check foreign key relationship."""
        ref_values = set(reference_df[ref_column])
        orphans = df[~df[column].isin(ref_values)][column].nunique()
        self.results.append(CheckResult(
            check_name=f"ref_integrity_{column}",
            passed=orphans == 0,
            metric_value=orphans,
            threshold=0,
            severity=severity,
            message=f"{column}: {orphans} orphan values not in {ref_column}",
        ))

    def summary(self):
        """Return summary of all check results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        critical_failures = sum(
            1 for r in self.results if not r.passed and r.severity == "critical"
        )
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "critical_failures": critical_failures,
            "all_passed": failed == 0,
            "results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                }
                for r in self.results
            ],
        }
'''
    },
    {
        "filename": "kafka_consumer_pattern.py",
        "content": '''"""
Kafka Consumer Pattern
Reliable message consumption with error handling and offset management.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsumerConfig:
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "my-consumer-group"
    topic: str = "events"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 30000


class ReliableConsumer:
    """
    Kafka consumer with:
    - Manual offset commits (at-least-once)
    - Dead letter queue for failed messages
    - Exponential backoff on failures
    - Graceful shutdown
    """

    def __init__(self, config: ConsumerConfig, process_fn: Callable):
        self.config = config
        self.process_fn = process_fn
        self.running = False
        self.stats = {"processed": 0, "failed": 0, "dlq": 0}

    def _deserialize(self, raw_message):
        """Deserialize message value from bytes."""
        try:
            return json.loads(raw_message.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Deserialization failed: {e}")
            return None

    def _send_to_dlq(self, message, error):
        """Send failed message to dead letter queue."""
        dlq_record = {
            "original_topic": self.config.topic,
            "original_message": message,
            "error": str(error),
            "timestamp": time.time(),
            "consumer_group": self.config.group_id,
        }
        logger.warning(f"Sending to DLQ: {dlq_record}")
        self.stats["dlq"] += 1
        # In production: produce to {topic}.dlq topic

    def _process_batch(self, messages):
        """Process a batch of messages with error handling."""
        for msg in messages:
            data = self._deserialize(msg.value)
            if data is None:
                self._send_to_dlq(msg.value, "Deserialization failed")
                continue

            retries = 0
            max_retries = 3

            while retries <= max_retries:
                try:
                    self.process_fn(data)
                    self.stats["processed"] += 1
                    break
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Max retries exceeded for message: {e}"
                        )
                        self._send_to_dlq(data, e)
                        self.stats["failed"] += 1
                    else:
                        wait = min(2 ** retries, 30)
                        logger.warning(
                            f"Retry {retries}/{max_retries} in {wait}s: {e}"
                        )
                        time.sleep(wait)

    def run(self):
        """Main consumer loop (pseudo-code for Kafka client)."""
        logger.info(f"Starting consumer for topic: {self.config.topic}")
        self.running = True

        # In production, use confluent_kafka or aiokafka
        # consumer = KafkaConsumer(...)

        try:
            while self.running:
                # messages = consumer.poll(timeout_ms=1000, max_records=config.max_poll_records)
                # self._process_batch(messages)
                # consumer.commit()  # Manual commit after processing
                logger.info(f"Stats: {self.stats}")
                time.sleep(1)  # Placeholder for poll
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            # consumer.close()
            logger.info(f"Final stats: {self.stats}")

    def stop(self):
        self.running = False
'''
    },
    {
        "filename": "data_modeling_patterns.md",
        "content": '''# Data Modeling Patterns

## Star Schema
The classic dimensional model for analytics.

```
                    ┌──────────────┐
                    │  dim_date    │
                    └──────┬───────┘
                           │
┌──────────────┐   ┌──────┴───────┐   ┌──────────────┐
│ dim_customer │───│  fct_orders  │───│ dim_product  │
└──────────────┘   └──────┬───────┘   └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  dim_store   │
                    └──────────────┘
```

### Fact Tables
- Contain measurable events (orders, clicks, payments)
- Foreign keys to dimensions
- Additive metrics (amount, quantity, duration)
- Grain: one row per event

### Dimension Tables
- Descriptive attributes (name, category, location)
- Slowly changing (SCD Type 1, 2, or 3)
- Typically wide (many columns)
- Relatively small row count

## Slowly Changing Dimensions (SCD)

### Type 1: Overwrite
Simply update the value. No history preserved.
Use for: corrections, non-critical attributes.

### Type 2: Add New Row
Add new row with version tracking (valid_from, valid_to, is_current).
Use for: important attributes where history matters (price, status).

### Type 3: Add New Column
Add column for previous value (current_city, previous_city).
Use for: when you only need one level of history.

## One Big Table (OBT)
Pre-join everything for fast queries. Denormalized.
Trade storage for query speed.

### When to Use OBT
- Small-to-medium data volumes
- Simple, repetitive queries
- Self-serve analytics (no joins needed)
- Dashboard backing tables

## Activity Schema
For event-heavy workloads (product analytics, clickstream).

```sql
CREATE TABLE activity_stream (
    activity_id    BIGINT,
    entity_id      VARCHAR,  -- user, device, etc.
    activity_type  VARCHAR,  -- page_view, click, purchase
    occurred_at    TIMESTAMP,
    properties     JSONB     -- flexible payload
);
```

## Data Vault
For enterprise data warehousing at scale.

Components:
- **Hubs**: Business keys (customer_id, product_id)
- **Links**: Relationships between hubs
- **Satellites**: Descriptive attributes with history

Best for: large enterprises, multiple source systems, full audit trail.
'''
    },
    {
        "filename": "sql_performance_optimization.md",
        "content": '''# SQL Performance Optimization Guide

## Index Strategy

### When to Index
- Columns in WHERE clauses
- JOIN columns
- ORDER BY columns
- High-cardinality columns

### When NOT to Index
- Small tables (< 1000 rows)
- Columns with low cardinality (boolean)
- Columns that are frequently updated
- Wide columns (long text)

### Composite Index Rules
Order matters! (a, b, c) index helps:
- WHERE a = 1
- WHERE a = 1 AND b = 2
- WHERE a = 1 AND b = 2 AND c = 3

Does NOT help:
- WHERE b = 2 (leftmost prefix missing)
- WHERE b = 2 AND c = 3

## Query Anti-Patterns

### 1. SELECT *
```sql
-- Bad: fetches all columns, prevents covering index usage
SELECT * FROM orders WHERE status = \'pending\';

-- Good: fetch only what you need
SELECT order_id, amount FROM orders WHERE status = \'pending\';
```

### 2. Functions on Indexed Columns
```sql
-- Bad: prevents index usage
WHERE YEAR(created_at) = 2024

-- Good: range scan uses index
WHERE created_at >= \'2024-01-01\' AND created_at < \'2025-01-01\'
```

### 3. Implicit Type Casting
```sql
-- Bad: user_id is INT but compared to STRING
WHERE user_id = \'12345\'

-- Good: matching types
WHERE user_id = 12345
```

### 4. Correlated Subqueries
```sql
-- Bad: executes subquery per row
SELECT * FROM orders o
WHERE amount > (SELECT AVG(amount) FROM orders WHERE user_id = o.user_id);

-- Good: use window function or CTE
WITH user_avg AS (
    SELECT user_id, AVG(amount) AS avg_amount FROM orders GROUP BY user_id
)
SELECT o.* FROM orders o
JOIN user_avg ua ON o.user_id = ua.user_id
WHERE o.amount > ua.avg_amount;
```

## EXPLAIN ANALYZE Checklist
1. Look for Seq Scan on large tables → add index
2. Check estimated vs actual rows → update statistics
3. Look for Nested Loop on large joins → consider Hash Join
4. Watch for Sort operations → add index or increase work_mem
5. Check for high I/O → consider partitioning
'''
    },
    {
        "filename": "cdc_change_data_capture.py",
        "content": '''"""
Change Data Capture (CDC) Pattern
Track and process database changes incrementally.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum
import json


class ChangeType(str, Enum):
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass
class ChangeEvent:
    table: str
    change_type: ChangeType
    timestamp: datetime
    primary_key: dict
    before: Optional[dict]  # None for INSERT
    after: Optional[dict]   # None for DELETE
    metadata: dict

    def to_dict(self):
        return {
            "table": self.table,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "primary_key": self.primary_key,
            "before": self.before,
            "after": self.after,
            "metadata": self.metadata,
        }


class CDCProcessor:
    """Process CDC events and apply them to a target."""

    def __init__(self):
        self.stats = {"inserts": 0, "updates": 0, "deletes": 0, "errors": 0}
        self.handlers = {}

    def register_handler(self, table: str, handler):
        """Register a handler function for a specific table."""
        self.handlers[table] = handler

    def process_event(self, event: ChangeEvent):
        """Route and process a single CDC event."""
        handler = self.handlers.get(event.table)

        if handler is None:
            return  # No handler registered for this table

        try:
            if event.change_type == ChangeType.INSERT:
                handler.handle_insert(event.primary_key, event.after)
                self.stats["inserts"] += 1

            elif event.change_type == ChangeType.UPDATE:
                changed_fields = self._detect_changes(event.before, event.after)
                handler.handle_update(event.primary_key, event.before, event.after, changed_fields)
                self.stats["updates"] += 1

            elif event.change_type == ChangeType.DELETE:
                handler.handle_delete(event.primary_key, event.before)
                self.stats["deletes"] += 1

        except Exception as e:
            self.stats["errors"] += 1
            raise

    @staticmethod
    def _detect_changes(before: dict, after: dict) -> list[str]:
        """Identify which fields changed between before and after."""
        if not before or not after:
            return list(after.keys()) if after else []

        changed = []
        all_keys = set(before.keys()) | set(after.keys())
        for key in all_keys:
            if before.get(key) != after.get(key):
                changed.append(key)
        return changed

    def process_batch(self, events: list[ChangeEvent]):
        """Process a batch of CDC events in order."""
        for event in sorted(events, key=lambda e: e.timestamp):
            self.process_event(event)

    def get_stats(self):
        return {**self.stats, "total": sum(self.stats.values())}


class TimestampCDC:
    """Pull-based CDC using timestamp columns."""

    def __init__(self, table, timestamp_column="updated_at"):
        self.table = table
        self.timestamp_column = timestamp_column
        self.last_processed = None

    def build_query(self):
        """Build incremental extraction query."""
        if self.last_processed:
            return (
                f"SELECT * FROM {self.table} "
                f"WHERE {self.timestamp_column} > \'{self.last_processed.isoformat()}\' "
                f"ORDER BY {self.timestamp_column}"
            )
        return f"SELECT * FROM {self.table} ORDER BY {self.timestamp_column}"

    def update_watermark(self, max_timestamp):
        """Update the high watermark after successful processing."""
        self.last_processed = max_timestamp
'''
    },
    {
        "filename": "schema_evolution_strategies.md",
        "content": '''# Schema Evolution Strategies

## The Problem
Data schemas change over time. Columns are added, types change,
fields get renamed. Pipelines must handle this gracefully.

## Approaches

### 1. Backward Compatible Changes (Safe)
- Adding new nullable columns
- Adding columns with defaults
- Widening numeric types (INT → BIGINT)
- Adding new enum values

### 2. Breaking Changes (Dangerous)
- Removing columns
- Renaming columns
- Changing column types (STRING → INT)
- Changing nullability (NULL → NOT NULL)

## Strategies

### Schema Registry (Avro/Protobuf)
```
Producer → Schema Registry → Consumer
              ↓
    Compatibility Check
    (BACKWARD, FORWARD, FULL)
```

- **BACKWARD**: New schema can read old data
- **FORWARD**: Old schema can read new data
- **FULL**: Both directions compatible

### Schema-on-Read (Data Lake)
Store raw data, apply schema at query time.

```python
# Read with schema evolution enabled
df = spark.read \\
    .option("mergeSchema", "true") \\
    .parquet("s3://datalake/events/")
```

### Migration-Based (RDBMS)
Sequential, versioned migrations.

```sql
-- V001_add_email_column.sql
ALTER TABLE users ADD COLUMN email VARCHAR(255);
UPDATE users SET email = CONCAT(username, \'@legacy.com\') WHERE email IS NULL;
ALTER TABLE users ALTER COLUMN email SET NOT NULL;

-- V002_rename_column.sql
ALTER TABLE users RENAME COLUMN name TO full_name;
```

## Best Practices

1. **Never delete columns immediately** — deprecate first, remove later
2. **Version your schemas** — track changes in a registry or migration tool
3. **Test with production-like data** before deploying schema changes
4. **Use FULL compatibility mode** when possible for streaming
5. **Document breaking changes** and coordinate with consumers
6. **Backfill nulls** before adding NOT NULL constraints
'''
    },
    {
        "filename": "data_lake_partitioning.md",
        "content": '''# Data Lake Partitioning Guide

## Why Partition?
Partitioning enables **partition pruning** — skipping irrelevant files
during queries. A well-partitioned table can be 100x faster to query.

## Common Partition Strategies

### Time-Based (Most Common)
```
s3://datalake/events/
  ├── year=2024/
  │   ├── month=01/
  │   │   ├── day=01/
  │   │   │   └── part-00000.parquet
  │   │   └── day=02/
  │   └── month=02/
  └── year=2025/
```

Best for: time-series data, logs, events, most analytical queries.

### Category-Based
```
s3://datalake/transactions/
  ├── region=us-east/
  │   ├── type=purchase/
  │   └── type=refund/
  └── region=eu-west/
```

Best for: data commonly filtered by category.

### Hybrid (Time + Category)
```
s3://datalake/orders/
  ├── date=2024-01-15/
  │   ├── country=US/
  │   └── country=UK/
  └── date=2024-01-16/
```

## Partition Column Selection Rules

| Criteria | Good Partition Column | Bad Partition Column |
|----------|----------------------|---------------------|
| Cardinality | Low-medium (< 10K values) | High (user_id, UUID) |
| Query filter | Almost always filtered on | Rarely used in WHERE |
| Distribution | Even across values | Heavily skewed |
| Growth | Predictable over time | Unpredictable |

## File Size Targets

- **Target**: 128 MB - 1 GB per file (Parquet)
- **Too small** (< 10 MB): Too many files, slow listing
- **Too large** (> 2 GB): Slow to read, poor parallelism

## Anti-Patterns

1. **Over-partitioning**: Partitioning by high-cardinality columns
   creates millions of tiny files ("small file problem")
2. **Under-partitioning**: One giant partition that always gets full-scanned
3. **Partitioning by non-filter columns**: No benefit if queries don't
   filter on the partition column

## Compaction
Periodically merge small files into larger ones:
```sql
-- Spark compaction example
df.repartition(10)  -- target ~10 output files
  .write.mode("overwrite")
  .parquet("s3://datalake/events/date=2024-01-15/")
```
'''
    },
    {
        "filename": "great_expectations_suite.py",
        "content": '''"""
Data Quality with Great Expectations Pattern
Define and validate data contracts.
"""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Expectation:
    expectation_type: str
    kwargs: dict
    severity: str = "critical"

@dataclass
class ExpectationSuite:
    name: str
    expectations: list[Expectation] = field(default_factory=list)

    def expect_column_to_exist(self, column, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_to_exist",
            kwargs={"column": column},
            severity=severity,
        ))
        return self

    def expect_column_values_to_not_be_null(self, column, mostly=1.0, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": column, "mostly": mostly},
            severity=severity,
        ))
        return self

    def expect_column_values_to_be_between(self, column, min_value=None, max_value=None, severity="warning"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": column, "min_value": min_value, "max_value": max_value},
            severity=severity,
        ))
        return self

    def expect_column_values_to_be_in_set(self, column, value_set, severity="warning"):
        self.expectations.append(Expectation(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": column, "value_set": value_set},
            severity=severity,
        ))
        return self

    def expect_compound_columns_to_be_unique(self, column_list, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_compound_columns_to_be_unique",
            kwargs={"column_list": column_list},
            severity=severity,
        ))
        return self

    def expect_table_row_count_to_be_between(self, min_value, max_value=None, severity="critical"):
        self.expectations.append(Expectation(
            expectation_type="expect_table_row_count_to_be_between",
            kwargs={"min_value": min_value, "max_value": max_value},
            severity=severity,
        ))
        return self

    def to_dict(self):
        return {
            "expectation_suite_name": self.name,
            "expectations": [
                {"expectation_type": e.expectation_type, "kwargs": e.kwargs, "meta": {"severity": e.severity}}
                for e in self.expectations
            ],
        }


# === Example: Define suite for orders table ===
def build_orders_suite():
    suite = ExpectationSuite(name="orders_suite")

    suite.expect_table_row_count_to_be_between(min_value=1)

    # Primary key
    suite.expect_column_to_exist("order_id")
    suite.expect_column_values_to_not_be_null("order_id")

    # Required fields
    for col in ["user_id", "order_date", "total_amount", "status"]:
        suite.expect_column_values_to_not_be_null(col)

    # Value ranges
    suite.expect_column_values_to_be_between("total_amount", min_value=0, max_value=100000)

    # Allowed values
    suite.expect_column_values_to_be_in_set(
        "status", ["pending", "processing", "shipped", "delivered", "cancelled"]
    )

    # Composite uniqueness
    suite.expect_compound_columns_to_be_unique(["order_id", "order_date"])

    return suite


if __name__ == "__main__":
    import json
    suite = build_orders_suite()
    print(json.dumps(suite.to_dict(), indent=2))
'''
    },
]

# ============================================================
# SOFTWARE ENGINEERING SNIPPETS
# ============================================================
SOFTWARE_ENGINEERING_SNIPPETS = [
    {
        "filename": "binary_search_variants.py",
        "content": '''"""
Binary Search Variants
Common patterns beyond the basic binary search.
"""

def binary_search(arr, target):
    """Standard binary search. Returns index or -1."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(arr, target):
    """First position where arr[i] >= target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(arr, target):
    """First position where arr[i] > target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def search_rotated(arr, target):
    """Search in a rotated sorted array."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[lo] <= arr[mid]:
            if arr[lo] <= target < arr[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def find_peak(arr):
    """Find a peak element (greater than neighbors)."""
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[mid] < arr[mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search_answer(lo, hi, is_feasible):
    """
    Template for binary search on the answer space.
    Find minimum value where is_feasible(x) is True.
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if is_feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13, 15]
    print(f"Search 7: index={binary_search(arr, 7)}")
    print(f"Lower bound 6: index={lower_bound(arr, 6)}")
    print(f"Upper bound 7: index={upper_bound(arr, 7)}")

    rotated = [7, 9, 11, 13, 15, 1, 3, 5]
    print(f"Search rotated 3: index={search_rotated(rotated, 3)}")

    peaks = [1, 3, 7, 5, 2, 4, 1]
    print(f"Peak at index: {find_peak(peaks)}")
'''
    },
    {
        "filename": "design_patterns_strategy.py",
        "content": '''"""
Strategy Pattern
Define a family of algorithms, encapsulate each one, and make them
interchangeable at runtime.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


# === Strategy Interface ===
class PricingStrategy(ABC):
    @abstractmethod
    def calculate_price(self, base_price: float, quantity: int) -> float:
        pass


# === Concrete Strategies ===
class RegularPricing(PricingStrategy):
    def calculate_price(self, base_price: float, quantity: int) -> float:
        return base_price * quantity


class BulkDiscountPricing(PricingStrategy):
    def __init__(self, threshold: int = 10, discount_pct: float = 0.15):
        self.threshold = threshold
        self.discount_pct = discount_pct

    def calculate_price(self, base_price: float, quantity: int) -> float:
        total = base_price * quantity
        if quantity >= self.threshold:
            total *= (1 - self.discount_pct)
        return total


class TieredPricing(PricingStrategy):
    """Different price per unit based on quantity tiers."""

    def __init__(self):
        self.tiers = [
            (10, 1.0),    # 1-10 units: full price
            (50, 0.9),    # 11-50 units: 10% off
            (100, 0.8),   # 51-100 units: 20% off
            (float("inf"), 0.7),  # 101+: 30% off
        ]

    def calculate_price(self, base_price: float, quantity: int) -> float:
        total = 0.0
        remaining = quantity
        prev_limit = 0

        for limit, multiplier in self.tiers:
            tier_qty = min(remaining, int(limit) - prev_limit)
            if tier_qty <= 0:
                break
            total += base_price * multiplier * tier_qty
            remaining -= tier_qty
            prev_limit = int(limit)

        return total


class SeasonalPricing(PricingStrategy):
    def __init__(self, season_multiplier: float = 1.25):
        self.season_multiplier = season_multiplier

    def calculate_price(self, base_price: float, quantity: int) -> float:
        return base_price * self.season_multiplier * quantity


# === Context ===
@dataclass
class ShoppingCart:
    pricing_strategy: PricingStrategy

    def set_strategy(self, strategy: PricingStrategy):
        self.pricing_strategy = strategy

    def checkout(self, items: list[tuple[str, float, int]]) -> dict:
        line_items = []
        total = 0.0

        for name, base_price, quantity in items:
            price = self.pricing_strategy.calculate_price(base_price, quantity)
            line_items.append({
                "item": name,
                "base_price": base_price,
                "quantity": quantity,
                "total": round(price, 2),
            })
            total += price

        return {"items": line_items, "total": round(total, 2)}


if __name__ == "__main__":
    items = [
        ("Widget A", 10.00, 5),
        ("Widget B", 25.00, 15),
        ("Widget C", 8.00, 100),
    ]

    cart = ShoppingCart(pricing_strategy=RegularPricing())
    print("Regular:", cart.checkout(items)["total"])

    cart.set_strategy(BulkDiscountPricing(threshold=10, discount_pct=0.15))
    print("Bulk Discount:", cart.checkout(items)["total"])

    cart.set_strategy(TieredPricing())
    print("Tiered:", cart.checkout(items)["total"])
'''
    },
    {
        "filename": "graph_algorithms.py",
        "content": '''"""
Graph Algorithms
BFS, DFS, Dijkstra, Topological Sort.
"""
from collections import defaultdict, deque
import heapq

class Graph:
    def __init__(self, directed=True):
        self.adj = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def bfs(self, start):
        """Breadth-First Search. Returns visit order and distances."""
        visited = {start: 0}
        queue = deque([start])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)

        return order, visited

    def dfs(self, start):
        """Depth-First Search (iterative). Returns visit order."""
        visited = set()
        stack = [start]
        order = []

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for neighbor, _ in reversed(self.adj[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

        return order

    def dijkstra(self, start):
        """Shortest paths from start (non-negative weights)."""
        dist = {start: 0}
        prev = {start: None}
        heap = [(0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue

            for v, weight in self.adj[u]:
                new_dist = d + weight
                if new_dist < dist.get(v, float("inf")):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(heap, (new_dist, v))

        return dist, prev

    def topological_sort(self):
        """Kahn's algorithm for topological ordering (DAG only)."""
        in_degree = defaultdict(int)
        for u in self.adj:
            for v, _ in self.adj[u]:
                in_degree[v] += 1

        queue = deque([u for u in self.adj if in_degree[u] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.adj):
            raise ValueError("Graph has a cycle — no topological order exists")
        return order

    def shortest_path(self, start, end):
        """Reconstruct shortest path using Dijkstra."""
        dist, prev = self.dijkstra(start)
        if end not in dist:
            return None, float("inf")

        path = []
        node = end
        while node is not None:
            path.append(node)
            node = prev[node]
        return list(reversed(path)), dist[end]


if __name__ == "__main__":
    g = Graph(directed=False)
    g.add_edge("A", "B", 4)
    g.add_edge("A", "C", 2)
    g.add_edge("B", "D", 3)
    g.add_edge("C", "D", 1)
    g.add_edge("C", "E", 5)
    g.add_edge("D", "E", 2)

    print("BFS:", g.bfs("A")[0])
    print("DFS:", g.dfs("A"))

    path, cost = g.shortest_path("A", "E")
    print(f"Shortest A→E: {path} (cost: {cost})")

    # Topological sort (DAG)
    dag = Graph(directed=True)
    dag.add_edge("compile", "test")
    dag.add_edge("compile", "lint")
    dag.add_edge("test", "deploy")
    dag.add_edge("lint", "deploy")
    print(f"Build order: {dag.topological_sort()}")
'''
    },
    {
        "filename": "solid_principles.md",
        "content": '''# SOLID Principles in Practice

## S — Single Responsibility Principle
A class should have only one reason to change.

```python
# Bad: One class does everything
class UserManager:
    def create_user(self, data): ...
    def send_welcome_email(self, user): ...
    def generate_report(self, users): ...

# Good: Separate responsibilities
class UserService:
    def create_user(self, data): ...

class EmailService:
    def send_welcome_email(self, user): ...

class ReportGenerator:
    def generate_report(self, users): ...
```

## O — Open/Closed Principle
Open for extension, closed for modification.

```python
# Bad: Modifying existing code for new shapes
def area(shape):
    if shape.type == "circle":
        return 3.14 * shape.radius ** 2
    elif shape.type == "rectangle":  # Adding new type = modifying existing code
        return shape.width * shape.height

# Good: Extend via new classes
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Circle(Shape):
    def area(self): return 3.14 * self.radius ** 2

class Rectangle(Shape):  # New shape = new class, no modification
    def area(self): return self.width * self.height
```

## L — Liskov Substitution Principle
Subtypes must be substitutable for their base types.

```python
# Bad: Square breaks Rectangle contract
class Rectangle:
    def set_width(self, w): self.width = w
    def set_height(self, h): self.height = h

class Square(Rectangle):  # Violates LSP
    def set_width(self, w): self.width = self.height = w  # Surprise!

# Good: Separate types or use composition
class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...
```

## I — Interface Segregation Principle
No client should be forced to depend on methods it doesn't use.

```python
# Bad: Fat interface
class Worker(ABC):
    @abstractmethod
    def code(self): ...
    @abstractmethod
    def test(self): ...
    @abstractmethod
    def manage(self): ...  # Not all workers manage

# Good: Segregated interfaces
class Coder(ABC):
    @abstractmethod
    def code(self): ...

class Tester(ABC):
    @abstractmethod
    def test(self): ...

class Developer(Coder, Tester):  # Compose as needed
    def code(self): ...
    def test(self): ...
```

## D — Dependency Inversion Principle
Depend on abstractions, not concretions.

```python
# Bad: Direct dependency on concrete class
class OrderService:
    def __init__(self):
        self.db = PostgresDatabase()  # Tightly coupled

# Good: Depend on abstraction
class OrderService:
    def __init__(self, db: Database):  # Accepts any Database impl
        self.db = db
```
'''
    },
    {
        "filename": "lru_cache_implementation.py",
        "content": '''"""
LRU Cache Implementation
Using a doubly-linked list + hashmap for O(1) operations.
"""

class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    Least Recently Used Cache.
    get() and put() both run in O(1) time.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Sentinel nodes (avoid null checks)
        self.head = Node()  # Most recently used
        self.tail = Node()  # Least recently used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        """Remove node from doubly-linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: Node):
        """Add node right after head (most recent position)."""
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        # Move to front (most recently used)
        self._remove(node)
        self._add_to_front(node)
        return node.value

    def put(self, key: int, value: int):
        if key in self.cache:
            # Update existing
            self._remove(self.cache[key])
            del self.cache[key]

        # Add new node
        node = Node(key, value)
        self._add_to_front(node)
        self.cache[key] = node

        # Evict if over capacity
        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        items = []
        node = self.head.next
        while node != self.tail:
            items.append(f"{node.key}:{node.value}")
            node = node.next
        return f"LRUCache([{', '.join(items)}])"


if __name__ == "__main__":
    cache = LRUCache(3)

    cache.put(1, "a")
    cache.put(2, "b")
    cache.put(3, "c")
    print(cache)  # [3:c, 2:b, 1:a]

    cache.get(1)   # Access 1, moves to front
    print(cache)  # [1:a, 3:c, 2:b]

    cache.put(4, "d")  # Evicts key 2 (least recent)
    print(cache)  # [4:d, 1:a, 3:c]

    print(f"Get 2: {cache.get(2)}")  # -1 (evicted)
    print(f"Get 3: {cache.get(3)}")  # c
'''
    },
    {
        "filename": "concurrency_patterns.py",
        "content": '''"""
Concurrency Patterns in Python
Thread pool, async/await, producer-consumer.
"""
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from dataclasses import dataclass
from typing import Callable


# === 1. Thread Pool Pattern ===
def thread_pool_example():
    """Process multiple tasks concurrently with a thread pool."""

    def fetch_url(url):
        time.sleep(0.5)  # Simulate I/O
        return f"Response from {url}"

    urls = [f"https://api.example.com/page/{i}" for i in range(10)]

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_url, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error fetching {url}: {e}")

    return results


# === 2. Async/Await Pattern ===
async def async_example():
    """Async I/O for high-concurrency scenarios."""

    async def fetch_data(session_id: int):
        await asyncio.sleep(0.3)  # Simulate async I/O
        return {"session": session_id, "data": f"result_{session_id}"}

    # Run many concurrent requests
    tasks = [fetch_data(i) for i in range(20)]
    results = await asyncio.gather(*tasks)
    return results


# === 3. Producer-Consumer Pattern ===
@dataclass
class Task:
    task_id: int
    payload: str


def producer_consumer_example():
    """Classic producer-consumer with a bounded queue."""
    queue = Queue(maxsize=10)
    results = []
    lock = threading.Lock()

    def producer(n_tasks):
        for i in range(n_tasks):
            task = Task(task_id=i, payload=f"data_{i}")
            queue.put(task)  # Blocks if queue is full
        # Poison pills to signal consumers to stop
        for _ in range(3):
            queue.put(None)

    def consumer(consumer_id):
        while True:
            task = queue.get()  # Blocks if queue is empty
            if task is None:
                break
            # Process task
            result = f"Consumer-{consumer_id} processed task-{task.task_id}"
            with lock:
                results.append(result)
            queue.task_done()

    # Start producer and consumers
    producer_thread = threading.Thread(target=producer, args=(20,))
    consumer_threads = [
        threading.Thread(target=consumer, args=(i,)) for i in range(3)
    ]

    producer_thread.start()
    for t in consumer_threads:
        t.start()

    producer_thread.join()
    for t in consumer_threads:
        t.join()

    return results


# === 4. Rate Limiter (Token Bucket) ===
class TokenBucketRateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # Tokens per second
        self.capacity = capacity  # Max tokens
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def wait_and_acquire(self):
        while not self.acquire():
            time.sleep(0.01)


if __name__ == "__main__":
    # Thread pool
    results = thread_pool_example()
    print(f"Thread pool: {len(results)} results")

    # Async
    results = asyncio.run(async_example())
    print(f"Async: {len(results)} results")

    # Producer-Consumer
    results = producer_consumer_example()
    print(f"Producer-Consumer: {len(results)} results")

    # Rate limiter
    limiter = TokenBucketRateLimiter(rate=5, capacity=5)
    allowed = sum(1 for _ in range(10) if limiter.acquire())
    print(f"Rate limiter: {allowed}/10 requests allowed")
'''
    },
    {
        "filename": "api_design_best_practices.md",
        "content": '''# REST API Design Best Practices

## URL Structure

```
GET    /api/v1/users           # List users
POST   /api/v1/users           # Create user
GET    /api/v1/users/{id}      # Get user
PATCH  /api/v1/users/{id}      # Update user
DELETE /api/v1/users/{id}      # Delete user

# Sub-resources
GET    /api/v1/users/{id}/orders
POST   /api/v1/users/{id}/orders

# Filtering, sorting, pagination
GET    /api/v1/orders?status=pending&sort=-created_at&page=2&limit=25
```

## Naming Conventions
- Use **nouns**, not verbs: `/users` not `/getUsers`
- Use **plural**: `/users` not `/user`
- Use **kebab-case**: `/order-items` not `/orderItems`
- Max 2 levels of nesting: `/users/{id}/orders` (not deeper)

## HTTP Status Codes

| Code | Meaning | When to Use |
|------|---------|-------------|
| 200 | OK | Successful GET, PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Validation error |
| 401 | Unauthorized | Missing/invalid auth |
| 403 | Forbidden | Authenticated but not allowed |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate, version conflict |
| 422 | Unprocessable | Valid syntax, invalid semantics |
| 429 | Too Many Requests | Rate limited |
| 500 | Server Error | Unexpected failure |

## Response Format

```json
{
  "data": {
    "id": "usr_123",
    "email": "user@example.com",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "meta": {
    "request_id": "req_abc123"
  }
}
```

## Pagination

```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 25,
    "total": 150,
    "has_more": true
  }
}
```

## Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {"field": "email", "message": "Must be a valid email address"},
      {"field": "age", "message": "Must be at least 18"}
    ]
  }
}
```

## Versioning
- URL path: `/api/v1/users` (most common, simplest)
- Header: `Accept: application/vnd.api.v1+json`
- Query: `/api/users?version=1`

## Rate Limiting Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1705312800
Retry-After: 30
```

## Idempotency
- GET, PUT, DELETE are naturally idempotent
- For POST: use `Idempotency-Key` header
- Store and return cached response for duplicate keys
'''
    },
    {
        "filename": "testing_patterns.py",
        "content": '''"""
Testing Patterns
Unit tests, fixtures, mocking, parametrize, and TDD patterns.
"""
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock, patch, MagicMock
import pytest


# === Code Under Test ===
@dataclass
class User:
    id: int
    name: str
    email: str
    is_active: bool = True


class UserRepository:
    def __init__(self, db_connection):
        self.db = db_connection

    def find_by_id(self, user_id: int) -> Optional[User]:
        row = self.db.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        if row:
            return User(**row)
        return None

    def save(self, user: User) -> User:
        self.db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (user.name, user.email),
        )
        return user


class UserService:
    def __init__(self, repo: UserRepository, email_client):
        self.repo = repo
        self.email_client = email_client

    def register(self, name: str, email: str) -> User:
        if not email or "@" not in email:
            raise ValueError("Invalid email address")

        user = User(id=0, name=name, email=email)
        saved = self.repo.save(user)
        self.email_client.send_welcome(email)
        return saved

    def deactivate(self, user_id: int) -> bool:
        user = self.repo.find_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        user.is_active = False
        self.repo.save(user)
        return True


# === Test Fixtures ===
@pytest.fixture
def mock_db():
    return Mock()

@pytest.fixture
def mock_email():
    return Mock()

@pytest.fixture
def repo(mock_db):
    return UserRepository(mock_db)

@pytest.fixture
def service(repo, mock_email):
    return UserService(repo, mock_email)

@pytest.fixture
def sample_user():
    return User(id=1, name="Alice", email="alice@test.com")


# === Unit Tests ===
class TestUserService:
    def test_register_success(self, service, mock_email):
        """Happy path: valid registration."""
        user = service.register("Bob", "bob@test.com")
        assert user.name == "Bob"
        assert user.email == "bob@test.com"
        mock_email.send_welcome.assert_called_once_with("bob@test.com")

    def test_register_invalid_email(self, service):
        """Should raise ValueError for invalid email."""
        with pytest.raises(ValueError, match="Invalid email"):
            service.register("Bob", "not-an-email")

    def test_register_empty_email(self, service):
        with pytest.raises(ValueError):
            service.register("Bob", "")

    def test_deactivate_user(self, service, mock_db):
        """Should deactivate an existing user."""
        mock_db.execute.return_value = {
            "id": 1, "name": "Alice", "email": "alice@test.com", "is_active": True
        }
        result = service.deactivate(1)
        assert result is True

    def test_deactivate_nonexistent_user(self, service, mock_db):
        """Should raise ValueError for unknown user."""
        mock_db.execute.return_value = None
        with pytest.raises(ValueError, match="not found"):
            service.deactivate(999)


# === Parametrized Tests ===
@pytest.mark.parametrize("email,valid", [
    ("user@example.com", True),
    ("a@b.co", True),
    ("no-at-sign", False),
    ("", False),
    ("spaces @test.com", True),  # Has @ so passes basic check
])
def test_email_validation(service, email, valid):
    if valid:
        user = service.register("Test", email)
        assert user.email == email
    else:
        with pytest.raises(ValueError):
            service.register("Test", email)


# === Test for Side Effects ===
class TestEmailIntegration:
    def test_welcome_email_sent_on_register(self, service, mock_email):
        service.register("New User", "new@test.com")
        mock_email.send_welcome.assert_called_once()

    def test_no_email_sent_on_failure(self, service, mock_email):
        with pytest.raises(ValueError):
            service.register("Bad", "")
        mock_email.send_welcome.assert_not_called()
'''
    },
    {
        "filename": "system_design_rate_limiter.md",
        "content": '''# System Design: Rate Limiter

## Requirements
- Limit requests per user/IP within a time window
- Low latency (< 1ms overhead)
- Distributed (works across multiple servers)
- Configurable rules (e.g., 100 req/min for API, 5 req/min for login)

## Algorithms

### 1. Token Bucket
Tokens refill at a fixed rate. Each request consumes a token.

```
Capacity: 10 tokens
Refill rate: 2 tokens/second

Request arrives → tokens > 0? → Allow (tokens -= 1)
                              → Deny (429 Too Many Requests)
```

**Pros**: Allows bursts up to bucket capacity
**Cons**: Memory per user (2 values: tokens + timestamp)

### 2. Sliding Window Log
Store timestamp of each request. Count requests in window.

**Pros**: Exact count, no boundary issues
**Cons**: High memory (stores every timestamp)

### 3. Sliding Window Counter
Hybrid: fixed window counts + weighted overlap.

```
Current window:  [====70%====]
Previous window: [====30%====]

Count = current_count * 0.7 + previous_count * 0.3
```

**Pros**: Low memory, smooth rate limiting
**Cons**: Approximate (but close enough)

## Distributed Implementation (Redis)

```python
def is_allowed(user_id, limit, window_seconds):
    key = f"rate:{user_id}"
    pipe = redis.pipeline()

    now = time.time()
    window_start = now - window_seconds

    pipe.zremrangebyscore(key, 0, window_start)  # Remove old entries
    pipe.zadd(key, {str(now): now})               # Add current request
    pipe.zcard(key)                                # Count in window
    pipe.expire(key, window_seconds)               # Auto-cleanup

    _, _, count, _ = pipe.execute()
    return count <= limit
```

## Architecture

```
Client → API Gateway → Rate Limiter → Backend
                            │
                        Redis Cluster
                       (shared state)
```

## Rules Configuration

```yaml
rate_limits:
  - name: api_default
    key: user_id
    limit: 100
    window: 60s

  - name: auth_login
    key: ip_address
    limit: 5
    window: 300s
    path: /api/auth/login

  - name: search
    key: user_id
    limit: 30
    window: 60s
    path: /api/search
```

## Response Headers

```
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705313400
Retry-After: 42
```
'''
    },
    {
        "filename": "clean_code_refactoring.py",
        "content": '''"""
Clean Code: Refactoring Examples
Before and after applying clean code principles.
"""

# ============================================================
# 1. EXTRACT METHOD — Break long functions into named steps
# ============================================================

# Before: One giant function
def process_order_before(order):
    # Validate
    if not order.get("items"):
        raise ValueError("No items")
    if order.get("total", 0) <= 0:
        raise ValueError("Invalid total")
    for item in order["items"]:
        if item["quantity"] <= 0:
            raise ValueError("Invalid quantity")

    # Calculate
    subtotal = sum(i["price"] * i["quantity"] for i in order["items"])
    tax = subtotal * 0.08
    shipping = 0 if subtotal > 50 else 5.99
    total = subtotal + tax + shipping

    # Format
    return {
        "order_id": order["id"],
        "subtotal": round(subtotal, 2),
        "tax": round(tax, 2),
        "shipping": round(shipping, 2),
        "total": round(total, 2),
    }


# After: Clear, named steps
def process_order_after(order):
    validate_order(order)
    pricing = calculate_pricing(order["items"])
    return format_order_summary(order["id"], pricing)


def validate_order(order):
    if not order.get("items"):
        raise ValueError("Order must contain at least one item")
    if order.get("total", 0) <= 0:
        raise ValueError("Order total must be positive")
    for item in order["items"]:
        if item["quantity"] <= 0:
            raise ValueError(f"Invalid quantity for item: {item.get(\'name\', \'unknown\')}")


def calculate_pricing(items):
    subtotal = sum(item["price"] * item["quantity"] for item in items)
    tax = subtotal * 0.08
    shipping = 0 if subtotal > 50 else 5.99
    return {"subtotal": subtotal, "tax": tax, "shipping": shipping}


def format_order_summary(order_id, pricing):
    total = sum(pricing.values())
    return {
        "order_id": order_id,
        **{k: round(v, 2) for k, v in pricing.items()},
        "total": round(total, 2),
    }


# ============================================================
# 2. REPLACE CONDITIONALS WITH POLYMORPHISM
# ============================================================

# Before: Switch on type
def calculate_area_before(shape):
    if shape["type"] == "circle":
        return 3.14159 * shape["radius"] ** 2
    elif shape["type"] == "rectangle":
        return shape["width"] * shape["height"]
    elif shape["type"] == "triangle":
        return 0.5 * shape["base"] * shape["height"]
    else:
        raise ValueError(f"Unknown shape: {shape[\'type\']}")


# After: Polymorphism
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14159 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height
    def area(self):
        return 0.5 * self.base * self.height


# ============================================================
# 3. GUARD CLAUSES — Flatten nested conditionals
# ============================================================

# Before: Deep nesting
def get_discount_before(user, order):
    if user is not None:
        if user.is_active:
            if order.total > 100:
                if user.is_premium:
                    return 0.20
                else:
                    return 0.10
            else:
                return 0.05
        else:
            return 0
    else:
        return 0

# After: Guard clauses
def get_discount_after(user, order):
    if user is None or not user.is_active:
        return 0

    if order.total <= 100:
        return 0.05

    return 0.20 if user.is_premium else 0.10
'''
    },
    {
        "filename": "hash_table_implementation.py",
        "content": '''"""
Hash Table Implementation
Open addressing with linear probing and dynamic resizing.
"""

class HashTable:
    DELETED = object()  # Sentinel for deleted slots

    def __init__(self, initial_capacity=8, load_factor_threshold=0.7):
        self.capacity = initial_capacity
        self.load_factor_threshold = load_factor_threshold
        self.size = 0
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, key):
        """Linear probing to find slot for key."""
        index = self._hash(key)
        first_deleted = None

        for _ in range(self.capacity):
            if self.keys[index] is None:
                return first_deleted if first_deleted is not None else index
            if self.keys[index] is self.DELETED:
                if first_deleted is None:
                    first_deleted = index
            elif self.keys[index] == key:
                return index
            index = (index + 1) % self.capacity

        return first_deleted  # Table is full of deleted/occupied

    def _resize(self):
        """Double capacity and rehash all entries."""
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [None] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0

        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)

    def put(self, key, value):
        if self.size / self.capacity >= self.load_factor_threshold:
            self._resize()

        index = self._probe(key)
        is_new = self.keys[index] is None or self.keys[index] is self.DELETED
        self.keys[index] = key
        self.values[index] = value
        if is_new:
            self.size += 1

    def get(self, key, default=None):
        index = self._hash(key)
        for _ in range(self.capacity):
            if self.keys[index] is None:
                return default
            if self.keys[index] == key:
                return self.values[index]
            index = (index + 1) % self.capacity
        return default

    def delete(self, key):
        index = self._hash(key)
        for _ in range(self.capacity):
            if self.keys[index] is None:
                raise KeyError(key)
            if self.keys[index] == key:
                self.keys[index] = self.DELETED
                self.values[index] = None
                self.size -= 1
                return
            index = (index + 1) % self.capacity
        raise KeyError(key)

    def __contains__(self, key):
        return self.get(key, self.DELETED) is not self.DELETED

    def __len__(self):
        return self.size

    def __repr__(self):
        items = []
        for k, v in zip(self.keys, self.values):
            if k is not None and k is not self.DELETED:
                items.append(f"{k!r}: {v!r}")
        return "{" + ", ".join(items) + "}"


if __name__ == "__main__":
    ht = HashTable()

    # Insert
    for i in range(20):
        ht.put(f"key_{i}", i * 10)

    print(f"Size: {len(ht)}, Capacity: {ht.capacity}")
    print(f"Get key_5: {ht.get('key_5')}")
    print(f"Contains key_15: {'key_15' in ht}")

    # Delete
    ht.delete("key_5")
    print(f"After delete, get key_5: {ht.get('key_5', 'NOT FOUND')}")
    print(f"Size after delete: {len(ht)}")
'''
    },
    {
        "filename": "observer_pattern.py",
        "content": '''"""
Observer Pattern (Event System)
Decouple event producers from consumers.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Any
from datetime import datetime


@dataclass
class Event:
    name: str
    data: dict
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = dt_class.utcnow().isoformat()


class EventBus:
    """Publish-subscribe event bus with support for:
    - Multiple handlers per event
    - Wildcard subscriptions
    - Handler priorities
    - Error isolation
    """

    def __init__(self):
        self._handlers: dict[str, list[tuple[int, Callable]]] = defaultdict(list)
        self._history: list[Event] = []

    def subscribe(self, event_name: str, handler: Callable, priority: int = 0):
        """Subscribe to an event. Lower priority number = called first."""
        self._handlers[event_name].append((priority, handler))
        self._handlers[event_name].sort(key=lambda x: x[0])
        return lambda: self.unsubscribe(event_name, handler)

    def unsubscribe(self, event_name: str, handler: Callable):
        self._handlers[event_name] = [
            (p, h) for p, h in self._handlers[event_name] if h != handler
        ]

    def publish(self, event: Event):
        """Publish event to all matching handlers."""
        self._history.append(event)

        # Exact match handlers
        for _, handler in self._handlers.get(event.name, []):
            try:
                handler(event)
            except Exception as e:
                print(f"Handler error for {event.name}: {e}")

        # Wildcard handlers
        for _, handler in self._handlers.get("*", []):
            try:
                handler(event)
            except Exception as e:
                print(f"Wildcard handler error: {e}")

    def get_history(self, event_name: str = None, limit: int = 10):
        events = self._history
        if event_name:
            events = [e for e in events if e.name == event_name]
        return events[-limit:]


# === Usage Example ===

class OrderService:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

    def create_order(self, user_id, items, total):
        order_id = f"ord_{hash((user_id, total)) % 10000:04d}"
        # Business logic here...

        self.event_bus.publish(Event(
            name="order.created",
            data={"order_id": order_id, "user_id": user_id, "total": total},
        ))
        return order_id


class EmailNotifier:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("order.created", self.on_order_created)

    def on_order_created(self, event: Event):
        print(f"Sending confirmation email for order {event.data[\'order_id\']}")


class AnalyticsTracker:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("*", self.track_all, priority=99)
        self.events_tracked = 0

    def track_all(self, event: Event):
        self.events_tracked += 1
        print(f"Analytics: tracked {event.name}")


class InventoryService:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("order.created", self.reserve_inventory, priority=-1)

    def reserve_inventory(self, event: Event):
        print(f"Reserving inventory for order {event.data[\'order_id\']}")


if __name__ == "__main__":
    bus = EventBus()

    # Wire up services
    orders = OrderService(bus)
    email = EmailNotifier(bus)
    analytics = AnalyticsTracker(bus)
    inventory = InventoryService(bus)

    # Create an order — all observers react
    order_id = orders.create_order("usr_42", ["widget_a"], 29.99)
    print(f"Order created: {order_id}")
    print(f"Events tracked: {analytics.events_tracked}")
'''
    },
]

# ============================================================
# MAIN: Content Selection & Generation
# ============================================================
ALL_SNIPPETS = {
    "ai-ml": AI_ML_SNIPPETS,
    "devops": DEVOPS_SNIPPETS,
    "data-engineering": DATA_ENGINEERING_SNIPPETS,
    "software-engineering": SOFTWARE_ENGINEERING_SNIPPETS,
}


def get_today_content():
    """Select today's content deterministically based on date."""
    today = dt_class.now()
    day_of_year = today.timetuple().tm_yday
    year = today.year

    # Create a combined pool with category info
    pool = []
    for category, snippets in ALL_SNIPPETS.items():
        for snippet in snippets:
            pool.append((category, snippet))

    # Use day + year for deterministic but yearly-varying selection
    seed = day_of_year + year * 1000
    index = seed % len(pool)

    category, snippet = pool[index]
    return category, snippet


# Daily tips appended to each file to make every commit unique,
# even when the base snippet cycles after 55 days
DAILY_TIPS = [
    "Tip: Always profile before optimizing — measure, don't guess.",
    "Tip: Write tests for the behavior you want, not the implementation you have.",
    "Tip: Prefer composition over inheritance in most cases.",
    "Tip: Code is read 10x more than it's written — optimize for readability.",
    "Tip: Make your functions do one thing well.",
    "Tip: Naming is the hardest problem — spend time on it.",
    "Tip: The best code is code you don't have to write.",
    "Tip: Fail fast, fail loudly — silent errors are debugging nightmares.",
    "Tip: Immutable data structures prevent entire categories of bugs.",
    "Tip: A good abstraction is worth more than clever code.",
    "Tip: If you can't explain it simply, you don't understand it well enough.",
    "Tip: Premature optimization is the root of all evil — Knuth.",
    "Tip: Log at the boundaries: inputs, outputs, errors.",
    "Tip: Every magic number deserves a named constant.",
    "Tip: Automate anything you do more than twice.",
    "Tip: Ship small, ship often — smaller PRs get better reviews.",
    "Tip: The fastest code is code that never runs — eliminate dead paths.",
    "Tip: Good error messages save hours of debugging.",
    "Tip: Idempotent operations make distributed systems safer.",
    "Tip: Cache invalidation is hard — start without caching, add when proven needed.",
    "Tip: Use feature flags to decouple deployment from release.",
    "Tip: Monitor the four golden signals: latency, traffic, errors, saturation.",
    "Tip: Database indexes are like a book's table of contents — choose them carefully.",
    "Tip: Always have a rollback plan before deploying to production.",
    "Tip: The best documentation lives close to the code it describes.",
    "Tip: Use structured logging (JSON) — your future self will thank you.",
    "Tip: Retry with exponential backoff and jitter, never fixed intervals.",
    "Tip: A circuit breaker prevents cascading failures across services.",
    "Tip: Treat infrastructure as code — version it, review it, test it.",
    "Tip: The CAP theorem isn't a choice — understand the tradeoffs in your system.",
    "Tip: Normalize until it hurts, denormalize until it works.",
]


def write_content():
    """Write today's content to the appropriate directory."""
    category, snippet = get_today_content()
    today = dt_class.now()
    today_str = today.strftime("%Y-%m-%d")

    # Create directory
    dir_path = os.path.join(CONTENT_DIR, category)
    os.makedirs(dir_path, exist_ok=True)

    # Write file with date prefix to avoid collisions
    filename = f"{today_str}_{snippet['filename']}"
    filepath = os.path.join(dir_path, filename)

    with open(filepath, "w") as f:
        f.write(snippet["content"])

    print(f"[{today_str}] Category: {category}")
    print(f"[{today_str}] File: {filepath}")
    print(f"[{today_str}] Size: {len(snippet['content'])} bytes")
    return filepath, category


if __name__ == "__main__":
    filepath, category = write_content()
    print(f"Content written to: {filepath}")
    print(f"CATEGORY={category}")
