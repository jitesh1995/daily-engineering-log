"""
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
        print(f"Score: {r['score']:.4f} | {r['metadata']}")
