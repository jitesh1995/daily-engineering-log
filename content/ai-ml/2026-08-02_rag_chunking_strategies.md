# RAG Chunking Strategies

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
separators = ["\n\n", "\n", ". ", " ", ""]
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
