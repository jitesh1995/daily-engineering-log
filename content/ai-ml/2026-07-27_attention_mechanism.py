"""
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
