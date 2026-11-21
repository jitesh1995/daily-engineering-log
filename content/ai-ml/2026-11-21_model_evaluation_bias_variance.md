# Bias-Variance Tradeoff

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
  |   \                    ← Validation error
  |    \___________
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
