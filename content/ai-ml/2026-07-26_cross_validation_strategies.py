"""
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

    print("\n=== Stratified K-Fold ===")
    for i, (train, test) in enumerate(stratified_k_fold(y)):
        print(f"Fold {i+1}: train={len(train)}, test={len(test)}, "
              f"test_class_1_ratio={np.mean(y[test]):.2f}")

    print("\n=== Time Series Split ===")
    for i, (train, test) in enumerate(time_series_split(n)):
        print(f"Split {i+1}: train=[0:{len(train)}], test=[{len(train)}:{len(train)+len(test)}]")
