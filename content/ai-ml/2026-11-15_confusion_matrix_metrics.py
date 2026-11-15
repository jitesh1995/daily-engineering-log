"""
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
    print(f"Confusion Matrix:\n{cm}")

    metrics = precision_recall_f1(y_true, y_pred)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {roc_auc(y_true, y_scores):.4f}")
