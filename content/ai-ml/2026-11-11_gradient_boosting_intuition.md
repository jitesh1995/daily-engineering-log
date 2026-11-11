# Gradient Boosting: Intuition & Key Concepts

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
