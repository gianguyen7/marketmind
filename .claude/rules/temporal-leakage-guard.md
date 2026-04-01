---
name: temporal-leakage-guard
description: Prevent future information from leaking into training data or features
---

# Temporal Leakage Guard

Temporal leakage is the single most common and damaging error in forecasting research. Enforce these checks at every stage:

## Feature Engineering
- Every feature must use only data available BEFORE the prediction timestamp
- Rolling/window features: backward-looking only (`df.rolling(window).shift(1)`)
- Never use: final resolution, closing price, post-event data as features
- Cross-market features (e.g., category base rate) must use only historically resolved markets

## Train/Test Split
- Split on time, never randomly: train < `train_cutoff` < test
- No market should appear in both train and test
- Validate: `assert train['timestamp'].max() < test['timestamp'].min()`

## Code Patterns to Flag
```python
# BAD — leaks future
df['feature'] = df.groupby('category')['resolved'].transform('mean')

# GOOD — backward-looking only
df['feature'] = df.groupby('category')['resolved'].transform(
    lambda x: x.expanding().mean().shift(1)
)

# BAD — random split
train, test = train_test_split(df, test_size=0.2)

# GOOD — temporal split
train = df[df['timestamp'] < train_cutoff]
test = df[df['timestamp'] >= test_cutoff]
```

## When Reviewing Code
If you see `train_test_split`, `shuffle=True`, or `KFold` (without `TimeSeries`), flag it immediately.
