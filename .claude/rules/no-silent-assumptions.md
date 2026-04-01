---
name: no-silent-assumptions
description: Log or print every assumption, default, threshold, or imputation decision
---

# No Silent Assumptions

Research code that silently makes decisions produces results you can't trust.

## What Counts as a Silent Assumption
- Filling NaN with 0 or mean without logging it
- Dropping rows without reporting how many and why
- Choosing a default threshold (e.g., probability > 0.5) without stating it
- Using a hyperparameter value not in the config
- Filtering data without logging the filter criteria and count removed

## Required Behavior
```python
# BAD
df = df.dropna()

# GOOD
n_before = len(df)
df = df.dropna(subset=['price', 'volume'])
n_after = len(df)
print(f"Dropped {n_before - n_after} rows with missing price/volume ({n_before} -> {n_after})")

# BAD
if prob > 0.5: label = 1

# GOOD
DECISION_THRESHOLD = 0.5  # arbitrary, used only for confusion matrix display
print(f"Using decision threshold: {DECISION_THRESHOLD}")
```

## When In Doubt
Print it. A noisy but transparent pipeline is worth more than a clean but opaque one.
