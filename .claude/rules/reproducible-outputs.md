---
name: reproducible-outputs
description: Every experiment must be fully reproducible from config + code + data
---

# Reproducible Outputs

## Requirements
1. **Random seed**: always load from `configs/modeling.yaml:random_seed` and pass to all stochastic operations (sklearn, xgboost, numpy, train/test splits)
2. **Config logging**: save a copy of the experiment config alongside results
3. **Dataset versioning**: log the hash of train/test parquet files used
4. **Dependency pinning**: `requirements.txt` must have pinned versions

## Naming Convention
All outputs include the experiment name and date:
```
outputs/models/{experiment}_{model}_{YYYY-MM-DD}.pkl
outputs/tables/{experiment}_results_{YYYY-MM-DD}.csv
outputs/figures/{experiment}_{plot_type}_{YYYY-MM-DD}.png
```

## Reproducibility Checklist (before reporting results)
- [ ] Can I re-run the experiment from config and get the same numbers?
- [ ] Is the random seed set before every stochastic call?
- [ ] Are train/test splits deterministic (time-based, not random)?
- [ ] Are all hyperparameters in config, not hardcoded?
