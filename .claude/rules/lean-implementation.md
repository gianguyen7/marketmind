---
name: lean-implementation
description: Keep implementation minimal — no premature abstractions, no unnecessary infrastructure
---

# Lean Implementation

## Do
- Write functions, not frameworks
- Use pandas + sklearn + xgboost directly — no custom ML wrappers unless needed
- One script per pipeline stage (`scripts/run_*.py`)
- YAML configs for anything you might change between experiments
- Parquet for data, pickle for models, PNG/PDF for figures

## Don't
- Build plugin systems, registries, or factory patterns for 6 models
- Add Docker, Airflow, MLflow, or other infra unless explicitly requested
- Write abstract base classes for models — sklearn already has that interface
- Create multiple config formats — YAML only
- Add logging frameworks — `print()` and basic `logging` are fine for research

## Rule of Thumb
If you're writing code that doesn't directly contribute to answering one of the four research questions, stop and reconsider.
