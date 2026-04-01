---
name: summarize-results
description: Generate a concise results summary table across all experiments
---

# /summarize-results

Aggregate results from all completed experiments into a single summary.

## Steps

1. **Scan `outputs/tables/`** for all `*_results.csv` files
2. **Merge into master comparison table**
   - Rows: models (base_rate, recency, logistic, RF, XGBoost, hybrid variants)
   - Columns: Brier score, log loss, ECE, sharpness
   - Include experiment name for context

3. **Rank models** by Brier score (primary) and log loss (secondary)

4. **Generate summary**
   - Markdown table with bold for best-in-column
   - 3-5 bullet point key findings:
     - Does hybrid beat standalone ML?
     - Does hybrid beat Polymarket alone?
     - Which model is best calibrated?
     - Where do simple baselines win?

5. **Output**
   - `outputs/reports/results_summary.md` — human-readable summary
   - `outputs/tables/master_comparison.csv` — full comparison data
   - Print the markdown summary to console

## Usage
```
/summarize-results
```
