---
name: build-streamlit-dashboard
description: Scaffold or update the Streamlit interactive comparison dashboard
---

# /build-streamlit-dashboard

Create or update the Streamlit dashboard for interactive model comparison.

## Dashboard Layout

### Sidebar
- Experiment selector (dropdown from `configs/experiments/`)
- Model multi-select (checkboxes)
- Subgroup filter (category, time bucket)

### Main Panel
1. **Results table** — loaded from `outputs/tables/master_comparison.csv`
2. **Calibration curves** — interactive, toggle models on/off
3. **Prediction distribution** — histogram of predicted probs per model
4. **Subgroup heatmap** — metric × model grid, colored by performance
5. **Raw predictions explorer** — sortable table of individual market predictions

## Implementation

- App entry: `src/dashboard/app.py`
- Load only from `outputs/` — never call APIs or retrain in the dashboard
- Use `st.cache_data` for all data loading
- Keep it single-file unless it grows past 300 lines

## Steps
1. Check if `src/dashboard/app.py` exists
2. If new: scaffold full dashboard with all panels above
3. If exists: update to reflect latest outputs and any new models
4. Verify it runs: `streamlit run src/dashboard/app.py`

## Usage
```
/build-streamlit-dashboard          # create or update
/build-streamlit-dashboard --check  # verify it loads without errors
```
