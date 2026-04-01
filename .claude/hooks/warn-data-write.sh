#!/usr/bin/env bash
# Warn if writing directly to data/ outside the pipeline
# Triggered on file writes — advisory only

FILE="$1"
if [[ "$FILE" == data/raw/* ]] || [[ "$FILE" == data/interim/* ]] || [[ "$FILE" == data/processed/* ]]; then
  echo "⚠ Direct write to $FILE detected."
  echo "  Use the data pipeline (scripts/run_data_pipeline.py) for reproducible data flow."
fi

exit 0
