"""Simple text-based features from market questions."""

import re

import numpy as np
import pandas as pd


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract lightweight features from market question text."""
    df = df.copy()

    if "question" not in df.columns:
        return df

    # Question length
    df["question_length"] = df["question"].fillna("").str.len()
    df["question_word_count"] = df["question"].fillna("").str.split().str.len()

    # Contains temporal keywords
    temporal_pattern = r"\b(?:before|after|by|until|end of|deadline|january|february|march|april|may|june|july|august|september|october|november|december|\d{4})\b"
    df["has_temporal_ref"] = (
        df["question"].fillna("").str.contains(temporal_pattern, case=False, regex=True).astype(int)
    )

    # Contains numeric target
    df["has_numeric_target"] = (
        df["question"].fillna("").str.contains(r"\d+%|\d+\.\d+|\$\d+", regex=True).astype(int)
    )

    # Question type indicators
    df["is_will_question"] = (
        df["question"].fillna("").str.lower().str.startswith("will").astype(int)
    )

    return df
