"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.calibration import calibration_curve, calibration_error, sharpness
from src.evaluation.compare_models import evaluate_predictions


def test_calibration_curve():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.4, 0.8, 0.15, 0.9, 0.85])
    cal = calibration_curve(y_true, y_prob, n_bins=5)
    assert len(cal) > 0
    assert "mean_predicted" in cal.columns
    assert "mean_observed" in cal.columns
    assert "count" in cal.columns


def test_calibration_error():
    # Perfect calibration
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    ece = calibration_error(y_true, y_prob, n_bins=5)
    assert ece < 0.2  # should be close to 0


def test_sharpness():
    confident = np.array([0.05, 0.95, 0.02, 0.98, 0.1, 0.9])
    uncertain = np.array([0.45, 0.55, 0.48, 0.52, 0.5, 0.5])

    sharp_confident = sharpness(confident)
    sharp_uncertain = sharpness(uncertain)

    assert sharp_confident["mean_sharpness"] > sharp_uncertain["mean_sharpness"]
    assert sharp_confident["pct_confident"] > sharp_uncertain["pct_confident"]


def test_evaluate_predictions():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.7, 0.3, 0.9])
    result = evaluate_predictions(y_true, y_prob, "test_model", "test")

    assert result["model"] == "test_model"
    assert result["n"] == 5
    assert 0 <= result["brier_score"] <= 1
    assert result["log_loss"] > 0


def test_evaluate_empty():
    result = evaluate_predictions(np.array([]), np.array([]), "empty", "test")
    assert result["n"] == 0
