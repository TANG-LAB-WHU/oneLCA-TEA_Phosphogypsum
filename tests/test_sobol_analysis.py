"""Tests for Sobol analysis API."""

import numpy as np
import pytest

from pgloop.uncertainty.sensitivity import SensitivityAnalyzer


def test_sobol_analysis_runs_or_skips_without_salib():
    analyzer = SensitivityAnalyzer()
    problem = {
        "num_vars": 2,
        "names": ["x1", "x2"],
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }

    def model_eval(X: np.ndarray) -> np.ndarray:
        return X[:, 0] + 2.0 * X[:, 1]

    try:
        result = analyzer.sobol_analysis(problem=problem, model_eval_fn=model_eval, n_samples=64)
    except ImportError:
        pytest.skip("SALib not installed in test environment.")

    assert "S1" in result
    assert "x1" in result["S1"]
    assert "ST" in result
