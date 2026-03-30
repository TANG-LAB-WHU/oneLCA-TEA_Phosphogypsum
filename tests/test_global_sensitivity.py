"""Tests for global sensitivity APIs."""

import numpy as np
import pytest

from pgloop.uncertainty.sensitivity import SensitivityAnalyzer


def test_delta_analysis_runs_or_skips_without_salib():
    analyzer = SensitivityAnalyzer()
    problem = {
        "num_vars": 2,
        "names": ["x1", "x2"],
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }
    X = np.random.rand(100, 2)
    Y = X[:, 0] + 0.5 * X[:, 1]

    try:
        result = analyzer.delta_analysis(problem, X, Y)
    except ImportError:
        pytest.skip("SALib not installed in test environment.")

    assert "delta" in result
    assert "x1" in result["delta"]
