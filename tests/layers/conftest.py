"""
Shared configuration and utilities for layer tests.
"""

import numpy as np
import pytest

TOLERANCE = 1e-5


def assert_shape_correct(tensor, expected_shape):
    """Assert tensor has correct shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_finite(tensor):
    """Assert all values in tensor are finite (no NaN or Inf)."""
    assert np.all(np.isfinite(tensor.data)), "Tensor contains NaN or Inf values"


def assert_weights_initialized(weight_tensor, in_features):
    """Assert weights are properly initialized with Xavier scaling."""
    expected_scale = np.sqrt(1.0 / in_features)
    # Check that values are roughly in the expected range (allow some variance)
    # Allow up to 4-5 standard deviations for the maximum value
    assert np.max(np.abs(weight_tensor.data)) < 5 * expected_scale, \
        "Weights exceed reasonable bounds for Xavier initialization"


def assert_bias_zeros(bias_tensor):
    """Assert bias is initialized to zeros."""
    assert np.allclose(bias_tensor.data, 0.0, atol=TOLERANCE), \
        "Bias should be initialized to zeros"


def assert_close(actual, expected, atol=TOLERANCE):
    """Assert two arrays are close within tolerance."""
    assert np.allclose(actual, expected, atol=atol), \
        f"Values not close: max diff = {np.max(np.abs(actual - expected))}"


def assert_parameters_count(layer, expected_count):
    """Assert layer has expected number of parameters."""
    params = layer.parameters()
    assert len(params) == expected_count, \
        f"Expected {expected_count} parameters, got {len(params)}"
