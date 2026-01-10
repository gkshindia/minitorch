"""
Shared fixtures and utilities for activation function tests.
"""
import numpy as np
import pytest
from core.tensor import Tensor

# Global tolerance for floating point comparisons
TOLERANCE = 1e-6


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture
def random_tensor():
    """Fixture: Random tensor for testing."""
    np.random.seed(42)
    return Tensor(np.random.randn(100) * 10)


@pytest.fixture
def batch_tensor():
    """Fixture: Batch tensor (4 samples × 3 features)."""
    return Tensor([
        [1.0, -2.0, 3.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -3.0],
        [10.0, -10.0, 0.5]
    ])


@pytest.fixture
def extreme_values():
    """Fixture: Extreme values for stability testing."""
    return Tensor([-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])


@pytest.fixture
def edge_cases():
    """Fixture: Edge cases around zero."""
    return Tensor([-0.5, -0.1, 0.0, 0.1, 0.5])


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def assert_shape_preserved(input_tensor, output_tensor, msg=""):
    """Assert that output tensor has same shape as input tensor."""
    assert output_tensor.shape == input_tensor.shape, \
        f"Shape mismatch: expected {input_tensor.shape}, got {output_tensor.shape}. {msg}"


def assert_bounded(tensor, low, high, msg=""):
    """Assert that all values in tensor are bounded [low, high]."""
    assert np.all(tensor.data >= low), \
        f"Values below lower bound {low}. {msg}"
    assert np.all(tensor.data <= high), \
        f"Values above upper bound {high}. {msg}"


def assert_finite(tensor, msg=""):
    """Assert that all values in tensor are finite (no inf or nan)."""
    assert np.all(np.isfinite(tensor.data)), \
        f"Tensor contains non-finite values (inf or nan). {msg}"


def assert_monotonic_increasing(values, atol=TOLERANCE, msg=""):
    """Assert that values are monotonically increasing."""
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1] + atol, \
            f"Not monotonically increasing at index {i}: {values[i]} > {values[i+1]}. {msg}"


def assert_symmetry(pos_val, neg_val, atol=TOLERANCE, msg=""):
    """Assert symmetry: f(-x) = -f(x)."""
    assert np.allclose(pos_val, -neg_val, atol=atol), \
        f"Symmetry failed: {pos_val} ≠ -{neg_val}. {msg}"
