"""
Shared fixtures and utilities for embedding tests.
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
def vocab_size():
    """Fixture: Standard vocabulary size."""
    return 1000


@pytest.fixture
def embed_dim():
    """Fixture: Standard embedding dimension."""
    return 128


@pytest.fixture
def max_seq_len():
    """Fixture: Standard maximum sequence length."""
    return 512


@pytest.fixture
def small_vocab_size():
    """Fixture: Small vocabulary for quick tests."""
    return 10


@pytest.fixture
def small_embed_dim():
    """Fixture: Small embedding dimension for quick tests."""
    return 8


@pytest.fixture
def small_seq_len():
    """Fixture: Small sequence length for quick tests."""
    return 16


@pytest.fixture
def batch_size():
    """Fixture: Standard batch size."""
    return 4


@pytest.fixture
def token_indices():
    """Fixture: Sample token indices."""
    return Tensor(np.array([1, 5, 3, 7]))


@pytest.fixture
def batch_token_indices(batch_size, small_seq_len):
    """Fixture: Batch of token indices."""
    np.random.seed(42)
    return Tensor(np.random.randint(0, 10, size=(batch_size, small_seq_len)))


# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================

def assert_shape(tensor, expected_shape, msg=""):
    """Assert that tensor has the expected shape."""
    assert tensor.shape == expected_shape, (
        f"{msg} Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_in_range(tensor, min_val, max_val, msg=""):
    """Assert that all values in tensor are within the given range."""
    assert np.all(tensor.data >= min_val) and np.all(tensor.data <= max_val), (
        f"{msg} Values out of range [{min_val}, {max_val}]"
    )


def assert_not_all_same(tensor, msg=""):
    """Assert that not all values in tensor are the same."""
    assert not np.allclose(tensor.data, tensor.data.flat[0]), (
        f"{msg} All values are the same"
    )
