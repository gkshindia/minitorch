"""Shared test configuration and fixtures for autograd tests.

This module provides common fixtures and utilities for all autograd tests.
"""

import pytest
import numpy as np
from core.tensor import Tensor
from core.autograd import enable_autograd


@pytest.fixture(autouse=True)
def setup_autograd():
    """Ensure autograd is enabled for all tests."""
    enable_autograd(quiet=True)
    yield


@pytest.fixture
def simple_tensor():
    """Fixture: Simple 1D tensor with gradients enabled."""
    return Tensor([1.0, 2.0, 3.0], requires_grad=True)


@pytest.fixture
def matrix_tensor():
    """Fixture: 2D matrix tensor with gradients enabled."""
    return Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)


@pytest.fixture
def batch_tensor():
    """Fixture: Batch of 2D tensors with gradients enabled."""
    return Tensor(np.random.randn(4, 3, 2), requires_grad=True)


def assert_gradient_exists(tensor, expected_shape=None):
    """Helper: Assert that tensor has gradients of correct shape."""
    assert tensor.grad is not None, "Gradient should exist"
    if expected_shape is not None:
        assert tensor.grad.shape == expected_shape, f"Gradient shape mismatch: {tensor.grad.shape} != {expected_shape}"


def assert_no_nan_inf(tensor):
    """Helper: Assert that tensor data has no NaN or Inf values."""
    assert not np.any(np.isnan(tensor.data)), "Tensor contains NaN"
    assert not np.any(np.isinf(tensor.data)), "Tensor contains Inf"
    if tensor.grad is not None:
        assert not np.any(np.isnan(tensor.grad)), "Gradient contains NaN"
        assert not np.any(np.isinf(tensor.grad)), "Gradient contains Inf"
