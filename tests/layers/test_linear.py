"""
Linear Layer Tests

Comprehensive test suite for Linear (fully-connected) layer with focus on:
- Proper weight initialization (Xavier/Glorot)
- Forward pass correctness
- Bias handling
- Parameter management
- Shape transformations
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.layers import Linear
from tests.layers.conftest import (
    TOLERANCE,
    assert_shape_correct,
    assert_finite,
    assert_weights_initialized,
    assert_bias_zeros,
    assert_close,
    assert_parameters_count,
)


class TestLinearInitialization:
    """Test Linear layer initialization."""

    def test_linear_init_with_bias(self):
        """Test Linear layer initialization with bias."""
        in_features = 10
        out_features = 5
        linear = Linear(in_features, out_features, bias=True)

        assert linear.in_features == in_features
        assert linear.out_features == out_features
        assert_shape_correct(linear.weight, (in_features, out_features))
        assert_shape_correct(linear.bias, (out_features,))
        assert linear.bias is not None

    def test_linear_init_without_bias(self):
        """Test Linear layer initialization without bias."""
        in_features = 10
        out_features = 5
        linear = Linear(in_features, out_features, bias=False)

        assert linear.in_features == in_features
        assert linear.out_features == out_features
        assert_shape_correct(linear.weight, (in_features, out_features))
        assert linear.bias is None

    def test_linear_weight_initialization(self):
        """Test weights are initialized with Xavier scaling."""
        in_features = 100
        out_features = 50
        linear = Linear(in_features, out_features)

        # Check weights are finite
        assert_finite(linear.weight)

        # Check Xavier initialization bounds
        assert_weights_initialized(linear.weight, in_features)

    def test_linear_bias_initialization(self):
        """Test bias is initialized to zeros."""
        in_features = 10
        out_features = 5
        linear = Linear(in_features, out_features, bias=True)

        # Bias should be zeros
        assert_bias_zeros(linear.bias)

    def test_linear_weight_dtype(self):
        """Test weights have correct dtype."""
        linear = Linear(10, 5)
        assert linear.weight.dtype == np.float32
        assert linear.bias.dtype == np.float32


class TestLinearForward:
    """Test Linear layer forward pass."""

    def test_linear_forward_basic(self):
        """Test basic forward pass computation."""
        # Create simple linear layer with known weights
        # Linear layer has weight shape (in_features, out_features)
        linear = Linear(2, 2, bias=False)
        # Set weight to [[1, 3], [2, 4]] (in_features=2, out_features=2)
        linear.weight = Tensor([[1.0, 3.0], [2.0, 4.0]])

        # Input: [[1, 2]] (batch_size=1, in_features=2)
        x = Tensor([[1.0, 2.0]])
        result = linear(x)

        # Expected: [[1, 2]] @ [[1, 3], [2, 4]] = [[1*1 + 2*2, 1*3 + 2*4]] = [[5, 11]]
        expected = np.array([[5.0, 11.0]], dtype=np.float32)
        assert_close(result.data, expected)

    def test_linear_forward_with_bias(self):
        """Test forward pass with bias."""
        linear = Linear(2, 2, bias=True)
        linear.weight = Tensor([[1.0, 3.0], [2.0, 4.0]])
        linear.bias = Tensor([10.0, 20.0])

        x = Tensor([[1.0, 2.0]])
        result = linear(x)

        # Expected: [[1, 2]] @ [[1, 3], [2, 4]] + [10, 20] = [[5, 11]] + [[10, 20]] = [[15, 31]]
        expected = np.array([[15.0, 31.0]], dtype=np.float32)
        assert_close(result.data, expected)

    def test_linear_forward_batch(self):
        """Test forward pass with batch input."""
        linear = Linear(3, 2, bias=False)
        linear.weight = Tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )  # shape (3, 2)

        # Batch of 4 samples
        x = Tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        )
        result = linear(x)

        assert_shape_correct(result, (4, 2))

        # Each sample should be transformed correctly
        expected = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [9.0, 12.0]], dtype=np.float32
        )
        assert_close(result.data, expected)

    def test_linear_forward_shape_preservation(self):
        """Test that forward pass preserves batch dimension."""
        in_features = 10
        out_features = 5
        batch_size = 32
        linear = Linear(in_features, out_features)

        x = Tensor(np.random.randn(batch_size, in_features))
        result = linear(x)

        assert_shape_correct(result, (batch_size, out_features))

    def test_linear_forward_1d_input(self):
        """Test forward pass with 1D input (single sample)."""
        linear = Linear(5, 3, bias=False)
        linear.weight = Tensor(np.ones((5, 3)))

        x = Tensor(np.ones(5))
        result = linear(x)

        # Sum of 5 ones for each output should be 5
        expected = np.ones(3) * 5
        # Handle shape difference (output might be 1D)
        assert_close(result.data.flatten(), expected)

    def test_linear_forward_output_finite(self):
        """Test forward pass produces finite values."""
        linear = Linear(10, 5)
        x = Tensor(np.random.randn(32, 10))
        result = linear(x)

        assert_finite(result)

    def test_linear_forward_no_input_mutation(self):
        """Test that forward pass doesn't mutate input."""
        linear = Linear(5, 3)
        x = Tensor(np.random.randn(10, 5))
        x_copy = x.data.copy()

        _ = linear(x)

        # Input should not be modified
        assert_close(x.data, x_copy)


class TestLinearParameters:
    """Test Linear layer parameter management."""

    def test_linear_parameters_with_bias(self):
        """Test parameters() returns weight and bias."""
        linear = Linear(10, 5, bias=True)
        params = linear.parameters()

        assert_parameters_count(linear, 2)
        assert params[0] is linear.weight
        assert params[1] is linear.bias

    def test_linear_parameters_without_bias(self):
        """Test parameters() returns only weight when no bias."""
        linear = Linear(10, 5, bias=False)
        params = linear.parameters()

        assert_parameters_count(linear, 1)
        assert params[0] is linear.weight

    def test_linear_parameters_are_tensors(self):
        """Test that all parameters are Tensors."""
        linear = Linear(10, 5, bias=True)
        params = linear.parameters()

        for param in params:
            assert isinstance(param, Tensor)

    def test_linear_weight_parameter_correct_shape(self):
        """Test weight parameter has correct shape."""
        in_features = 15
        out_features = 8
        linear = Linear(in_features, out_features)
        params = linear.parameters()

        weight = params[0]
        assert_shape_correct(weight, (in_features, out_features))


class TestLinearEdgeCases:
    """Test Linear layer edge cases."""

    def test_linear_single_input_single_output(self):
        """Test Linear layer with 1 input and 1 output."""
        linear = Linear(1, 1, bias=False)
        linear.weight = Tensor([[2.0]])

        x = Tensor([[3.0]])
        result = linear(x)

        expected = np.array([[6.0]], dtype=np.float32)
        assert_close(result.data, expected)

    def test_linear_large_dimensions(self):
        """Test Linear layer with large input/output dimensions."""
        in_features = 1000
        out_features = 512
        linear = Linear(in_features, out_features)

        x = Tensor(np.random.randn(4, in_features))
        result = linear(x)

        assert_shape_correct(result, (4, out_features))
        assert_finite(result)

    def test_linear_zero_input(self):
        """Test forward pass with zero input."""
        linear = Linear(5, 3, bias=False)
        x = Tensor(np.zeros((2, 5)))
        result = linear(x)

        # Output should be zero (since 0 @ W = 0)
        expected = np.zeros((2, 3), dtype=np.float32)
        assert_close(result.data, expected)

    def test_linear_zero_input_with_bias(self):
        """Test forward pass with zero input but with bias."""
        linear = Linear(5, 3, bias=True)
        linear.bias = Tensor(np.array([1.0, 2.0, 3.0]))

        x = Tensor(np.zeros((2, 5)))
        result = linear(x)

        # Output should be just the bias
        expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        assert_close(result.data, expected)

    def test_linear_high_values(self):
        """Test forward pass with high input values."""
        linear = Linear(3, 2, bias=False)
        linear.weight = Tensor(np.ones((3, 2)))

        x = Tensor(np.full((2, 3), 100.0))
        result = linear(x)

        # 100 + 100 + 100 = 300 for each output
        expected = np.full((2, 2), 300.0, dtype=np.float32)
        assert_close(result.data, expected)


class TestLinearRepr:
    """Test Linear layer string representation."""

    def test_linear_repr_with_bias(self):
        """Test __repr__ with bias."""
        linear = Linear(10, 5, bias=True)
        repr_str = repr(linear)

        assert "Linear" in repr_str
        assert "10" in repr_str
        assert "5" in repr_str
        assert "bias=True" in repr_str

    def test_linear_repr_without_bias(self):
        """Test __repr__ without bias."""
        linear = Linear(10, 5, bias=False)
        repr_str = repr(linear)

        assert "Linear" in repr_str
        assert "10" in repr_str
        assert "5" in repr_str
        assert "bias=False" in repr_str


class TestLinearNumericalStability:
    """Test numerical stability of Linear layer."""

    def test_linear_symmetric_input(self):
        """Test forward pass with symmetric input values."""
        linear = Linear(4, 2, bias=False)
        linear.weight = Tensor(np.ones((4, 2)))

        x = Tensor([[-1.0, -0.5, 0.5, 1.0]])
        result = linear(x)

        # Sum: -1 - 0.5 + 0.5 + 1 = 0
        expected = np.zeros((1, 2), dtype=np.float32)
        assert_close(result.data, expected, atol=TOLERANCE)

    def test_linear_mixed_scales(self):
        """Test forward pass with mixed magnitude input."""
        linear = Linear(3, 2, bias=False)
        linear.weight = Tensor([[1e-3, 1e3], [1.0, 1.0], [1e3, 1e-3]])

        x = Tensor([[1.0, 1.0, 1.0]])
        result = linear(x)

        assert_finite(result)

    def test_linear_repeated_forward_passes(self):
        """Test that repeated forward passes give consistent results."""
        linear = Linear(5, 3)
        x = Tensor(np.random.randn(2, 5))

        result1 = linear(x)
        result2 = linear(x)

        assert_close(result1.data, result2.data)


class TestLinearCallable:
    """Test Linear layer is callable."""

    def test_linear_call_method(self):
        """Test Linear layer can be called with __call__."""
        linear = Linear(5, 3)
        x = Tensor(np.random.randn(10, 5))

        # Should work with both forward() and __call__()
        result_forward = linear.forward(x)
        result_call = linear(x)

        assert_close(result_forward.data, result_call.data)

    def test_linear_returns_tensor(self):
        """Test forward pass returns a Tensor."""
        linear = Linear(5, 3)
        x = Tensor(np.random.randn(1, 5))
        result = linear(x)

        assert isinstance(result, Tensor)
