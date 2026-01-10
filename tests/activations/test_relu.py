"""
ReLU Activation Function Tests

Comprehensive test suite for ReLU activation with focus on:
- Piecewise linear behavior (identity for positive, zero for negative)
- Sparsity properties in neural networks
- Linearity preservation for positive values
- Dead neuron scenarios
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import ReLU
from tests.activations.conftest import TOLERANCE, assert_shape_preserved, assert_bounded, assert_finite, assert_monotonic_increasing


class TestReLUBasics:
    """Basic functionality tests for ReLU."""
    
    def test_relu_positive_identity(self):
        """ReLU passes through positive values unchanged."""
        relu = ReLU()
        x = Tensor([1.0, 2.0, 3.0])
        result = relu(x)
        assert np.array_equal(result.data, x.data)
    
    def test_relu_negative_zero(self):
        """ReLU zeros out negative values."""
        relu = ReLU()
        x = Tensor([-1.0, -2.0, -3.0])
        result = relu(x)
        assert np.array_equal(result.data, np.zeros(3, dtype=np.float32))
    
    def test_relu_zero_boundary(self):
        """ReLU at zero should be zero."""
        relu = ReLU()
        x = Tensor([0.0])
        result = relu(x)
        assert result.data[0] == 0.0
    
    def test_relu_mixed_values(self):
        """ReLU handles mixed positive/negative values correctly."""
        relu = ReLU()
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu(x)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        assert np.array_equal(result.data, expected)


class TestReLUMathematical:
    """Mathematical properties of ReLU."""
    
    def test_relu_identity_positive(self):
        """f(x) = x for x > 0."""
        relu = ReLU()
        positive_vals = [0.5, 1.0, 10.0, 100.0, 1000.0]
        x = Tensor(positive_vals)
        result = relu(x)
        assert np.array_equal(result.data, x.data)
    
    def test_relu_zero_negative(self):
        """f(x) = 0 for x < 0."""
        relu = ReLU()
        negative_vals = [-0.5, -1.0, -10.0, -100.0, -1000.0]
        x = Tensor(negative_vals)
        result = relu(x)
        assert np.all(result.data == 0.0)


class TestReLUProperties:
    """Shape and range properties."""
    
    def test_relu_shape_1d(self):
        """ReLU preserves 1D shape."""
        relu = ReLU()
        x = Tensor([1.0, -2.0, 3.0])
        result = relu(x)
        assert_shape_preserved(x, result)
    
    def test_relu_shape_2d(self):
        """ReLU preserves 2D shape."""
        relu = ReLU()
        x = Tensor([[1.0, -2.0], [-3.0, 4.0]])
        result = relu(x)
        assert_shape_preserved(x, result)
    
    def test_relu_shape_3d(self):
        """ReLU preserves 3D shape."""
        relu = ReLU()
        x = Tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
        result = relu(x)
        assert_shape_preserved(x, result)
    
    def test_relu_output_non_negative(self):
        """ReLU output always >= 0."""
        relu = ReLU()
        np.random.seed(42)
        x = Tensor(np.random.randn(100) * 10)
        result = relu(x)
        assert np.all(result.data >= 0.0)


class TestReLULinearity:
    """Linearity properties for positive values."""
    
    def test_relu_scaling_positive(self):
        """ReLU(ax) = a*ReLU(x) for positive x."""
        relu = ReLU()
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result_x = relu(x)
        
        a = 2.5
        x_scaled = Tensor(x.data * a)
        result_scaled = relu(x_scaled)
        
        expected = result_x.data * a
        assert np.allclose(result_scaled.data, expected, atol=TOLERANCE)
    
    def test_relu_additivity_positive(self):
        """ReLU(x1 + x2) = ReLU(x1) + ReLU(x2) for positive values."""
        relu = ReLU()
        x1 = Tensor([1.0, 2.0, 3.0])
        x2 = Tensor([4.0, 5.0, 6.0])
        
        result_sum = relu(Tensor(x1.data + x2.data))
        result_individual_sum = relu(x1).data + relu(x2).data
        
        assert np.allclose(result_sum.data, result_individual_sum, atol=TOLERANCE)


class TestReLUSparsity:
    """Sparsity properties (zero outputs for negative inputs)."""
    
    def test_relu_sparsity_count(self):
        """Count of zero outputs matches negative input count."""
        relu = ReLU()
        x = Tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        result = relu(x)
        
        num_zeros = np.sum(result.data == 0.0)
        assert num_zeros == 4  # Three negatives + zero
    
    def test_relu_dead_neurons(self):
        """All zeros for all-negative input (dead neurons)."""
        relu = ReLU()
        mostly_negative = Tensor([-10.0, -5.0, -1.0, -0.1, -0.01])
        result = relu(mostly_negative)
        assert np.all(result.data == 0.0)
    
    def test_relu_sparsity_percentage(self):
        """Verify sparsity percentage with biased negative distribution."""
        relu = ReLU()
        np.random.seed(123)
        random_data = np.random.randn(1000) - 1.0  # Biased toward negative
        x = Tensor(random_data)
        result = relu(x)
        sparsity = np.sum(result.data == 0.0) / len(result.data)
        assert sparsity > 0.5


class TestReLUAPI:
    """Test API consistency."""
    
    def test_relu_call_vs_forward(self):
        """__call__ and forward should give identical results."""
        relu = ReLU()
        x = Tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        result_forward = relu.forward(x)
        result_call = relu(x)
        assert np.array_equal(result_forward.data, result_call.data)
    
    def test_relu_repr(self):
        """String representation should contain 'ReLU'."""
        relu = ReLU()
        repr_str = repr(relu)
        assert "ReLU" in repr_str


class TestReLUUseCases:
    """Real-world usage scenarios."""
    
    def test_relu_hidden_layer(self):
        """ReLU in hidden layer with mixed activation pattern."""
        relu = ReLU()
        
        # Simulate pre-activation values
        pre_activation = Tensor([
            [2.5, -1.0, 0.5, -3.0],   # 2 active, 2 inactive
            [-0.5, 3.0, -2.0, 1.5],   # 2 active, 2 inactive
            [1.0, 1.0, 1.0, 1.0],     # all active
            [-1.0, -1.0, -1.0, -1.0]  # all inactive (dead)
        ])
        
        activations = relu(pre_activation)
        
        expected = np.array([
            [2.5, 0.0, 0.5, 0.0],
            [0.0, 3.0, 0.0, 1.5],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        assert np.array_equal(activations.data, expected)
        
        # Verify some active neurons
        num_active = np.sum(activations.data > 0.0)
        assert num_active > 0
    
    def test_relu_batch_processing(self, batch_tensor):
        """ReLU processes batches correctly."""
        relu = ReLU()
        result = relu(batch_tensor)
        
        assert_shape_preserved(batch_tensor, result)
        assert np.all(result.data >= 0.0)
        
        # Verify independent processing
        for i in range(batch_tensor.shape[0]):
            sample_result = relu(Tensor(batch_tensor.data[i]))
            assert np.array_equal(result.data[i], sample_result.data)
        
        expected = np.array([
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.5]
        ], dtype=np.float32)
        assert np.array_equal(result.data, expected)
