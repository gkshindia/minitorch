"""
GELU Activation Function Tests

Comprehensive test suite for GELU activation with focus on:
- Smooth approximation of ReLU (no hard cutoff like ReLU)
- Used in modern transformers (BERT, GPT, etc.)
- Unbounded positive output, bounded negative
- Mathematical smoothness (differentiable everywhere)
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import GELU as Gelu, ReLU
from tests.activations.conftest import TOLERANCE, assert_shape_preserved, assert_finite


class TestGeluBasics:
    """Basic functionality tests for GELU."""
    
    def test_gelu_zero_point(self):
        """GELU at zero should be ≈0."""
        gelu = Gelu()
        x = Tensor([0.0])
        result = gelu(x)
        assert np.allclose(result.data, 0.0, atol=0.01)
    
    def test_gelu_positive_values(self):
        """GELU output > 0 for positive inputs."""
        gelu = Gelu()
        x = Tensor([1.0, 2.0, 3.0])
        result = gelu(x)
        assert np.all(result.data > 0.0)
    
    def test_gelu_negative_values(self):
        """GELU output < 0 for small negative inputs."""
        gelu = Gelu()
        x = Tensor([-1.0, -0.5, -0.1])
        result = gelu(x)
        # GELU can be negative for negative values
        assert result.data[0] < 0.0  # GELU(-1.0) < 0
    
    def test_gelu_large_positive_identity(self):
        """GELU(x) ≈ x for large positive values."""
        gelu = Gelu()
        x = Tensor([10.0])
        result = gelu(x)
        assert np.allclose(result.data, 10.0, rtol=0.01)


class TestGeluMathematical:
    """Mathematical properties of GELU."""
    
    def test_gelu_zero(self):
        """GELU(0) ≈ 0"""
        gelu = Gelu()
        x = Tensor([0.0])
        result = gelu(x)
        assert np.allclose(result.data, 0.0, atol=0.01)
    
    def test_gelu_one(self):
        """GELU(1) ≈ 0.841"""
        gelu = Gelu()
        x = Tensor([1.0])
        result = gelu(x)
        expected = 0.841
        assert np.allclose(result.data, expected, atol=0.01)
    
    def test_gelu_two(self):
        """GELU(2) ≈ 1.936 (nearly identity)"""
        gelu = Gelu()
        x = Tensor([2.0])
        result = gelu(x)
        expected = 1.936
        assert np.allclose(result.data, expected, atol=0.02)
    
    def test_gelu_negative_one(self):
        """GELU(-1) ≈ -0.159"""
        gelu = Gelu()
        x = Tensor([-1.0])
        result = gelu(x)
        expected = -0.159
        assert np.allclose(result.data, expected, atol=0.01)


class TestGeluSmoothness:
    """Smoothness properties (no sharp transitions)."""
    
    def test_gelu_smooth_near_zero(self):
        """GELU is smooth around zero (unlike ReLU)."""
        gelu = Gelu()
        
        # Generate smooth values around zero
        x_values = np.linspace(-0.5, 0.5, 100)
        x = Tensor(x_values)
        result = gelu(x)
        
        # Check for smooth transitions (no sharp discontinuity)
        gelu_diffs = np.abs(np.diff(result.data))
        max_gelu_diff = np.max(gelu_diffs)
        
        # GELU should have small max difference
        assert max_gelu_diff < 0.1, "GELU should be smooth"
    
    def test_gelu_smoother_than_relu(self):
        """GELU should be smoother than ReLU near zero."""
        gelu = Gelu()
        relu = ReLU()
        
        x_values = np.linspace(-0.5, 0.5, 100)
        x = Tensor(x_values)
        
        gelu_result = gelu(x)
        relu_result = relu(x)
        
        # Compare smoothness
        gelu_diffs = np.abs(np.diff(gelu_result.data))
        relu_diffs = np.abs(np.diff(relu_result.data))
        
        # GELU should have smoother behavior
        assert np.max(gelu_diffs) < np.max(relu_diffs)


class TestGeluApproximation:
    """GELU as smooth approximation of ReLU."""
    
    def test_gelu_approximates_relu_large_positive(self):
        """GELU ≈ ReLU for large positive values."""
        gelu = Gelu()
        relu = ReLU()
        
        large_positive = Tensor([5.0, 10.0, 20.0, 50.0])
        gelu_result = gelu(large_positive)
        relu_result = relu(large_positive)
        
        # Should be very close
        assert np.allclose(gelu_result.data, relu_result.data, rtol=0.01)
        assert np.allclose(gelu_result.data, large_positive.data, rtol=0.01)


class TestGeluProperties:
    """Shape and range properties."""
    
    def test_gelu_shape_1d(self):
        """GELU preserves 1D shape."""
        gelu = Gelu()
        x = Tensor([1.0, -2.0, 3.0])
        result = gelu(x)
        assert_shape_preserved(x, result)
    
    def test_gelu_shape_2d(self):
        """GELU preserves 2D shape."""
        gelu = Gelu()
        x = Tensor([[1.0, -2.0], [-3.0, 4.0]])
        result = gelu(x)
        assert_shape_preserved(x, result)
    
    def test_gelu_shape_3d(self):
        """GELU preserves 3D shape."""
        gelu = Gelu()
        x = Tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
        result = gelu(x)
        assert_shape_preserved(x, result)


class TestGeluNumerical:
    """Numerical stability tests."""
    
    def test_gelu_large_positive_numerical(self):
        """GELU handles large positive values without overflow."""
        gelu = Gelu()
        large_positive = Tensor([100.0, 500.0, 700.0])
        result = gelu(large_positive)
        
        assert np.allclose(result.data, large_positive.data, rtol=0.01)
        assert_finite(result)
    
    def test_gelu_large_negative_numerical(self):
        """GELU handles large negative values without overflow."""
        gelu = Gelu()
        large_negative = Tensor([-100.0, -500.0, -700.0])
        result = gelu(large_negative)
        
        assert np.all(result.data < 0.01)
        assert np.all(result.data > -1.0)
        assert_finite(result)


class TestGeluAPI:
    """Test API consistency."""
    
    def test_gelu_call_vs_forward(self):
        """__call__ and forward should give identical results."""
        gelu = Gelu()
        x = Tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        result_forward = gelu.forward(x)
        result_call = gelu(x)
        assert np.allclose(result_forward.data, result_call.data, atol=TOLERANCE)
    
    def test_gelu_repr(self):
        """String representation should contain 'GELU' or 'Gelu'."""
        gelu = Gelu()
        repr_str = repr(gelu)
        assert "GELU" in repr_str or "Gelu" in repr_str


class TestGeluUseCases:
    """Real-world usage scenarios."""
    
    def test_gelu_transformer_ffn(self):
        """GELU in transformer feed-forward network (BERT/GPT)."""
        gelu = Gelu()
        
        # Simulate FFN pre-activation values
        ffn_hidden = Tensor([
            [2.0, -1.5, 0.5, -0.3],   # Token 1
            [1.0, 0.0, -2.0, 3.0],    # Token 2
            [-0.5, 0.8, 1.5, -1.0],   # Token 3
        ])
        
        activated = gelu(ffn_hidden)
        
        # Verify shape preservation
        assert_shape_preserved(ffn_hidden, activated)
        assert_finite(activated)
        
        # Check approximate behavior
        assert activated.data[0, 0] > 1.9   # GELU(2.0) ≈ 1.95
        assert activated.data[1, 2] < 0.0   # GELU(-2.0) < 0
    
    def test_gelu_batch_processing(self, batch_tensor):
        """GELU processes batches correctly."""
        gelu = Gelu()
        result = gelu(batch_tensor)
        
        assert_shape_preserved(batch_tensor, result)
        assert_finite(result)
        
        # Verify independent processing
        for i in range(batch_tensor.shape[0]):
            sample_result = gelu(Tensor(batch_tensor.data[i]))
            assert np.allclose(result.data[i], sample_result.data, atol=TOLERANCE)
        
        # Check specific values
        assert np.allclose(result.data[1], [0.0, 0.0, 0.0], atol=0.01)  # GELU(0) ≈ 0
