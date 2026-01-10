"""
Tanh Activation Function Tests

Comprehensive test suite for Tanh activation with focus on:
- Odd symmetry: tanh(-x) = -tanh(x)
- Bounded output in [-1, 1] range
- Zero-centered activation (unlike sigmoid)
- Use in RNNs and LSTMs to maintain bounded hidden states
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import Tanh
from tests.activations.conftest import TOLERANCE, assert_shape_preserved, assert_bounded, assert_finite, assert_monotonic_increasing, assert_symmetry


class TestTanhBasics:
    """Basic functionality tests for Tanh."""
    
    def test_tanh_zero_point(self):
        """Tanh at zero should be 0."""
        tanh = Tanh()
        x = Tensor([0.0])
        result = tanh(x)
        assert np.allclose(result.data, 0.0, atol=TOLERANCE)
    
    def test_tanh_positive_values(self):
        """Tanh output > 0 for positive inputs."""
        tanh = Tanh()
        x = Tensor([1.0, 2.0, 3.0])
        result = tanh(x)
        assert np.all(result.data > 0.0)
        assert np.all(result.data < 1.0)
    
    def test_tanh_negative_values(self):
        """Tanh output < 0 for negative inputs."""
        tanh = Tanh()
        x = Tensor([-1.0, -2.0, -3.0])
        result = tanh(x)
        assert np.all(result.data < 0.0)
        assert np.all(result.data > -1.0)


class TestTanhMathematical:
    """Mathematical properties of Tanh."""
    
    def test_tanh_zero(self):
        """tanh(0) = 0"""
        tanh = Tanh()
        x = Tensor([0.0])
        result = tanh(x)
        assert np.allclose(result.data, 0.0, atol=TOLERANCE)
    
    def test_tanh_log_two(self):
        """tanh(ln(2)) ≈ 0.6"""
        tanh = Tanh()
        x = Tensor([np.log(2)])
        result = tanh(x)
        assert np.allclose(result.data, 0.6, atol=TOLERANCE)
    
    def test_tanh_extreme_limits(self):
        """tanh(±∞) → ±1"""
        tanh = Tanh()
        large_positive = Tensor([100.0])
        large_negative = Tensor([-100.0])
        assert np.allclose(tanh(large_positive).data, 1.0, atol=0.01)
        assert np.allclose(tanh(large_negative).data, -1.0, atol=0.01)


class TestTanhSymmetry:
    """Odd symmetry property: tanh(-x) = -tanh(x)."""
    
    def test_tanh_symmetry_single_values(self):
        """Symmetry with individual values."""
        tanh = Tanh()
        test_values = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        for val in test_values:
            x_pos = Tensor([val])
            x_neg = Tensor([-val])
            result_pos = tanh(x_pos)
            result_neg = tanh(x_neg)
            assert_symmetry(result_pos.data[0], result_neg.data[0], atol=TOLERANCE)
    
    def test_tanh_symmetry_array(self):
        """Symmetry verified across arrays."""
        tanh = Tanh()
        x = Tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        result = tanh(x)
        
        # Check symmetry pairs
        assert np.allclose(result.data[0], -result.data[6], atol=TOLERANCE)  # -5 and 5
        assert np.allclose(result.data[1], -result.data[5], atol=TOLERANCE)  # -2 and 2
        assert np.allclose(result.data[2], -result.data[4], atol=TOLERANCE)  # -1 and 1
        assert np.allclose(result.data[3], 0.0, atol=TOLERANCE)  # 0


class TestTanhNumerical:
    """Numerical stability tests for Tanh."""
    
    def test_tanh_large_positive(self):
        """Tanh saturates at 1 for large positive values."""
        tanh = Tanh()
        large_positive = Tensor([100.0, 500.0, 700.0])
        result = tanh(large_positive)
        assert np.all(result.data > 0.99)
        assert np.all(result.data <= 1.0)
        assert_finite(result)
    
    def test_tanh_large_negative(self):
        """Tanh saturates at -1 for large negative values."""
        tanh = Tanh()
        large_negative = Tensor([-100.0, -500.0, -700.0])
        result = tanh(large_negative)
        assert np.all(result.data < -0.99)
        assert np.all(result.data >= -1.0)
        assert_finite(result)


class TestTanhProperties:
    """Shape and range properties."""
    
    def test_tanh_shape_1d(self):
        """Tanh preserves 1D shape."""
        tanh = Tanh()
        x = Tensor([1.0, -2.0, 3.0])
        result = tanh(x)
        assert_shape_preserved(x, result)
    
    def test_tanh_shape_2d(self):
        """Tanh preserves 2D shape."""
        tanh = Tanh()
        x = Tensor([[1.0, -2.0], [-3.0, 4.0]])
        result = tanh(x)
        assert_shape_preserved(x, result)
    
    def test_tanh_shape_3d(self):
        """Tanh preserves 3D shape."""
        tanh = Tanh()
        x = Tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
        result = tanh(x)
        assert_shape_preserved(x, result)
    
    def test_tanh_output_range(self):
        """Tanh output always in [-1, 1]."""
        tanh = Tanh()
        np.random.seed(42)
        x = Tensor(np.random.randn(100) * 10)
        result = tanh(x)
        assert_bounded(result, -1.0, 1.0)


class TestTanhMonotonicity:
    """Test monotonicity property."""
    
    def test_tanh_monotonic_increasing(self):
        """Tanh is monotonically increasing."""
        tanh = Tanh()
        x_values = np.linspace(-10, 10, 100)
        x = Tensor(x_values)
        result = tanh(x)
        assert_monotonic_increasing(result.data)


class TestTanhAPI:
    """Test API consistency."""
    
    def test_tanh_call_vs_forward(self):
        """__call__ and forward should give identical results."""
        tanh = Tanh()
        x = Tensor([1.0, -2.0, 3.0, -4.0, 5.0])
        result_forward = tanh.forward(x)
        result_call = tanh(x)
        assert np.allclose(result_forward.data, result_call.data, atol=TOLERANCE)
    
    def test_tanh_repr(self):
        """String representation should contain 'Tanh'."""
        tanh = Tanh()
        repr_str = repr(tanh)
        assert "Tanh" in repr_str


class TestTanhUseCases:
    """Real-world usage scenarios."""
    
    def test_tanh_rnn_hidden_states(self):
        """Tanh for bounded hidden states in RNNs/LSTMs."""
        tanh = Tanh()
        
        # Simulate hidden states
        hidden_states = Tensor([
            [0.5, -0.3, 0.8, -0.2],   # Time step 1
            [1.5, -1.2, 0.1, -0.5],   # Time step 2
            [2.0, -2.5, 0.0, 1.0],    # Time step 3
        ])
        
        activated_states = tanh(hidden_states)
        
        # Verify all values bounded
        assert np.all(activated_states.data >= -1.0)
        assert np.all(activated_states.data <= 1.0)
        assert_shape_preserved(hidden_states, activated_states)
        
        # Check extreme value squashing
        assert activated_states.data[2, 0] > 0.95   # tanh(2.0) ≈ 0.96
        assert activated_states.data[2, 1] < -0.95  # tanh(-2.5) ≈ -0.99
    
    def test_tanh_batch_processing(self, batch_tensor):
        """Tanh processes batches correctly."""
        tanh = Tanh()
        result = tanh(batch_tensor)
        
        assert_shape_preserved(batch_tensor, result)
        assert_bounded(result, -1.0, 1.0)
        
        # Verify independent processing
        for i in range(batch_tensor.shape[0]):
            sample_result = tanh(Tensor(batch_tensor.data[i]))
            assert np.allclose(result.data[i], sample_result.data, atol=TOLERANCE)
