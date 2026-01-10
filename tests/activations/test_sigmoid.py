"""
Sigmoid Activation Function Tests

Comprehensive test suite for Sigmoid activation with focus on:
- Basic functionality and mathematical properties
- Numerical stability with extreme values
- Shape preservation and batching
- Real-world use cases (binary classification)
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import Sigmoid
from tests.activations.conftest import TOLERANCE, assert_shape_preserved, assert_bounded, assert_finite, assert_monotonic_increasing


class TestSigmoidBasics:
    """Basic functionality tests for Sigmoid."""
    
    def test_sigmoid_zero_point(self):
        """Sigmoid at zero should be 0.5."""
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        result = sigmoid(x)
        assert np.allclose(result.data, 0.5, atol=TOLERANCE)
    
    def test_sigmoid_positive_values(self):
        """Sigmoid output > 0.5 for positive inputs."""
        sigmoid = Sigmoid()
        x = Tensor([1.0, 2.0, 3.0])
        result = sigmoid(x)
        assert np.all(result.data > 0.5)
        assert np.all(result.data < 1.0)
    
    def test_sigmoid_negative_values(self):
        """Sigmoid output < 0.5 for negative inputs."""
        sigmoid = Sigmoid()
        x = Tensor([-1.0, -2.0, -3.0])
        result = sigmoid(x)
        assert np.all(result.data < 0.5)
        assert np.all(result.data > 0.0)


class TestSigmoidMathematical:
    """Mathematical properties of Sigmoid."""
    
    def test_sigmoid_zero(self):
        """σ(0) = 0.5"""
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        result = sigmoid(x)
        assert np.allclose(result.data, 0.5, atol=TOLERANCE)
    
    def test_sigmoid_log_three(self):
        """σ(ln(3)) ≈ 0.75"""
        sigmoid = Sigmoid()
        x = Tensor([np.log(3)])
        result = sigmoid(x)
        assert np.allclose(result.data, 0.75, atol=TOLERANCE)
    
    def test_sigmoid_neg_log_three(self):
        """σ(-ln(3)) ≈ 0.25"""
        sigmoid = Sigmoid()
        x = Tensor([-np.log(3)])
        result = sigmoid(x)
        assert np.allclose(result.data, 0.25, atol=TOLERANCE)
    
    def test_sigmoid_symmetry(self):
        """σ(x) + σ(-x) = 1 (complementary property)."""
        sigmoid = Sigmoid()
        x_val = 2.5
        x_pos = Tensor([x_val])
        x_neg = Tensor([-x_val])
        result_pos = sigmoid(x_pos)
        result_neg = sigmoid(x_neg)
        assert np.allclose(result_pos.data + result_neg.data, 1.0, atol=TOLERANCE)


class TestSigmoidNumerical:
    """Numerical stability tests for Sigmoid."""
    
    def test_sigmoid_large_positive(self, extreme_values):
        """Sigmoid saturates at 1 for large positive values."""
        sigmoid = Sigmoid()
        large_positive = Tensor([100.0, 500.0, 700.0])
        result = sigmoid(large_positive)
        assert np.all(result.data > 0.999)
        assert_finite(result)
    
    def test_sigmoid_large_negative(self, extreme_values):
        """Sigmoid saturates at 0 for large negative values."""
        sigmoid = Sigmoid()
        large_negative = Tensor([-100.0, -500.0, -700.0])
        result = sigmoid(large_negative)
        assert np.all(result.data < 0.001)
        assert np.all(result.data >= 0.0)
        assert_finite(result)


class TestSigmoidProperties:
    """Shape and range properties."""
    
    def test_sigmoid_shape_1d(self):
        """Sigmoid preserves 1D shape."""
        sigmoid = Sigmoid()
        x = Tensor([1.0, 2.0, 3.0])
        result = sigmoid(x)
        assert_shape_preserved(x, result)
    
    def test_sigmoid_shape_2d(self):
        """Sigmoid preserves 2D shape."""
        sigmoid = Sigmoid()
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = sigmoid(x)
        assert_shape_preserved(x, result)
    
    def test_sigmoid_shape_3d(self):
        """Sigmoid preserves 3D shape."""
        sigmoid = Sigmoid()
        x = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = sigmoid(x)
        assert_shape_preserved(x, result)
    
    def test_sigmoid_output_range(self):
        """Sigmoid output always in (0, 1)."""
        sigmoid = Sigmoid()
        np.random.seed(42)
        x = Tensor(np.random.randn(100) * 10)
        result = sigmoid(x)
        assert_bounded(result, 0.0, 1.0)


class TestSigmoidMonotonicity:
    """Test monotonicity property."""
    
    def test_sigmoid_monotonic_increasing(self):
        """Sigmoid is monotonically increasing."""
        sigmoid = Sigmoid()
        x_values = np.linspace(-10, 10, 100)
        x = Tensor(x_values)
        result = sigmoid(x)
        assert_monotonic_increasing(result.data)


class TestSigmoidAPI:
    """Test API consistency."""
    
    def test_sigmoid_call_vs_forward(self):
        """__call__ and forward should give identical results."""
        sigmoid = Sigmoid()
        x = Tensor([1.0, 2.0, 3.0, -1.0, -2.0])
        result_forward = sigmoid.forward(x)
        result_call = sigmoid(x)
        assert np.allclose(result_forward.data, result_call.data, atol=TOLERANCE)
    
    def test_sigmoid_repr(self):
        """String representation should contain 'Sigmoid'."""
        sigmoid = Sigmoid()
        repr_str = repr(sigmoid)
        assert "Sigmoid" in repr_str


class TestSigmoidUseCases:
    """Real-world usage scenarios."""
    
    def test_sigmoid_binary_classification(self):
        """Sigmoid for binary classification."""
        sigmoid = Sigmoid()
        
        # Positive logits → high probability
        logits_positive = Tensor([2.0, 3.5, 5.0])
        probs_positive = sigmoid(logits_positive)
        assert np.all(probs_positive.data > 0.5)
        
        # Negative logits → low probability
        logits_negative = Tensor([-2.0, -3.5, -5.0])
        probs_negative = sigmoid(logits_negative)
        assert np.all(probs_negative.data < 0.5)
        
        # Decision boundary at 0.5
        logits = Tensor([-1.5, 0.5, 2.0, -0.3])
        probs = sigmoid(logits)
        predictions = (probs.data > 0.5).astype(int)
        expected = np.array([0, 1, 1, 0])
        assert np.array_equal(predictions, expected)
    
    def test_sigmoid_batch_processing(self, batch_tensor):
        """Sigmoid processes batches correctly."""
        sigmoid = Sigmoid()
        result = sigmoid(batch_tensor)
        
        assert_shape_preserved(batch_tensor, result)
        assert_bounded(result, 0.0, 1.0)
        
        # Verify independent processing
        for i in range(batch_tensor.shape[0]):
            sample_result = sigmoid(Tensor(batch_tensor.data[i]))
            assert np.allclose(result.data[i], sample_result.data, atol=TOLERANCE)
