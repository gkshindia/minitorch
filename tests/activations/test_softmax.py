"""
Softmax Activation Function Tests

Comprehensive test suite for Softmax activation with focus on:
- Probability distribution conversion (sums to 1.0)
- Numerical stability with large values
- Multi-dimensional handling (1D, 2D, 3D tensors)
- Dimensionality parameter handling
- Used extensively in classification tasks (output layer)
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import Softmax
from tests.activations.conftest import TOLERANCE, assert_shape_preserved, assert_finite


class TestSoftmaxBasics:
    """Basic functionality tests for Softmax."""
    
    def test_softmax_simple_case(self):
        """Softmax basic operation on simple vector."""
        softmax = Softmax()
        x = Tensor([1.0, 2.0, 3.0])
        result = softmax(x)
        
        # All values should be positive
        assert np.all(result.data > 0.0)
        # All values should be < 1
        assert np.all(result.data < 1.0)
    
    def test_softmax_probability_distribution(self):
        """Softmax output sums to 1.0 (probability distribution)."""
        softmax = Softmax()
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        result = softmax(x)
        
        # Sum should be 1.0 (within tolerance)
        total = np.sum(result.data)
        assert np.allclose(total, 1.0, atol=TOLERANCE)
    
    def test_softmax_zero_vector(self):
        """Softmax on all-zero vector should give equal probabilities."""
        softmax = Softmax()
        x = Tensor([0.0, 0.0, 0.0, 0.0])
        result = softmax(x)
        
        # All values should be equal (1/4 = 0.25)
        expected = 0.25
        assert np.allclose(result.data, expected, atol=TOLERANCE)
        
        # Sum to 1
        assert np.allclose(np.sum(result.data), 1.0, atol=TOLERANCE)


class TestSoftmaxProbabilityProperty:
    """Tests for probability distribution property."""
    
    def test_softmax_sums_to_one_1d(self):
        """Softmax 1D vector sums to 1."""
        softmax = Softmax()
        x = Tensor(np.random.randn(10))
        result = softmax(x)
        
        assert np.allclose(np.sum(result.data), 1.0, atol=TOLERANCE)
    
    def test_softmax_sums_to_one_2d(self):
        """Softmax 2D matrix: each row sums to 1."""
        softmax = Softmax()
        x = Tensor(np.random.randn(4, 5))
        result = softmax.forward(x, dim=1)  # Sum along columns
        
        row_sums = np.sum(result.data, axis=1)
        assert np.allclose(row_sums, 1.0, atol=TOLERANCE)
    
    def test_softmax_sums_to_one_3d(self):
        """Softmax 3D tensor: sums to 1 along specified dimension."""
        softmax = Softmax()
        x = Tensor(np.random.randn(2, 3, 4))
        result = softmax.forward(x, dim=2)  # Sum along last dimension
        
        # Each slice along last dimension should sum to 1
        for i in range(2):
            for j in range(3):
                total = np.sum(result.data[i, j, :])
                assert np.allclose(total, 1.0, atol=TOLERANCE)
    
    def test_softmax_probabilities_bounded(self):
        """All softmax outputs are in [0, 1]."""
        softmax = Softmax()
        np.random.seed(42)
        x = Tensor(np.random.randn(100) * 10)
        result = softmax(x)
        
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)


class TestSoftmaxMathematical:
    """Mathematical properties of Softmax."""
    
    def test_softmax_proportional_to_exp(self):
        """Softmax(x_i) ∝ exp(x_i)."""
        softmax = Softmax()
        x = Tensor([1.0, 2.0, 3.0])
        result = softmax(x)
        
        # Larger input → larger output
        assert result.data[0] < result.data[1] < result.data[2]
    
    def test_softmax_max_gets_highest_prob(self):
        """Maximum input gets highest probability."""
        softmax = Softmax()
        x = Tensor([1.0, 5.0, 2.0, 3.0])
        result = softmax(x)
        
        # Index 1 (value 5) should have highest probability
        max_idx = np.argmax(result.data)
        assert max_idx == 1
    
    def test_softmax_min_gets_lowest_prob(self):
        """Minimum input gets lowest probability."""
        softmax = Softmax()
        x = Tensor([1.0, 5.0, 2.0, -3.0])
        result = softmax(x)
        
        # Index 3 (value -3) should have lowest probability
        min_idx = np.argmin(result.data)
        assert min_idx == 3
    
    def test_softmax_known_values_simple(self):
        """Softmax on simple inputs with known approximations."""
        softmax = Softmax()
        
        # [0, 0] should give [0.5, 0.5]
        x = Tensor([0.0, 0.0])
        result = softmax(x)
        assert np.allclose(result.data, [0.5, 0.5], atol=TOLERANCE)
        
        # [1, -1] should be symmetric
        x = Tensor([1.0, -1.0])
        result = softmax(x)
        assert np.allclose(result.data[0], 1.0 - result.data[1], atol=TOLERANCE)


class TestSoftmaxNumerical:
    """Numerical stability tests."""
    
    def test_softmax_large_positive_stability(self):
        """Softmax handles large positive values without overflow."""
        softmax = Softmax()
        x = Tensor([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        
        # Should sum to 1 despite large values
        assert np.allclose(np.sum(result.data), 1.0, atol=1e-5)
        assert_finite(result)
    
    def test_softmax_large_negative_stability(self):
        """Softmax handles large negative values without underflow."""
        softmax = Softmax()
        x = Tensor([-1000.0, -1001.0, -1002.0])
        result = softmax(x)
        
        # Should sum to 1 despite large negative values
        assert np.allclose(np.sum(result.data), 1.0, atol=1e-5)
        assert_finite(result)
    
    def test_softmax_extreme_values_mixed(self):
        """Softmax handles mix of extreme values."""
        softmax = Softmax()
        x = Tensor([-500.0, 0.0, 500.0])
        result = softmax(x)
        
        # Extreme positive should dominate
        assert result.data[2] > 0.99
        assert result.data[0] < 0.01
        
        # Sum to 1
        assert np.allclose(np.sum(result.data), 1.0, atol=1e-5)
        assert_finite(result)


class TestSoftmaxDimensionality:
    """Tests for multi-dimensional tensor handling."""
    
    def test_softmax_1d_tensor(self):
        """Softmax on 1D tensor."""
        softmax = Softmax()
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = softmax(x)
        
        assert_shape_preserved(x, result)
        assert np.allclose(np.sum(result.data), 1.0, atol=TOLERANCE)
    
    def test_softmax_2d_tensor_rows(self):
        """Softmax on 2D tensor along rows (dim=1)."""
        softmax = Softmax()
        x = Tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        result = softmax.forward(x, dim=1)  # Sum along columns
        
        assert_shape_preserved(x, result)
        
        # Each row should sum to 1
        for i in range(3):
            assert np.allclose(np.sum(result.data[i, :]), 1.0, atol=TOLERANCE)
    
    def test_softmax_2d_tensor_columns(self):
        """Softmax on 2D tensor along columns (dim=0)."""
        softmax = Softmax()
        x = Tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        result = softmax.forward(x, dim=0)  # Sum along rows
        
        assert_shape_preserved(x, result)
        
        # Each column should sum to 1
        for j in range(3):
            assert np.allclose(np.sum(result.data[:, j]), 1.0, atol=TOLERANCE)
    
    def test_softmax_3d_tensor(self):
        """Softmax on 3D tensor."""
        softmax = Softmax()
        x = Tensor(np.random.randn(2, 3, 4))
        result = softmax(x, dim=2)
        
        assert_shape_preserved(x, result)
        
        # Each depth slice should sum to 1
        for i in range(2):
            for j in range(3):
                assert np.allclose(np.sum(result.data[i, j, :]), 1.0, atol=TOLERANCE)
    
    def test_softmax_default_dimension(self):
        """Softmax uses last dimension by default (dim=-1)."""
        softmax = Softmax()
        x = Tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        result_default = softmax(x)  # Uses dim=-1
        result_explicit = softmax(x, dim=-1)
        
        assert np.allclose(result_default.data, result_explicit.data, atol=TOLERANCE)


class TestSoftmaxProperties:
    """Shape and range properties."""
    
    def test_softmax_preserves_shape(self):
        """Softmax preserves input shape."""
        softmax = Softmax()
        shapes = [
            (5,),
            (3, 4),
            (2, 3, 4),
            (2, 3, 4, 5)
        ]
        
        for shape in shapes:
            x = Tensor(np.random.randn(*shape))
            result = softmax(x)
            assert result.shape == x.shape
    
    def test_softmax_output_positive(self):
        """All softmax outputs are non-negative (>= 0)."""
        softmax = Softmax()
        np.random.seed(42)
        # Use smaller range to avoid extreme underflow
        x = Tensor(np.random.randn(50) * 10)
        result = softmax(x)
        
        assert np.all(result.data >= 0.0)
    
    def test_softmax_output_less_than_one(self):
        """All softmax outputs are <= 1."""
        softmax = Softmax()
        np.random.seed(42)
        # Use smaller range to avoid extreme values
        x = Tensor(np.random.randn(50) * 10)
        result = softmax(x)
        
        assert np.all(result.data <= 1.0)


class TestSoftmaxAPI:
    """Test API consistency."""
    
    def test_softmax_call_vs_forward(self):
        """__call__ and forward should give identical results."""
        softmax = Softmax()
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        
        result_forward = softmax.forward(x)
        result_call = softmax(x)
        
        assert np.allclose(result_forward.data, result_call.data, atol=TOLERANCE)
    
    def test_softmax_forward_with_dim(self):
        """Forward method supports dim parameter."""
        softmax = Softmax()
        x = Tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        result = softmax.forward(x, dim=1)
        
        # Each row should sum to 1
        for i in range(2):
            assert np.allclose(np.sum(result.data[i, :]), 1.0, atol=TOLERANCE)
    
    def test_softmax_repr(self):
        """String representation should contain 'Softmax'."""
        softmax = Softmax()
        repr_str = repr(softmax)
        
        assert "Softmax" in repr_str


class TestSoftmaxUseCases:
    """Real-world usage scenarios."""
    
    def test_softmax_multiclass_classification(self):
        """Softmax for multiclass classification output layer."""
        softmax = Softmax()
        
        # Simulate logits from 3-class classifier (batch_size=4)
        logits = Tensor([
            [2.0, 1.0, 0.5],   # Sample 1: Class 0 most likely
            [0.5, 2.0, 1.0],   # Sample 2: Class 1 most likely
            [1.0, 1.0, 2.0],   # Sample 3: Class 2 most likely
            [0.1, 0.2, 0.3]    # Sample 4: Class 2 most likely
        ])
        
        probabilities = softmax(logits, dim=1)
        
        # Shape preserved
        assert probabilities.shape == logits.shape
        
        # Each row is probability distribution
        for i in range(4):
            row_sum = np.sum(probabilities.data[i, :])
            assert np.allclose(row_sum, 1.0, atol=TOLERANCE)
        
        # Predictions match logits
        pred_logits = np.argmax(logits.data, axis=1)
        pred_probs = np.argmax(probabilities.data, axis=1)
        assert np.array_equal(pred_logits, pred_probs)
    
    def test_softmax_sequence_classification(self):
        """Softmax for sequence labeling (batch × seq_len × classes)."""
        softmax = Softmax()
        
        # Shape: (batch_size=2, seq_length=5, num_classes=3)
        logits = Tensor(np.random.randn(2, 5, 3))
        
        # Apply softmax to class dimension (dim=2)
        probabilities = softmax(logits, dim=2)
        
        # Shape preserved
        assert probabilities.shape == logits.shape
        
        # Each position sums to 1
        for i in range(2):
            for j in range(5):
                assert np.allclose(np.sum(probabilities.data[i, j, :]), 1.0, atol=TOLERANCE)
    
    def test_softmax_one_hot_conversion(self):
        """Softmax with one-hot encoded output."""
        softmax = Softmax()
        
        # All-zero vector except one element
        x = Tensor([100.0, 0.0, 0.0, 0.0])
        result = softmax(x)
        
        # Should be close to one-hot [1, 0, 0, 0]
        assert result.data[0] > 0.99
        assert np.all(result.data[1:] < 0.01)
    
    def test_softmax_batch_processing(self, batch_tensor):
        """Softmax processes batches correctly."""
        softmax = Softmax()
        result = softmax(batch_tensor, dim=1)
        
        assert_shape_preserved(batch_tensor, result)
        assert_finite(result)
        
        # Each row sums to 1
        for i in range(batch_tensor.shape[0]):
            assert np.allclose(np.sum(result.data[i, :]), 1.0, atol=TOLERANCE)
    
    def test_softmax_attention_weights(self):
        """Softmax for generating attention weights."""
        softmax = Softmax()
        
        # Simulate attention scores (queries × keys)
        # Shape: (batch=2, seq_len=4, seq_len=4)
        attention_scores = Tensor(np.random.randn(2, 4, 4))
        
        # Apply softmax to get attention weights
        attention_weights = softmax(attention_scores, dim=2)
        
        # Each query should have weights that sum to 1
        for b in range(2):
            for q in range(4):
                weight_sum = np.sum(attention_weights.data[b, q, :])
                assert np.allclose(weight_sum, 1.0, atol=TOLERANCE)
        
        # All weights should be in [0, 1]
        assert np.all(attention_weights.data >= 0.0)
        assert np.all(attention_weights.data <= 1.0)


class TestSoftmaxComparison:
    """Comparison with other activations."""
    
    def test_softmax_vs_sigmoid_batch(self):
        """Softmax handles batches differently than element-wise Sigmoid."""
        from core.activations import Sigmoid
        softmax = Softmax()
        sigmoid = Sigmoid()
        
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        
        softmax_result = softmax(x)
        sigmoid_result = sigmoid(x)
        
        # Sigmoid is element-wise (doesn't constrain sum)
        assert not np.allclose(np.sum(sigmoid_result.data), 1.0)
        
        # Softmax constrains sum to 1
        assert np.allclose(np.sum(softmax_result.data), 1.0, atol=TOLERANCE)
    
    def test_softmax_monotonicity_along_dim(self):
        """Softmax preserves relative order of inputs within a sample."""
        softmax = Softmax()
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = softmax(x)
        
        # If x is sorted, softmax output should also be sorted
        assert np.all(np.diff(result.data) > 0)
