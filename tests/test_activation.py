import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import Sigmoid

TOLERANCE = 1e-6  # Tolerance for floating point comparisons


def test_sigmoid_basic():
    """🧪 Test basic sigmoid activation functionality."""
    print("🧪 Unit Test: Sigmoid Basic Functionality...")
    
    # Test simple case: sigmoid(0) = 0.5
    sigmoid = Sigmoid()
    x = Tensor([0.0])
    result = sigmoid(x)
    assert np.allclose(result.data, 0.5, atol=TOLERANCE)
    
    # Test positive values: sigmoid should be > 0.5
    x = Tensor([1.0, 2.0, 3.0])
    result = sigmoid(x)
    assert np.all(result.data > 0.5)
    assert np.all(result.data < 1.0)
    
    # Test negative values: sigmoid should be < 0.5
    x = Tensor([-1.0, -2.0, -3.0])
    result = sigmoid(x)
    assert np.all(result.data < 0.5)
    assert np.all(result.data > 0.0)
    
    print("✅ Basic sigmoid functionality works!")


def test_sigmoid_known_values():
    """🧪 Test sigmoid with known mathematical values."""
    print("🧪 Unit Test: Sigmoid Known Values...")
    
    sigmoid = Sigmoid()
    
    # Test σ(0) = 0.5
    x = Tensor([0.0])
    result = sigmoid(x)
    expected = 0.5
    assert np.allclose(result.data, expected, atol=TOLERANCE)
    
    # Test σ(ln(3)) ≈ 0.75 (since σ(x) = 1/(1+e^(-x)) = 1/(1+e^(-ln(3))) = 1/(1+1/3) = 0.75)
    x = Tensor([np.log(3)])
    result = sigmoid(x)
    expected = 0.75
    assert np.allclose(result.data, expected, atol=TOLERANCE)
    
    # Test σ(-ln(3)) ≈ 0.25
    x = Tensor([-np.log(3)])
    result = sigmoid(x)
    expected = 0.25
    assert np.allclose(result.data, expected, atol=TOLERANCE)
    
    # Test symmetry: σ(x) + σ(-x) = 1
    x_val = 2.5
    x_pos = Tensor([x_val])
    x_neg = Tensor([-x_val])
    result_pos = sigmoid(x_pos)
    result_neg = sigmoid(x_neg)
    assert np.allclose(result_pos.data + result_neg.data, 1.0, atol=TOLERANCE)
    
    print("✅ Sigmoid known values correct!")


def test_sigmoid_numerical_stability():
    """🧪 Test sigmoid numerical stability with extreme values."""
    print("🧪 Unit Test: Sigmoid Numerical Stability...")
    
    sigmoid = Sigmoid()
    
    # Test large positive values (should approach 1.0)
    large_positive = Tensor([100.0, 500.0, 700.0])
    result = sigmoid(large_positive)
    assert np.all(result.data > 0.999), "Large positive values should be close to 1.0"
    assert np.all(np.isfinite(result.data)), "Should not produce inf"
    assert not np.any(np.isnan(result.data)), "Should not produce NaN"
    
    # Test large negative values (should approach 0.0)
    large_negative = Tensor([-100.0, -500.0, -700.0])
    result = sigmoid(large_negative)
    assert np.all(result.data < 0.001), "Large negative values should be close to 0.0"
    assert np.all(result.data >= 0.0), "Should be non-negative (may underflow to 0 in float32)"
    assert np.all(np.isfinite(result.data)), "Should not produce inf"
    assert not np.any(np.isnan(result.data)), "Should not produce NaN"
    
    # Test extreme values at clipping boundary
    extreme = Tensor([-1000.0, 1000.0])
    result = sigmoid(extreme)
    assert np.all(np.isfinite(result.data)), "Extreme values should be handled"
    assert result.data[0] < 0.001, "σ(-1000) should be near 0"
    assert result.data[1] > 0.999, "σ(1000) should be near 1"
    
    print("✅ Sigmoid numerical stability verified!")


def test_sigmoid_shape_preservation():
    """🧪 Test that sigmoid preserves tensor shapes."""
    print("🧪 Unit Test: Sigmoid Shape Preservation...")
    
    sigmoid = Sigmoid()
    
    # Test 1D tensor
    x_1d = Tensor([1.0, 2.0, 3.0])
    result = sigmoid(x_1d)
    assert result.shape == x_1d.shape, f"Expected {x_1d.shape}, got {result.shape}"
    
    # Test 2D tensor (like a batch of features)
    x_2d = Tensor([[1.0, 2.0], [3.0, 4.0]])
    result = sigmoid(x_2d)
    assert result.shape == x_2d.shape, f"Expected {x_2d.shape}, got {result.shape}"
    
    # Test 3D tensor (like batch × sequence × features)
    x_3d = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    result = sigmoid(x_3d)
    assert result.shape == x_3d.shape, f"Expected {x_3d.shape}, got {result.shape}"
    
    # Test single scalar
    x_scalar = Tensor([5.0])
    result = sigmoid(x_scalar)
    assert result.shape == x_scalar.shape, f"Expected {x_scalar.shape}, got {result.shape}"
    
    print("✅ Sigmoid shape preservation works!")


def test_sigmoid_range():
    """🧪 Test that sigmoid output is always in (0, 1) range."""
    print("🧪 Unit Test: Sigmoid Output Range...")
    
    sigmoid = Sigmoid()
    
    # Test with random values
    np.random.seed(42)
    random_values = np.random.randn(100) * 10  # Random values in [-30, 30] roughly
    x = Tensor(random_values)
    result = sigmoid(x)
    
    # Verify all outputs are in [0, 1] (float32 may saturate at edges)
    assert np.all(result.data >= 0.0), "All values should be >= 0"
    assert np.all(result.data <= 1.0), "All values should be <= 1"
    
    # Test edge cases
    edge_cases = Tensor([-100, -10, -1, 0, 1, 10, 100])
    result = sigmoid(edge_cases)
    assert np.all(result.data >= 0.0), "All values should be >= 0"
    assert np.all(result.data <= 1.0), "All values should be <= 1"
    
    print("✅ Sigmoid output range verified!")


def test_sigmoid_monotonicity():
    """🧪 Test that sigmoid is monotonically increasing."""
    print("🧪 Unit Test: Sigmoid Monotonicity...")
    
    sigmoid = Sigmoid()
    
    # Create sorted input values
    x_values = np.linspace(-10, 10, 100)
    x = Tensor(x_values)
    result = sigmoid(x)
    
    # Check that output is monotonically increasing
    # For any i < j, σ(x[i]) < σ(x[j])
    for i in range(len(result.data) - 1):
        assert result.data[i] < result.data[i + 1], \
            f"Sigmoid should be increasing: σ({x_values[i]})={result.data[i]} should be < σ({x_values[i+1]})={result.data[i+1]}"
    
    print("✅ Sigmoid monotonicity verified!")


def test_sigmoid_call_vs_forward():
    """🧪 Test that __call__ and forward produce identical results."""
    print("🧪 Unit Test: Sigmoid __call__ vs forward...")
    
    sigmoid = Sigmoid()
    x = Tensor([1.0, 2.0, 3.0, -1.0, -2.0])
    
    # Test forward method
    result_forward = sigmoid.forward(x)
    
    # Test __call__ method
    result_call = sigmoid(x)
    
    # Should produce identical results
    assert np.allclose(result_forward.data, result_call.data, atol=TOLERANCE)
    
    print("✅ Sigmoid __call__ and forward are consistent!")


def test_sigmoid_binary_classification():
    """🧪 Test sigmoid in binary classification scenario."""
    print("🧪 Unit Test: Sigmoid Binary Classification...")
    
    sigmoid = Sigmoid()
    
    # Simulate logits from a binary classifier
    # Positive logits → high probability (> 0.5)
    # Negative logits → low probability (< 0.5)
    logits_positive = Tensor([2.0, 3.5, 5.0])  # Strong positive predictions
    probs_positive = sigmoid(logits_positive)
    assert np.all(probs_positive.data > 0.5), "Positive logits should give prob > 0.5"
    
    logits_negative = Tensor([-2.0, -3.5, -5.0])  # Strong negative predictions
    probs_negative = sigmoid(logits_negative)
    assert np.all(probs_negative.data < 0.5), "Negative logits should give prob < 0.5"
    
    # Threshold at 0.5 for classification
    logits = Tensor([-1.5, 0.5, 2.0, -0.3])
    probs = sigmoid(logits)
    predictions = (probs.data > 0.5).astype(int)
    expected_predictions = np.array([0, 1, 1, 0])  # Class 0 or 1
    assert np.array_equal(predictions, expected_predictions)
    
    print("✅ Sigmoid binary classification scenario works!")


def test_sigmoid_repr():
    """🧪 Test sigmoid string representation."""
    print("🧪 Unit Test: Sigmoid __repr__...")
    
    sigmoid = Sigmoid()
    repr_str = repr(sigmoid)
    
    assert "Sigmoid" in repr_str, "Repr should contain 'Sigmoid'"
    assert repr_str == "Sigmoid()", f"Expected 'Sigmoid()', got '{repr_str}'"
    
    print("✅ Sigmoid __repr__ works!")


def test_sigmoid_batch_processing():
    """🧪 Test sigmoid with batch processing (common in ML)."""
    print("🧪 Unit Test: Sigmoid Batch Processing...")
    
    sigmoid = Sigmoid()
    
    # Simulate a batch of samples (batch_size=4, features=3)
    batch = Tensor([
        [1.0, -2.0, 3.0],   # Sample 1
        [0.5, 0.0, -0.5],   # Sample 2
        [-1.0, 2.0, -3.0],  # Sample 3
        [10.0, -10.0, 0.0]  # Sample 4 (extreme values)
    ])
    
    result = sigmoid(batch)
    
    # Verify shape is preserved
    assert result.shape == batch.shape
    
    # Verify all values in valid range
    assert np.all(result.data > 0.0)
    assert np.all(result.data < 1.0)
    
    # Verify each sample processed independently
    for i in range(batch.shape[0]):
        sample_result = sigmoid(Tensor(batch.data[i]))
        assert np.allclose(result.data[i], sample_result.data, atol=TOLERANCE)
    
    print("✅ Sigmoid batch processing works!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 RUNNING SIGMOID ACTIVATION TESTS")
    print("=" * 60 + "\n")
    
    test_sigmoid_basic()
    test_sigmoid_known_values()
    test_sigmoid_numerical_stability()
    test_sigmoid_shape_preservation()
    test_sigmoid_range()
    test_sigmoid_monotonicity()
    test_sigmoid_call_vs_forward()
    test_sigmoid_binary_classification()
    test_sigmoid_repr()
    test_sigmoid_batch_processing()
    
    print("\n" + "=" * 60)
    print("🎉 ALL SIGMOID TESTS PASSED!")
    print("=" * 60)