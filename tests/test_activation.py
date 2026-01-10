import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import Sigmoid, ReLU

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


# ============================================================================
# ReLU ACTIVATION TESTS
# ============================================================================

def test_relu_basic():
    """🧪 Test basic ReLU activation functionality."""
    print("🧪 Unit Test: ReLU Basic Functionality...")
    
    relu = ReLU()
    
    # Test positive values (should pass through unchanged)
    x = Tensor([1.0, 2.0, 3.0])
    result = relu(x)
    assert np.array_equal(result.data, x.data), "Positive values should pass through"
    
    # Test negative values (should be zeroed)
    x = Tensor([-1.0, -2.0, -3.0])
    result = relu(x)
    assert np.array_equal(result.data, np.zeros(3, dtype=np.float32)), "Negative values should be zero"
    
    # Test zero (boundary case)
    x = Tensor([0.0])
    result = relu(x)
    assert result.data[0] == 0.0, "Zero should remain zero"
    
    # Test mixed values
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    assert np.array_equal(result.data, expected), "Mixed values should be correctly clipped"
    
    print("✅ Basic ReLU functionality works!")


def test_relu_known_values():
    """🧪 Test ReLU with known mathematical values."""
    print("🧪 Unit Test: ReLU Known Values...")
    
    relu = ReLU()
    
    # Test identity for positive values
    positive_vals = [0.5, 1.0, 10.0, 100.0, 1000.0]
    x = Tensor(positive_vals)
    result = relu(x)
    assert np.array_equal(result.data, x.data), "f(x) = x for x > 0"
    
    # Test zero mapping for negative values
    negative_vals = [-0.5, -1.0, -10.0, -100.0, -1000.0]
    x = Tensor(negative_vals)
    result = relu(x)
    assert np.all(result.data == 0.0), "f(x) = 0 for x < 0"
    
    # Test piecewise nature
    x = Tensor([5.0, -5.0])
    result = relu(x)
    assert result.data[0] == 5.0, "Positive value preserved"
    assert result.data[1] == 0.0, "Negative value zeroed"
    
    print("✅ ReLU known values correct!")


def test_relu_shape_preservation():
    """🧪 Test that ReLU preserves tensor shapes."""
    print("🧪 Unit Test: ReLU Shape Preservation...")
    
    relu = ReLU()
    
    # Test 1D tensor
    x_1d = Tensor([1.0, -2.0, 3.0])
    result = relu(x_1d)
    assert result.shape == x_1d.shape, f"Expected {x_1d.shape}, got {result.shape}"
    
    # Test 2D tensor (batch × features)
    x_2d = Tensor([[1.0, -2.0], [-3.0, 4.0]])
    result = relu(x_2d)
    assert result.shape == x_2d.shape, f"Expected {x_2d.shape}, got {result.shape}"
    
    # Test 3D tensor (batch × height × width)
    x_3d = Tensor([[[1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [7.0, -8.0]]])
    result = relu(x_3d)
    assert result.shape == x_3d.shape, f"Expected {x_3d.shape}, got {result.shape}"
    
    # Test single scalar
    x_scalar = Tensor([5.0])
    result = relu(x_scalar)
    assert result.shape == x_scalar.shape, f"Expected {x_scalar.shape}, got {result.shape}"
    
    print("✅ ReLU shape preservation works!")


def test_relu_output_range():
    """🧪 Test that ReLU output is always non-negative."""
    print("🧪 Unit Test: ReLU Output Range...")
    
    relu = ReLU()
    
    # Test with random values
    np.random.seed(42)
    random_values = np.random.randn(100) * 10  # Random values
    x = Tensor(random_values)
    result = relu(x)
    
    # Verify all outputs are non-negative
    assert np.all(result.data >= 0.0), "All ReLU outputs should be >= 0"
    
    # Test extreme values
    extreme = Tensor([-1000.0, -0.001, 0.0, 0.001, 1000.0])
    result = relu(extreme)
    assert np.all(result.data >= 0.0), "Even extreme values should be >= 0"
    
    print("✅ ReLU output range verified!")


def test_relu_linearity_positive():
    """🧪 Test that ReLU is linear for positive values."""
    print("🧪 Unit Test: ReLU Linearity for Positive Values...")
    
    relu = ReLU()
    
    # For x > 0, ReLU(ax) = a*ReLU(x) = ax
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    result_x = relu(x)
    
    # Scale by constant
    a = 2.5
    x_scaled = Tensor(x.data * a)
    result_scaled = relu(x_scaled)
    
    # Should maintain linearity
    expected = result_x.data * a
    assert np.allclose(result_scaled.data, expected, atol=TOLERANCE)
    
    # Test additivity for positive values
    x1 = Tensor([1.0, 2.0, 3.0])
    x2 = Tensor([4.0, 5.0, 6.0])
    result_sum = relu(Tensor(x1.data + x2.data))
    result_individual_sum = relu(x1).data + relu(x2).data
    assert np.allclose(result_sum.data, result_individual_sum, atol=TOLERANCE)
    
    print("✅ ReLU linearity verified!")


def test_relu_sparsity():
    """🧪 Test ReLU sparsity (zero outputs for negative inputs)."""
    print("🧪 Unit Test: ReLU Sparsity...")
    
    relu = ReLU()
    
    # Create input with known number of negative values
    x = Tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    result = relu(x)
    
    # Count zeros (should be 4: three negatives + zero)
    num_zeros = np.sum(result.data == 0.0)
    assert num_zeros == 4, f"Expected 4 zeros, got {num_zeros}"
    
    # Test with mostly negative values (dead ReLU scenario)
    mostly_negative = Tensor([-10.0, -5.0, -1.0, -0.1, -0.01])
    result = relu(mostly_negative)
    assert np.all(result.data == 0.0), "All negative inputs should produce zero"
    
    # Test sparsity percentage
    np.random.seed(123)
    random_data = np.random.randn(1000) - 1.0  # Biased toward negative
    x = Tensor(random_data)
    result = relu(x)
    sparsity = np.sum(result.data == 0.0) / len(result.data)
    assert sparsity > 0.5, f"Expected high sparsity, got {sparsity:.2%}"
    
    print("✅ ReLU sparsity verified!")


def test_relu_call_vs_forward():
    """🧪 Test that __call__ and forward produce identical results."""
    print("🧪 Unit Test: ReLU __call__ vs forward...")
    
    relu = ReLU()
    x = Tensor([1.0, -2.0, 3.0, -4.0, 5.0])
    
    # Test forward method
    result_forward = relu.forward(x)
    
    # Test __call__ method
    result_call = relu(x)
    
    # Should produce identical results
    assert np.array_equal(result_forward.data, result_call.data)
    
    print("✅ ReLU __call__ and forward are consistent!")


def test_relu_hidden_layer():
    """🧪 Test ReLU in hidden layer scenario (common in neural networks)."""
    print("🧪 Unit Test: ReLU Hidden Layer...")
    
    relu = ReLU()
    
    # Simulate pre-activation values from a hidden layer
    # Some positive (active neurons), some negative (inactive neurons)
    pre_activation = Tensor([
        [2.5, -1.0, 0.5, -3.0],   # Sample 1: 2 active, 2 inactive
        [-0.5, 3.0, -2.0, 1.5],   # Sample 2: 2 active, 2 inactive
        [1.0, 1.0, 1.0, 1.0],     # Sample 3: all active
        [-1.0, -1.0, -1.0, -1.0]  # Sample 4: all inactive (dead)
    ])
    
    activations = relu(pre_activation)
    
    # Verify correct activation pattern
    expected = np.array([
        [2.5, 0.0, 0.5, 0.0],
        [0.0, 3.0, 0.0, 1.5],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    assert np.array_equal(activations.data, expected)
    
    # Check that at least some neurons are active (non-dead network)
    num_active = np.sum(activations.data > 0.0)
    assert num_active > 0, "Network should have some active neurons"
    
    print("✅ ReLU hidden layer scenario works!")


def test_relu_repr():
    """🧪 Test ReLU string representation."""
    print("🧪 Unit Test: ReLU __repr__...")
    
    relu = ReLU()
    repr_str = repr(relu)
    
    assert "ReLU" in repr_str, "Repr should contain 'ReLU'"
    assert repr_str == "ReLU()", f"Expected 'ReLU()', got '{repr_str}'"
    
    print("✅ ReLU __repr__ works!")


def test_relu_batch_processing():
    """🧪 Test ReLU with batch processing."""
    print("🧪 Unit Test: ReLU Batch Processing...")
    
    relu = ReLU()
    
    # Simulate a batch of samples (batch_size=4, features=3)
    batch = Tensor([
        [1.0, -2.0, 3.0],    # Sample 1
        [0.0, 0.0, 0.0],     # Sample 2 (zero)
        [-1.0, -2.0, -3.0],  # Sample 3 (all negative)
        [10.0, -10.0, 0.5]   # Sample 4 (mixed)
    ])
    
    result = relu(batch)
    
    # Verify shape is preserved
    assert result.shape == batch.shape
    
    # Verify all values are non-negative
    assert np.all(result.data >= 0.0)
    
    # Verify each sample processed independently
    for i in range(batch.shape[0]):
        sample_result = relu(Tensor(batch.data[i]))
        assert np.array_equal(result.data[i], sample_result.data, equal_nan=False)
    
    # Check expected output
    expected = np.array([
        [1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.5]
    ], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    
    print("✅ ReLU batch processing works!")


def test_relu_comparison_with_sigmoid():
    """🧪 Test conceptual differences between ReLU and Sigmoid."""
    print("🧪 Unit Test: ReLU vs Sigmoid Comparison...")
    
    relu = ReLU()
    sigmoid = Sigmoid()
    
    x = Tensor([10.0])
    
    # ReLU preserves magnitude
    relu_output = relu(x)
    assert relu_output.data[0] == 10.0, "ReLU preserves large positive values"
    
    # Sigmoid saturates
    sigmoid_output = sigmoid(x)
    assert sigmoid_output.data[0] <= 1.0, "Sigmoid saturates at 1"
    assert sigmoid_output.data[0] > 0.9, "Sigmoid close to 1 for large x"
    
    # ReLU has unbounded output for positive values
    large_x = Tensor([1000.0])
    relu_large = relu(large_x)
    assert relu_large.data[0] == 1000.0, "ReLU has no upper bound"
    
    print("✅ ReLU vs Sigmoid comparison complete!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 RUNNING ACTIVATION FUNCTION TESTS")
    print("=" * 60 + "\n")
    
    print("📋 SIGMOID TESTS")
    print("-" * 60)
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
    
    print("\n📋 RELU TESTS")
    print("-" * 60)
    test_relu_basic()
    test_relu_known_values()
    test_relu_shape_preservation()
    test_relu_output_range()
    test_relu_linearity_positive()
    test_relu_sparsity()
    test_relu_call_vs_forward()
    test_relu_hidden_layer()
    test_relu_repr()
    test_relu_batch_processing()
    test_relu_comparison_with_sigmoid()
    
    print("\n" + "=" * 60)
    print("🎉 ALL ACTIVATION TESTS PASSED!")
    print("=" * 60)