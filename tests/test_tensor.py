import numpy as np
import pytest
from core.tensor import Tensor

def test_initialization_and_properties():
    """🧪 Test tensor creation and core properties."""
    print("\n🧪 Unit Test: Initialization & Properties...")
    
    # Test 1D tensor (like a feature vector)
    data = [1, 2, 3]
    t = Tensor(data)
    assert np.array_equal(t.data, np.array(data, dtype=np.float32))
    assert t.shape == (3,)
    assert t.size == 3
    assert t.dtype == np.float32
    
    # Test 2D tensor (like a batch of samples)
    t2 = Tensor([[1, 2], [3, 4]])
    assert t2.shape == (2, 2)
    assert t2.size == 4
    
    # Test 3D tensor (like image batch: batch_size × height × width)
    t3 = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert t3.shape == (2, 2, 2)
    assert t3.size == 8
    
    # Test scalar tensor
    scalar = Tensor(5)
    assert scalar.size == 1
    
    # Test numpy() method returns underlying array
    numpy_arr = t.numpy()
    assert isinstance(numpy_arr, np.ndarray)
    assert np.array_equal(numpy_arr, t.data)
    
    # Test memory footprint calculation
    # float32 = 4 bytes, 3 elements = 12 bytes
    assert t.memory_footprint() == 12
    assert t2.memory_footprint() == 16
    
    # Test string representations
    assert "Tensor" in str(t)
    assert "shape=" in repr(t)
    assert str(t.shape) in repr(t)
    
    print("✅ Initialization tests passed!")

def test_arithmetic_operations():
    """🧪 Test arithmetic operations with broadcasting."""
    print("🧪 Unit Test: Arithmetic Operations...")

    # Test 1: Element-wise addition (tensor + tensor)
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = a + b
    assert np.array_equal(result.data, np.array([5, 7, 9], dtype=np.float32))

    # Test 2: Scalar addition (bias term in neural networks)
    result = a + 10
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))

    # Test 3: Broadcasting - matrix + vector (adding bias to each row)
    matrix = Tensor([[1, 2], [3, 4]])
    vector = Tensor([10, 20])
    result = matrix + vector
    # Each row of matrix gets vector added: [[1+10, 2+20], [3+10, 4+20]]
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test 4: Subtraction (data centering - removing mean)
    result = b - a
    assert np.array_equal(result.data, np.array([3, 3, 3], dtype=np.float32))
    
    # Test 5: Scalar subtraction (centering around a value)
    result = a - 2
    assert np.array_equal(result.data, np.array([-1, 0, 1], dtype=np.float32))

    # Test 6: Element-wise multiplication (masking in attention)
    result = a * b
    # [1*4, 2*5, 3*6] = [4, 10, 18]
    assert np.array_equal(result.data, np.array([4, 10, 18], dtype=np.float32))

    # Test 7: Scalar multiplication (scaling/learning rate application)
    result = a * 2
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    # Test 8: Element-wise division
    result = b / a
    # [4/1, 5/2, 6/3] = [4, 2.5, 2]
    assert np.array_equal(result.data, np.array([4.0, 2.5, 2.0], dtype=np.float32))

    # Test 9: Scalar division (normalization)
    result = b / 2
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))

    # Test 10: Chained operations (standardization: z = (x - μ) / σ)
    data = Tensor([1, 2, 3, 4, 5])
    mean_val = 3.0  # mean of [1,2,3,4,5]
    std_val = 1.5  # approximate std
    standardized = (data - mean_val) / std_val
    # [-2/1.5, -1/1.5, 0, 1/1.5, 2/1.5]
    expected = np.array([-1.333, -0.667, 0.0, 0.667, 1.333], dtype=np.float32)
    assert np.allclose(standardized.data, expected, atol=0.01)
    
    # Test 11: Complex broadcasting (3D tensor with 1D)
    tensor_3d = Tensor(np.ones((2, 3, 4)))
    bias = Tensor([1, 2, 3, 4])
    result = tensor_3d + bias
    assert result.shape == (2, 3, 4)
    # Check one slice: each [i,j,:] should be [2, 3, 4, 5]
    assert np.array_equal(result.data[0, 0, :], np.array([2, 3, 4, 5], dtype=np.float32))

    print("✅ Arithmetic operations work correctly!")

def test_matrix_operations():
    """🧪 Test matrix multiplication operations."""
    print("🧪 Unit Test: Matrix Multiplication...")

    # Test 1: Basic 2×2 matrix multiplication
    a = Tensor([[1, 2], [3, 4]])  # 2×2
    b = Tensor([[5, 6], [7, 8]])  # 2×2
    result = a.matmul(b)
    # Expected: [[1×5+2×7, 1×6+2×8], [3×5+4×7, 3×6+4×8]] = [[19, 22], [43, 50]]
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test 2: Rectangular matrices (common in neural networks: input→hidden layer)
    c = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3 (batch_size=2, features=3)
    d = Tensor([[7, 8], [9, 10], [11, 12]])  # 3×2 (features=3, outputs=2)
    result = c.matmul(d)
    # Expected: [[1×7+2×9+3×11, 1×8+2×10+3×12], [4×7+5×9+6×11, 4×8+5×10+6×12]]
    # = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]] = [[58, 64], [139, 154]]
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test 3: Matrix-vector multiplication (forward pass)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3
    vector = Tensor([[1], [2], [3]])  # 3×1
    result = matrix.matmul(vector)
    # Expected: [[1×1+2×2+3×3], [4×1+5×2+6×3]] = [[14], [32]]
    expected = np.array([[14], [32]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test 4: Using @ operator (syntactic sugar)
    A = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    B = Tensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    C = A @ B  # testing __matmul__ alias
    expected = np.array([[22, 28], [49, 64]], dtype=np.float32)
    assert np.allclose(C.data, expected)
    
    # Test 5: Batched matmul (3D - batch processing)
    # Like processing multiple samples at once
    batched_A = Tensor(np.ones((2, 2, 3)))  # 2 batches of 2×3 matrices
    batched_B = Tensor(np.ones((2, 3, 2)))  # 2 batches of 3×2 matrices
    batched_C = batched_A.matmul(batched_B)
    assert batched_C.shape == (2, 2, 2)
    # Each result is 2×2 with all elements = 3 (sum of 3 ones)
    assert np.allclose(batched_C.data[0], np.ones((2, 2), dtype=np.float32) * 3)
    
    # Test 6: Identity matrix property (A @ I = A)
    identity = Tensor([[1, 0], [0, 1]])
    test_mat = Tensor([[2, 3], [4, 5]])
    result = test_mat @ identity
    assert np.array_equal(result.data, test_mat.data)

    # Test 7: Shape validation - should raise clear error
    try:
        incompatible_a = Tensor([[1, 2]])  # 1×2
        incompatible_b = Tensor([[1], [2], [3]])  # 3×1
        incompatible_a.matmul(incompatible_b)  # 1×2 @ 3×1 should fail (2 ≠ 3)
        assert False, "Should have raised ValueError for incompatible shapes"
    except ValueError as e:
        assert "Inner dimensions must match" in str(e)
        assert "2 ≠ 3" in str(e)  # Should show specific dimensions
    
    # Test 8: Type validation
    try:
        Tensor([[1, 2]]).matmul(5)  # Can't multiply matrix with scalar
        assert False, "Should have raised TypeError"
    except TypeError as e:
        assert "Expected Tensor" in str(e)

    print("✅ Matrix multiplication tests passed!")

def test_shape_manipulation():
    """🧪 Test reshaping and transposing (vital for DL architectures)."""
    print("🧪 Unit Test: Shape Manipulation...")

    # Test 1: Basic reshape (flatten to matrix)
    t = Tensor(np.arange(12))  # [0, 1, 2, ..., 11]
    reshaped = t.reshape(3, 4)
    assert reshaped.shape == (3, 4)
    # Verify data integrity
    assert reshaped.data[0, 0] == 0
    assert reshaped.data[2, 3] == 11
    
    # Test 2: Auto-reshape with -1 (inference)
    auto = t.reshape(2, -1)  # Should infer 6: 12/2 = 6
    assert auto.shape == (2, 6)
    
    auto3d = t.reshape(2, -1, 2)  # Should infer 3: 12/(2*2) = 3
    assert auto3d.shape == (2, 3, 2)
    
    # Test 3: Reshape with tuple argument
    reshaped_tuple = t.reshape((4, 3))
    assert reshaped_tuple.shape == (4, 3)
    
    # Test 4: Reshape to different dimensions (3D to 2D)
    t3d = Tensor(np.arange(24).reshape((2, 3, 4)))
    flattened = t3d.reshape(6, 4)
    assert flattened.shape == (6, 4)
    
    # Test 5: Reshape error - incompatible size
    try:
        t.reshape(5, 5)  # 25 ≠ 12
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Total elements must match" in str(e)
    
    # Test 6: Multiple -1 error
    try:
        t.reshape(-1, -1)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Can only specify one unknown dimension" in str(e)
    
    # Test 7: Default transpose (swap last two dims)
    # Important for matrix operations: (batch, seq_len, features) → (batch, features, seq_len)
    t_2d = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    t_T = t_2d.transpose()
    assert t_T.shape == (3, 2)
    # Verify: original [0, 1] (value 2) should be at [1, 0] after transpose
    assert t_T.data[1, 0] == 2
    
    # Test 8: 3D transpose default (for attention mechanisms)
    t3d = Tensor(np.arange(24).reshape((2, 3, 4)))
    t3d_T = t3d.transpose()
    assert t3d_T.shape == (2, 4, 3)  # Last two swapped
    
    # Test 9: Specific dimension transpose (NCHW to NHWC conversion)
    # N=batch, C=channels, H=height, W=width
    nchw = Tensor(np.arange(24).reshape((2, 3, 4, 1)))  # Simplified example
    # Swap dims 1 and 2 (C and H)
    permuted = nchw.transpose(1, 2)
    assert permuted.shape == (2, 4, 3, 1)
    
    # Test 10: 1D transpose (should return copy)
    t1d = Tensor([1, 2, 3])
    t1d_T = t1d.transpose()
    assert t1d_T.shape == (3,)
    assert np.array_equal(t1d.data, t1d_T.data)

    print("✅ Shape manipulation tests passed!")

def test_reductions_and_stats():
    """🧪 Test statistical reductions (essential for loss calculations)."""
    print("🧪 Unit Test: Reductions & Stats...")
    
    t = Tensor([[1, 2, 3], [4, 5, 6]]) # (2, 3)
    
    # Test 1: Global sum (total loss calculation)
    total_sum = t.sum()
    assert total_sum.data == 21  # 1+2+3+4+5+6
    assert total_sum.shape == ()  # Scalar
    
    # Test 2: Sum over axis 0 (column-wise - aggregate across batch)
    col_sum = t.sum(axis=0)
    assert col_sum.shape == (3,)
    # [1+4, 2+5, 3+6] = [5, 7, 9]
    assert np.array_equal(col_sum.data, np.array([5, 7, 9], dtype=np.float32))
    
    # Test 3: Sum over axis 1 (row-wise - aggregate across features)
    row_sum = t.sum(axis=1)
    assert row_sum.shape == (2,)
    # [1+2+3, 4+5+6] = [6, 15]
    assert np.array_equal(row_sum.data, np.array([6, 15], dtype=np.float32))
    
    # Test 4: Sum with keepdims (broadcasting compatibility)
    col_sum_keep = t.sum(axis=0, keepdims=True)
    assert col_sum_keep.shape == (1, 3)  # Maintains 2D structure
    assert np.array_equal(col_sum_keep.data, np.array([[5, 7, 9]], dtype=np.float32))
    
    row_sum_keep = t.sum(axis=1, keepdims=True)
    assert row_sum_keep.shape == (2, 1)
    assert np.array_equal(row_sum_keep.data, np.array([[6], [15]], dtype=np.float32))
    
    # Test 5: Mean (average - for normalization)
    avg = t.mean()
    assert avg.data == 3.5  # 21/6
    
    col_mean = t.mean(axis=0)
    # [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
    assert np.allclose(col_mean.data, np.array([2.5, 3.5, 4.5], dtype=np.float32))
    
    # Test 6: Mean with keepdims
    row_mean_keep = t.mean(axis=1, keepdims=True)
    assert row_mean_keep.shape == (2, 1)
    # [(1+2+3)/3, (4+5+6)/3] = [2, 5]
    assert np.array_equal(row_mean_keep.data, np.array([[2], [5]], dtype=np.float32))
    
    # Test 7: Max (max pooling)
    global_max = t.max()
    assert global_max.data == 6
    
    row_max = t.max(axis=1)
    assert row_max.shape == (2,)
    # [max(1,2,3), max(4,5,6)] = [3, 6]
    assert np.array_equal(row_max.data, np.array([3, 6], dtype=np.float32))
    
    # Test 8: Max with keepdims (important for broadcasting)
    row_max_keep = t.max(axis=1, keepdims=True)
    assert row_max_keep.shape == (2, 1)
    assert np.array_equal(row_max_keep.data, np.array([[3], [6]], dtype=np.float32))
    
    # Test 9: 3D reductions (batch processing)
    t3d = Tensor(np.arange(24).reshape((2, 3, 4)))
    
    # Sum over batch dimension (axis 0)
    batch_sum = t3d.sum(axis=0)
    assert batch_sum.shape == (3, 4)
    
    # Mean over last dimension (feature-wise)
    feature_mean = t3d.mean(axis=2, keepdims=True)
    assert feature_mean.shape == (2, 3, 1)

    print("✅ Reduction tests passed!")

def test_slicing_and_indexing():
    """🧪 Test Pythonic indexing behavior (data access patterns)."""
    print("🧪 Unit Test: Indexing & Slicing...")
    
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    # Test 1: Row access (getting a sample from batch)
    row0 = t[0]
    assert row0.shape == (3,)
    assert np.array_equal(row0.data, np.array([1, 2, 3], dtype=np.float32))
    
    row1 = t[1]
    assert np.array_equal(row1.data, np.array([4, 5, 6], dtype=np.float32))
    
    # Test 2: Single element access
    val = t[1, 2]  # Row 1, Col 2 = 6
    assert val.data == 6
    
    val_00 = t[0, 0]  # Top-left = 1
    assert val_00.data == 1
    
    # Test 3: Column access (slicing)
    col0 = t[:, 0]  # All rows, first column
    assert col0.shape == (2,)
    assert np.array_equal(col0.data, np.array([1, 4], dtype=np.float32))
    
    # Test 4: Range slicing (subset of data)
    sub = t[0:2, 1:3]  # Rows 0-1, Cols 1-2
    assert sub.shape == (2, 2)
    assert np.array_equal(sub.data, np.array([[2, 3], [5, 6]], dtype=np.float32))
    
    # Test 5: 1D tensor indexing
    t1d = Tensor([10, 20, 30, 40, 50])
    assert t1d[2].data == 30
    
    # Test 6: Negative indexing
    assert t1d[-1].data == 50  # Last element
    assert t1d[-2].data == 40  # Second to last
    
    # Test 7: 3D indexing (batch, height, width)
    t3d = Tensor(np.arange(24).reshape((2, 3, 4)))
    # Get first batch
    batch0 = t3d[0]
    assert batch0.shape == (3, 4)
    
    # Get specific element
    element = t3d[1, 2, 3]  # Last element of second batch
    assert element.data == 23

    print("✅ Indexing & Slicing tests passed!")

def test_broadcasting_advanced():
    """🧪 Test advanced broadcasting scenarios (critical for neural networks)."""
    print("🧪 Unit Test: Advanced Broadcasting...")
    
    # Test 1: Scalar broadcasting
    t = Tensor([[1, 2], [3, 4]])
    result = t * 2
    assert np.array_equal(result.data, np.array([[2, 4], [6, 8]], dtype=np.float32))
    
    # Test 2: Vector to matrix broadcasting (add bias to each row)
    matrix = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # (3, 3)
    bias = Tensor([10, 20, 30])  # (3,)
    result = matrix + bias
    # Each row gets [10, 20, 30] added
    expected = np.array([[11, 22, 33], [14, 25, 36], [17, 28, 39]], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    
    # Test 3: Column vector broadcasting
    col_vec = Tensor([[10], [20], [30]])  # (3, 1)
    result = matrix + col_vec
    # Each row gets its corresponding scalar added
    expected = np.array([[11, 12, 13], [24, 25, 26], [37, 38, 39]], dtype=np.float32)
    assert np.array_equal(result.data, expected)
    
    # Test 4: 3D broadcasting (batch normalization-like)
    batch = Tensor(np.ones((2, 3, 4)))  # (batch, height, width)
    channel_bias = Tensor([1, 2, 3])  # (channels,)
    # Reshape for broadcasting: (1, 3, 1)
    channel_bias_reshaped = channel_bias.reshape(1, 3, 1)
    result = batch + channel_bias_reshaped
    assert result.shape == (2, 3, 4)
    # First channel should have +1, second +2, third +3
    assert np.all(result.data[:, 0, :] == 2)
    assert np.all(result.data[:, 1, :] == 3)
    assert np.all(result.data[:, 2, :] == 4)

    print("✅ Advanced broadcasting tests passed!")

def test_ml_use_cases():
    """🧪 Test realistic ML pipeline scenarios."""
    print("🧪 Unit Test: ML Use Cases...")
    
    # Test 1: Softmax preparation (subtract max for numerical stability)
    logits = Tensor([[2, 1, 0.1], [1, 3, 0.2]])
    max_vals = logits.max(axis=1, keepdims=True)
    # Subtract max from each row
    stable_logits = logits - max_vals
    # Each row should have max value of 0
    assert np.allclose(stable_logits.max(axis=1).data, np.zeros(2, dtype=np.float32))
    
    # Test 2: Layer normalization pattern
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # (batch=2, features=3)
    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    # Mean of centered should be ~0
    assert np.allclose(centered.mean(axis=1).data, np.zeros(2, dtype=np.float32), atol=1e-6)
    
    # Test 3: Attention score calculation pattern
    Q = Tensor([[1, 2], [3, 4]])  # Query (2, 2)
    K = Tensor([[5, 6], [7, 8]])  # Key (2, 2)
    # scores = Q @ K^T
    K_T = K.transpose()
    scores = Q @ K_T
    assert scores.shape == (2, 2)
    
    # Test 4: Gradient clipping simulation
    gradients = Tensor([0.5, -1.5, 2.0, -0.3])
    clip_value = 1.0
    # This would need max/min operations, but we can test the concept
    # For now, just verify we can do element-wise comparisons
    assert gradients.max().data == 2.0
    
    print("✅ ML use case tests passed!")

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_initialization_and_properties()
        test_arithmetic_operations()
        test_matrix_operations()
        test_shape_manipulation()
        test_reductions_and_stats()
        test_slicing_and_indexing()
        test_broadcasting_advanced()
        test_ml_use_cases()
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
