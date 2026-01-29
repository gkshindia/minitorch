"""🧪 Test Suite: SumBackward

Tests gradient computation for tensor sum reduction.

Mathematical Rule: If z = sum(a), then ∂z/∂a[i] = 1 for all i
"""

import numpy as np
from core.tensor import Tensor


def test_sum_backward_simple():
    """Test basic sum gradient."""
    print("\n🔬 Testing simple sum gradients...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    # Forward: sum all elements
    y = x.sum()
    assert np.allclose(y.data, 10.0)
    
    # Backward
    y.backward()
    
    # All elements should have gradient of 1
    assert x.grad is not None
    assert np.allclose(x.grad, [1.0, 1.0, 1.0, 1.0])
    
    print("✅ Simple sum gradients correct!")


def test_sum_backward_matrix():
    """Test sum of matrix."""
    print("\n🔬 Testing matrix sum gradients...")
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    
    # Sum all elements
    y = x.sum()
    assert np.allclose(y.data, 21.0)
    
    y.backward()
    
    # All elements should have gradient of 1
    expected_grad = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Matrix sum gradients correct!")


def test_sum_backward_weighted():
    """Test sum with weighted upstream gradient."""
    print("\n🔬 Testing weighted sum gradients...")
    
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    
    y = x.sum()
    
    # Apply weighted gradient (e.g., from loss scaling)
    grad_output = 5.0
    y.backward(gradient=grad_output)
    
    # All elements should have gradient of 5.0
    assert np.allclose(x.grad, [5.0, 5.0, 5.0])
    
    print("✅ Weighted sum gradients correct!")


def test_sum_backward_chain():
    """Test sum in computation chain."""
    print("\n🔬 Testing sum in chain...")
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Chain: multiply then sum
    y = x * 2.0  # [2.0, 4.0, 6.0]
    z = y.sum()  # 12.0
    
    z.backward()
    
    # Gradient: ∂z/∂x = 2.0 (from multiplication) * 1.0 (from sum)
    assert np.allclose(x.grad, [2.0, 2.0, 2.0])
    
    print("✅ Chained sum gradients correct!")


def test_sum_backward_mean_simulation():
    """Test sum used to compute mean."""
    print("\n🔬 Testing mean via sum...")
    
    x = Tensor([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    n = len(x.data)
    
    # Mean = sum / n
    total = x.sum()
    mean = total / n
    
    mean.backward()
    
    # Gradient: ∂mean/∂x = 1/n for all elements
    expected_grad = [0.25, 0.25, 0.25, 0.25]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Mean via sum gradients correct!")


def test_sum_backward_loss():
    """Test sum in loss computation."""
    print("\n🔬 Testing sum in loss...")
    
    # Simulate squared error loss
    predictions = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    targets = Tensor([1.5, 2.5, 2.5])
    
    # Loss = sum((pred - target)^2)
    diff = predictions - targets  # [-0.5, -0.5, 0.5]
    squared = diff * diff  # [0.25, 0.25, 0.25]
    loss = squared.sum()  # 0.75
    
    loss.backward()
    
    # Gradient: ∂loss/∂pred = 2 * (pred - target)
    expected_grad = 2 * (predictions.data - targets.data)
    assert np.allclose(predictions.grad, expected_grad)
    
    print("✅ Sum in loss gradients correct!")


def test_sum_backward_batch():
    """Test sum over batch dimension."""
    print("\n🔬 Testing batch sum...")
    
    # Batch of vectors
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    
    # Sum all elements (across batch and features)
    y = x.sum()
    assert np.allclose(y.data, 21.0)
    
    y.backward()
    
    # All elements should have gradient of 1
    expected_grad = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Batch sum gradients correct!")


def test_module():
    """🧪 Module Test: SumBackward Complete Test

    Run all SumBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING SUMBACKWARD MODULE TEST")
    print("="*60)
    
    test_sum_backward_simple()
    test_sum_backward_matrix()
    test_sum_backward_weighted()
    test_sum_backward_chain()
    test_sum_backward_mean_simulation()
    test_sum_backward_loss()
    test_sum_backward_batch()
    
    print("\n" + "="*60)
    print("🎉 ALL SUMBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
