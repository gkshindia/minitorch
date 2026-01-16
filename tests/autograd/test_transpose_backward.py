"""🧪 Test Suite: TransposeBackward

Tests gradient computation for transpose operation.

Mathematical Rule: If Y = X.T, then ∂Y/∂X = grad_Y.T
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import TransposeBackward, enable_autograd


def test_transpose_backward_simple():
    """Test basic transpose gradient."""
    print("\n🔬 Testing simple transpose gradients...")
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    
    # Forward: transpose
    y = x.transpose()
    assert y.shape == (3, 2)
    expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert np.allclose(y.data, expected)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should be transposed back
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    
    print("✅ Simple transpose gradients correct!")


def test_transpose_backward_square():
    """Test transpose of square matrix."""
    print("\n🔬 Testing square matrix transpose...")
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = x.transpose()
    
    assert y.shape == (2, 2)
    expected = [[1.0, 3.0], [2.0, 4.0]]
    assert np.allclose(y.data, expected)
    
    grad_output = np.array([[2.0, 3.0], [4.0, 5.0]])
    y.backward(grad_output)
    
    # Gradient should be transposed back
    expected_grad = [[2.0, 4.0], [3.0, 5.0]]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Square matrix transpose gradients correct!")


def test_transpose_backward_specific_dims():
    """Test transpose with specific dimensions."""
    print("\n🔬 Testing transpose with specific dims...")
    
    # 3D tensor: (2, 3, 4)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    
    # Transpose last two dimensions
    y = x.transpose(dim0=1, dim1=2)
    assert y.shape == (2, 4, 3)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should have original shape
    assert x.grad.shape == x.shape
    
    print("✅ Specific dimension transpose gradients correct!")


def test_transpose_backward_chain():
    """Test transpose in computation chain."""
    print("\n🔬 Testing transpose in chain...")
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    W = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    
    # Chain: X.T @ W
    x_t = x.transpose()
    y = x_t.matmul(W)
    
    y.backward(np.ones_like(y.data))
    
    assert x.grad is not None
    assert W.grad is not None
    assert x.grad.shape == x.shape
    assert W.grad.shape == W.shape
    
    print("✅ Chained transpose gradients correct!")


def test_transpose_backward_double():
    """Test double transpose (should return to original)."""
    print("\n🔬 Testing double transpose...")
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    
    # Double transpose
    y = x.transpose().transpose()
    assert y.shape == x.shape
    assert np.allclose(y.data, x.data)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should pass through unchanged
    assert np.allclose(x.grad, grad_output)
    
    print("✅ Double transpose gradients correct!")


def test_transpose_backward_3d():
    """Test transpose on 3D tensor."""
    print("\n🔬 Testing 3D tensor transpose...")
    
    # Batch of matrices: (batch=2, rows=3, cols=4)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    
    # Transpose last two dims (common in attention)
    y = x.transpose()  # Default transposes last two
    assert y.shape == (2, 4, 3)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    assert x.grad.shape == x.shape
    
    print("✅ 3D tensor transpose gradients correct!")


def test_module():
    """🧪 Module Test: TransposeBackward Complete Test

    Run all TransposeBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING TRANSPOSEBACKWARD MODULE TEST")
    print("="*60)
    
    test_transpose_backward_simple()
    test_transpose_backward_square()
    test_transpose_backward_specific_dims()
    test_transpose_backward_chain()
    test_transpose_backward_double()
    test_transpose_backward_3d()
    
    print("\n" + "="*60)
    print("🎉 ALL TRANSPOSEBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
