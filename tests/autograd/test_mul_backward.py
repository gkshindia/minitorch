"""🧪 Test Suite: MulBackward

Tests gradient computation for tensor multiplication.

Mathematical Rule: If z = a * b, then ∂z/∂a = b and ∂z/∂b = a
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import MulBackward, enable_autograd


def test_mul_backward_simple():
    """Test basic multiplication gradient computation."""
    print("\n🔬 Testing simple multiplication gradients...")
    
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    
    # Forward: c = a * b = [8.0, 15.0]
    c = a * b
    assert np.allclose(c.data, [8.0, 15.0])
    
    # Backward
    c.backward(np.ones_like(c.data))
    
    # ∂c/∂a = b, ∂c/∂b = a
    assert np.allclose(a.grad, [4.0, 5.0])  # gradient = b
    assert np.allclose(b.grad, [2.0, 3.0])  # gradient = a
    
    print("✅ Simple multiplication gradients correct!")


def test_mul_backward_scalar():
    """Test multiplication with scalar."""
    print("\n🔬 Testing multiplication with scalar...")
    
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    scalar = 5.0
    
    c = a * scalar
    assert np.allclose(c.data, [5.0, 10.0, 15.0])
    
    c.backward(np.ones_like(c.data))
    
    # ∂(a * 5)/∂a = 5
    assert np.allclose(a.grad, [5.0, 5.0, 5.0])
    
    print("✅ Scalar multiplication gradients correct!")


def test_mul_backward_matrix():
    """Test element-wise multiplication of matrices."""
    print("\n🔬 Testing matrix element-wise multiplication...")
    
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    
    c = a * b
    expected = [[0.5, 1.0], [1.5, 2.0]]
    assert np.allclose(c.data, expected)
    
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    # ∂c/∂a = b, ∂c/∂b = a
    assert np.allclose(a.grad, [[0.5, 0.5], [0.5, 0.5]])
    assert np.allclose(b.grad, [[1.0, 2.0], [3.0, 4.0]])
    
    print("✅ Matrix multiplication gradients correct!")


def test_mul_backward_chain():
    """Test multiplication in computation chain."""
    print("\n🔬 Testing multiplication in chain...")
    
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([4.0], requires_grad=True)
    
    # Chain: (a * b) * c = 2 * 3 * 4 = 24
    d = a * b  # 6
    e = d * c  # 24
    
    e.backward()
    
    # ∂e/∂a = b * c = 12
    # ∂e/∂b = a * c = 8
    # ∂e/∂c = a * b = 6
    assert np.allclose(a.grad, [12.0])
    assert np.allclose(b.grad, [8.0])
    assert np.allclose(c.grad, [6.0])
    
    print("✅ Chained multiplication gradients correct!")


def test_mul_backward_mixed():
    """Test multiplication mixed with addition."""
    print("\n🔬 Testing mixed operations (mul + add)...")
    
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([1.0], requires_grad=True)
    
    # y = a * b + c = 2 * 3 + 1 = 7
    d = a * b
    y = d + c
    
    y.backward()
    
    # ∂y/∂a = b = 3
    # ∂y/∂b = a = 2
    # ∂y/∂c = 1
    assert np.allclose(a.grad, [3.0])
    assert np.allclose(b.grad, [2.0])
    assert np.allclose(c.grad, [1.0])
    
    print("✅ Mixed operation gradients correct!")


def test_mul_backward_weighted():
    """Test multiplication with weighted gradient."""
    print("\n🔬 Testing weighted gradient through multiplication...")
    
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    
    c = a * b
    
    # Apply weighted gradient
    grad_output = np.array([2.0, 3.0])
    c.backward(grad_output)
    
    # ∂c/∂a = b * grad_output
    # ∂c/∂b = a * grad_output
    assert np.allclose(a.grad, [8.0, 15.0])  # [4*2, 5*3]
    assert np.allclose(b.grad, [4.0, 9.0])   # [2*2, 3*3]
    
    print("✅ Weighted gradients through multiplication correct!")


def test_module():
    """🧪 Module Test: MulBackward Complete Test

    Run all MulBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING MULBACKWARD MODULE TEST")
    print("="*60)
    
    test_mul_backward_simple()
    test_mul_backward_scalar()
    test_mul_backward_matrix()
    test_mul_backward_chain()
    test_mul_backward_mixed()
    test_mul_backward_weighted()
    
    print("\n" + "="*60)
    print("🎉 ALL MULBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
