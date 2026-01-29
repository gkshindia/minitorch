"""🧪 Test Suite: AddBackward

Tests gradient computation for tensor addition.

Mathematical Rule: If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1
"""

import numpy as np
from core.tensor import Tensor


def test_add_backward_simple():
    """Test basic addition gradient computation."""
    print("\n🔬 Testing simple addition gradients...")
    
    # Create tensors
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    
    # Forward pass
    c = a + b
    assert np.allclose(c.data, [6.0, 8.0])
    
    # Backward pass
    c.backward(np.ones_like(c.data))
    
    # Check gradients: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])
    
    print("✅ Simple addition gradients correct!")


def test_add_backward_scalar():
    """Test addition with scalar."""
    print("\n🔬 Testing addition with scalar...")
    
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = 5.0
    
    c = a + b
    assert np.allclose(c.data, [6.0, 7.0, 8.0])
    
    c.backward(np.ones_like(c.data))
    
    # Only 'a' should have gradients (scalar doesn't)
    assert a.grad is not None
    assert np.allclose(a.grad, [1.0, 1.0, 1.0])
    
    print("✅ Scalar addition gradients correct!")


def test_add_backward_matrix():
    """Test addition with matrices."""
    print("\n🔬 Testing matrix addition gradients...")
    
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    
    c = a + b
    expected = [[1.1, 2.2], [3.3, 4.4]]
    assert np.allclose(c.data, expected)
    
    # Backward with gradient
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    # Gradients should be 1s everywhere
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(b.grad, [[1.0, 1.0], [1.0, 1.0]])
    
    print("✅ Matrix addition gradients correct!")


def test_add_backward_chain():
    """Test addition in computation chain."""
    print("\n🔬 Testing addition in chain...")
    
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([4.0], requires_grad=True)
    
    # Chain: (a + b) + c
    d = a + b  # d = 5
    e = d + c  # e = 9
    
    e.backward()
    
    # All gradients should be 1 (addition distributes equally)
    assert np.allclose(a.grad, [1.0])
    assert np.allclose(b.grad, [1.0])
    assert np.allclose(c.grad, [1.0])
    
    print("✅ Chained addition gradients correct!")


def test_add_backward_no_grad():
    """Test that gradients don't compute when requires_grad=False."""
    print("\n🔬 Testing addition with requires_grad=False...")
    
    a = Tensor([1.0, 2.0], requires_grad=False)
    b = Tensor([3.0, 4.0], requires_grad=True)
    
    c = a + b
    c.backward(np.ones_like(c.data))
    
    # Only b should have gradients
    assert a.grad is None or np.allclose(a.grad, [0.0, 0.0])
    assert b.grad is not None
    assert np.allclose(b.grad, [1.0, 1.0])
    
    print("✅ Selective gradient computation works!")


def test_add_backward_weighted():
    """Test addition with weighted gradient."""
    print("\n🔬 Testing weighted gradient through addition...")
    
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    
    c = a + b
    
    # Apply weighted gradient
    grad_output = np.array([2.0, 3.0])
    c.backward(grad_output)
    
    # Gradients should match grad_output (∂(a+b)/∂a = 1)
    assert np.allclose(a.grad, [2.0, 3.0])
    assert np.allclose(b.grad, [2.0, 3.0])
    
    print("✅ Weighted gradients through addition correct!")


def test_module():
    """🧪 Module Test: AddBackward Complete Test

    Run all AddBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING ADDBACKWARD MODULE TEST")
    print("="*60)
    
    test_add_backward_simple()
    test_add_backward_scalar()
    test_add_backward_matrix()
    test_add_backward_chain()
    test_add_backward_no_grad()
    test_add_backward_weighted()
    
    print("\n" + "="*60)
    print("🎉 ALL ADDBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
