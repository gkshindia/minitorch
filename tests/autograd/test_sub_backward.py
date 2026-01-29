"""🧪 Test Suite: SubBackward

Tests gradient computation for tensor subtraction.

Mathematical Rule: If z = a - b, then ∂z/∂a = 1 and ∂z/∂b = -1
"""

import numpy as np
from core.tensor import Tensor


def test_sub_backward_simple():
    """Test basic subtraction gradient computation."""
    print("\n🔬 Testing simple subtraction gradients...")
    
    a = Tensor([5.0, 7.0], requires_grad=True)
    b = Tensor([2.0, 3.0], requires_grad=True)
    
    # Forward: c = a - b = [3.0, 4.0]
    c = a - b
    assert np.allclose(c.data, [3.0, 4.0])
    
    # Backward
    c.backward(np.ones_like(c.data))
    
    # ∂c/∂a = 1, ∂c/∂b = -1
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [-1.0, -1.0])
    
    print("✅ Simple subtraction gradients correct!")


def test_sub_backward_scalar():
    """Test subtraction with scalar."""
    print("\n🔬 Testing subtraction with scalar...")
    
    a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    scalar = 5.0
    
    c = a - scalar
    assert np.allclose(c.data, [5.0, 15.0, 25.0])
    
    c.backward(np.ones_like(c.data))
    
    # ∂(a - 5)/∂a = 1
    assert np.allclose(a.grad, [1.0, 1.0, 1.0])
    
    print("✅ Scalar subtraction gradients correct!")


def test_sub_backward_matrix():
    """Test subtraction with matrices."""
    print("\n🔬 Testing matrix subtraction gradients...")
    
    a = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    b = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    c = a - b
    expected = [[4.0, 4.0], [4.0, 4.0]]
    assert np.allclose(c.data, expected)
    
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    # ∂c/∂a = 1, ∂c/∂b = -1
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(b.grad, [[-1.0, -1.0], [-1.0, -1.0]])
    
    print("✅ Matrix subtraction gradients correct!")


def test_sub_backward_chain():
    """Test subtraction in computation chain."""
    print("\n🔬 Testing subtraction in chain...")
    
    a = Tensor([10.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([2.0], requires_grad=True)
    
    # Chain: (a - b) - c = 10 - 3 - 2 = 5
    d = a - b  # 7
    e = d - c  # 5
    
    e.backward()
    
    # ∂e/∂a = 1
    # ∂e/∂b = -1
    # ∂e/∂c = -1
    assert np.allclose(a.grad, [1.0])
    assert np.allclose(b.grad, [-1.0])
    assert np.allclose(c.grad, [-1.0])
    
    print("✅ Chained subtraction gradients correct!")


def test_sub_backward_mixed():
    """Test subtraction mixed with multiplication."""
    print("\n🔬 Testing mixed operations (sub + mul)...")
    
    a = Tensor([5.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    c = Tensor([3.0], requires_grad=True)
    
    # y = (a - b) * c = (5 - 2) * 3 = 9
    d = a - b  # 3
    y = d * c  # 9
    
    y.backward()
    
    # ∂y/∂a = c = 3 (from subtraction gradient 1 * c)
    # ∂y/∂b = -c = -3 (from subtraction gradient -1 * c)
    # ∂y/∂c = (a - b) = 3
    assert np.allclose(a.grad, [3.0])
    assert np.allclose(b.grad, [-3.0])
    assert np.allclose(c.grad, [3.0])
    
    print("✅ Mixed operation gradients correct!")


def test_sub_backward_weighted():
    """Test subtraction with weighted gradient."""
    print("\n🔬 Testing weighted gradient through subtraction...")
    
    a = Tensor([8.0, 10.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    
    c = a - b
    
    # Apply weighted gradient
    grad_output = np.array([2.0, 3.0])
    c.backward(grad_output)
    
    # ∂c/∂a = 1 * grad_output
    # ∂c/∂b = -1 * grad_output
    assert np.allclose(a.grad, [2.0, 3.0])
    assert np.allclose(b.grad, [-2.0, -3.0])
    
    print("✅ Weighted gradients through subtraction correct!")


def test_module():
    """🧪 Module Test: SubBackward Complete Test

    Run all SubBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING SUBBACKWARD MODULE TEST")
    print("="*60)
    
    test_sub_backward_simple()
    test_sub_backward_scalar()
    test_sub_backward_matrix()
    test_sub_backward_chain()
    test_sub_backward_mixed()
    test_sub_backward_weighted()
    
    print("\n" + "="*60)
    print("🎉 ALL SUBBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
