"""🧪 Test Suite: DivBackward

Tests gradient computation for tensor division.

Mathematical Rule: If z = a / b, then:
- ∂z/∂a = 1/b
- ∂z/∂b = -a/b²
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import DivBackward, enable_autograd


def test_div_backward_simple():
    """Test basic division gradient computation."""
    print("\n🔬 Testing simple division gradients...")
    
    a = Tensor([6.0, 8.0], requires_grad=True)
    b = Tensor([2.0, 4.0], requires_grad=True)
    
    # Forward: c = a / b = [3.0, 2.0]
    c = a / b
    assert np.allclose(c.data, [3.0, 2.0])
    
    # Backward
    c.backward(np.ones_like(c.data))
    
    # ∂c/∂a = 1/b = [0.5, 0.25]
    # ∂c/∂b = -a/b² = [-1.5, -0.5]
    assert np.allclose(a.grad, [0.5, 0.25])
    assert np.allclose(b.grad, [-1.5, -0.5])
    
    print("✅ Simple division gradients correct!")


def test_div_backward_scalar():
    """Test division by scalar."""
    print("\n🔬 Testing division by scalar...")
    
    a = Tensor([10.0, 20.0, 30.0], requires_grad=True)
    scalar = 5.0
    
    c = a / scalar
    assert np.allclose(c.data, [2.0, 4.0, 6.0])
    
    c.backward(np.ones_like(c.data))
    
    # ∂(a / 5)/∂a = 1/5 = 0.2
    assert np.allclose(a.grad, [0.2, 0.2, 0.2])
    
    print("✅ Scalar division gradients correct!")


def test_div_backward_matrix():
    """Test element-wise division of matrices."""
    print("\n🔬 Testing matrix element-wise division...")
    
    a = Tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    b = Tensor([[2.0, 4.0], [5.0, 8.0]], requires_grad=True)
    
    c = a / b
    expected = [[5.0, 5.0], [6.0, 5.0]]
    assert np.allclose(c.data, expected)
    
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    # ∂c/∂a = 1/b
    expected_grad_a = [[0.5, 0.25], [0.2, 0.125]]
    assert np.allclose(a.grad, expected_grad_a, rtol=1e-5)
    
    # ∂c/∂b = -a/b²
    expected_grad_b = [[-2.5, -1.25], [-1.2, -0.625]]
    assert np.allclose(b.grad, expected_grad_b, rtol=1e-5)
    
    print("✅ Matrix division gradients correct!")


def test_div_backward_chain():
    """Test division in computation chain."""
    print("\n🔬 Testing division in chain...")
    
    a = Tensor([12.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = Tensor([2.0], requires_grad=True)
    
    # Chain: (a / b) / c = 12 / 3 / 2 = 2
    d = a / b  # 4
    e = d / c  # 2
    
    e.backward()
    
    # ∂e/∂a = 1/(b*c) = 1/6 ≈ 0.1667
    # ∂e/∂b = -a/(b²*c) = -12/18 = -0.6667
    # ∂e/∂c = -a/(b*c²) = -12/12 = -1.0
    assert np.allclose(a.grad, [1.0/6.0], rtol=1e-4)
    assert np.allclose(b.grad, [-12.0/18.0], rtol=1e-4)
    assert np.allclose(c.grad, [-12.0/12.0], rtol=1e-4)
    
    print("✅ Chained division gradients correct!")


def test_div_backward_mixed():
    """Test division mixed with multiplication."""
    print("\n🔬 Testing mixed operations (div + mul)...")
    
    a = Tensor([10.0], requires_grad=True)
    b = Tensor([2.0], requires_grad=True)
    c = Tensor([3.0], requires_grad=True)
    
    # y = (a / b) * c = (10 / 2) * 3 = 15
    d = a / b  # 5
    y = d * c  # 15
    
    y.backward()
    
    # ∂y/∂a = c/b = 3/2 = 1.5
    # ∂y/∂b = -a*c/b² = -30/4 = -7.5
    # ∂y/∂c = a/b = 10/2 = 5
    assert np.allclose(a.grad, [1.5])
    assert np.allclose(b.grad, [-7.5])
    assert np.allclose(c.grad, [5.0])
    
    print("✅ Mixed operation gradients correct!")


def test_div_backward_weighted():
    """Test division with weighted gradient."""
    print("\n🔬 Testing weighted gradient through division...")
    
    a = Tensor([10.0, 20.0], requires_grad=True)
    b = Tensor([2.0, 5.0], requires_grad=True)
    
    c = a / b
    
    # Apply weighted gradient
    grad_output = np.array([2.0, 3.0])
    c.backward(grad_output)
    
    # ∂c/∂a = grad_output/b
    assert np.allclose(a.grad, [1.0, 0.6])  # [2/2, 3/5]
    
    # ∂c/∂b = -grad_output * a/b²
    assert np.allclose(b.grad, [-5.0, -2.4])  # [-2*10/4, -3*20/25]
    
    print("✅ Weighted gradients through division correct!")


def test_module():
    """🧪 Module Test: DivBackward Complete Test

    Run all DivBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING DIVBACKWARD MODULE TEST")
    print("="*60)
    
    test_div_backward_simple()
    test_div_backward_scalar()
    test_div_backward_matrix()
    test_div_backward_chain()
    test_div_backward_mixed()
    test_div_backward_weighted()
    
    print("\n" + "="*60)
    print("🎉 ALL DIVBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
