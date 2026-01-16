"""🧪 Test Suite: GELUBackward

Tests gradient computation for GELU activation.

GELU: f(x) = x * Φ(x) where Φ is CDF of standard normal
Approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import GELUBackward, enable_autograd


def gelu(x):
    """Helper: compute GELU approximation."""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    x_cubed = x ** 3
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1 + np.tanh(tanh_arg))


def test_gelu_backward_simple():
    """Test basic GELU gradient."""
    print("\n🔬 Testing simple GELU gradients...")
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Forward: GELU
    gelu_data = gelu(x.data)
    y = Tensor(gelu_data)
    y.requires_grad = True
    y._grad_fn = GELUBackward(x)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Verify gradient computed
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # GELU gradient should be non-zero for all inputs (unlike ReLU)
    assert not np.allclose(x.grad, 0.0)
    
    print("✅ Simple GELU gradients correct!")


def test_gelu_backward_zero():
    """Test GELU at zero."""
    print("\n🔬 Testing GELU at zero...")
    
    x = Tensor([0.0], requires_grad=True)
    
    gelu_data = gelu(x.data)
    y = Tensor(gelu_data)
    y.requires_grad = True
    y._grad_fn = GELUBackward(x)
    
    # GELU(0) = 0
    assert np.allclose(y.data, [0.0], atol=1e-6)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # GELU'(0) ≈ 0.5 (derivative at zero is approximately 0.5)
    assert x.grad[0] > 0.4 and x.grad[0] < 0.6
    
    print("✅ GELU at zero gradient correct!")


def test_gelu_backward_positive():
    """Test GELU on positive inputs."""
    print("\n🔬 Testing GELU on positive inputs...")
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    gelu_data = gelu(x.data)
    y = Tensor(gelu_data)
    y.requires_grad = True
    y._grad_fn = GELUBackward(x)
    
    # For large positive x, GELU ≈ x (gradient ≈ 1)
    assert y.data[2] > 2.9  # GELU(3) ≈ 3
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient for large positive should be close to 1
    assert x.grad[2] > 0.9
    
    print("✅ GELU on positive inputs gradients correct!")


def test_gelu_backward_negative():
    """Test GELU on negative inputs."""
    print("\n🔬 Testing GELU on negative inputs...")
    
    x = Tensor([-3.0, -2.0, -1.0], requires_grad=True)
    
    gelu_data = gelu(x.data)
    y = Tensor(gelu_data)
    y.requires_grad = True
    y._grad_fn = GELUBackward(x)
    
    # For large negative x, GELU ≈ 0 (but gradients non-zero, unlike ReLU)
    assert y.data[0] < 0.01  # GELU(-3) ≈ 0
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Unlike ReLU, GELU has non-zero gradients for negatives
    assert not np.allclose(x.grad, 0.0)
    
    print("✅ GELU on negative inputs gradients correct!")


def test_gelu_backward_matrix():
    """Test GELU on matrix."""
    print("\n🔬 Testing matrix GELU...")
    
    x = Tensor([[-1.0, 0.0], [1.0, 2.0]], requires_grad=True)
    
    gelu_data = gelu(x.data)
    y = Tensor(gelu_data)
    y.requires_grad = True
    y._grad_fn = GELUBackward(x)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # All gradients should be computed
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    print("✅ Matrix GELU gradients correct!")


def test_gelu_backward_vs_relu():
    """Test GELU compared to ReLU behavior."""
    print("\n🔬 Testing GELU vs ReLU behavior...")
    
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    
    gelu_data = gelu(x.data)
    y = Tensor(gelu_data)
    y.requires_grad = True
    y._grad_fn = GELUBackward(x)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Key difference: GELU has non-zero gradients for negative inputs
    assert x.grad[0] != 0.0  # Unlike ReLU which would be 0
    
    # Both should have positive gradient for positive input
    assert x.grad[2] > 0.0
    
    print("✅ GELU vs ReLU behavior correct!")


def test_gelu_backward_chain():
    """Test GELU in computation chain."""
    print("\n🔬 Testing GELU in chain...")
    
    x = Tensor([1.0, 2.0], requires_grad=True)
    
    # Chain: multiply -> GELU
    y = x * 2.0  # [2, 4]
    gelu_data = gelu(y.data)
    z = Tensor(gelu_data)
    z.requires_grad = True
    z._grad_fn = GELUBackward(y)
    
    z.backward(np.ones_like(z.data))
    
    # Gradient flows through GELU then multiply
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    print("✅ Chained GELU gradients correct!")


def test_module():
    """🧪 Module Test: GELUBackward Complete Test

    Run all GELUBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING GELUBACKWARD MODULE TEST")
    print("="*60)
    
    test_gelu_backward_simple()
    test_gelu_backward_zero()
    test_gelu_backward_positive()
    test_gelu_backward_negative()
    test_gelu_backward_matrix()
    test_gelu_backward_vs_relu()
    test_gelu_backward_chain()
    
    print("\n" + "="*60)
    print("🎉 ALL GELUBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
