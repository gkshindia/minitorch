"""🧪 Test Suite: ReshapeBackward

Tests gradient computation for reshape operation.

Mathematical Rule: If Y = X.reshape(new_shape), then:
∂Y/∂X = grad_Y.reshape(X.shape)
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import ReshapeBackward, enable_autograd


def test_reshape_backward_simple():
    """Test basic reshape gradient."""
    print("\n🔬 Testing simple reshape gradients...")
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    original_shape = x.shape
    
    # Reshape from (2, 3) to (3, 2)
    y = x.reshape(3, 2)
    assert y.shape == (3, 2)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should be reshaped back to original
    assert x.grad is not None
    assert x.grad.shape == original_shape
    assert np.allclose(x.grad, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    
    print("✅ Simple reshape gradients correct!")


def test_reshape_backward_flatten():
    """Test flattening (reshape to 1D)."""
    print("\n🔬 Testing flatten reshape...")
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    original_shape = x.shape
    
    # Flatten to (6,)
    y = x.reshape(-1)
    assert y.shape == (6,)
    assert np.allclose(y.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    grad_output = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y.backward(grad_output)
    
    # Gradient should be reshaped back to (3, 2)
    expected_grad = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Flatten reshape gradients correct!")


def test_reshape_backward_expand():
    """Test expanding dimensions."""
    print("\n🔬 Testing expand dimensions reshape...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    # Reshape from (4,) to (2, 2)
    y = x.reshape(2, 2)
    assert y.shape == (2, 2)
    
    grad_output = np.array([[1.0, 2.0], [3.0, 4.0]])
    y.backward(grad_output)
    
    # Gradient should be reshaped back to (4,)
    assert np.allclose(x.grad, [1.0, 2.0, 3.0, 4.0])
    
    print("✅ Expand dimensions reshape gradients correct!")


def test_reshape_backward_batch():
    """Test reshaping batch data."""
    print("\n🔬 Testing batch reshape...")
    
    # Simulate batch of images: (batch=2, height=2, width=2, channels=3)
    x = Tensor(np.random.randn(2, 2, 2, 3), requires_grad=True)
    original_shape = x.shape
    
    # Flatten spatial dimensions: (2, 4, 3)
    y = x.reshape(2, 4, 3)
    assert y.shape == (2, 4, 3)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    assert x.grad.shape == original_shape
    
    print("✅ Batch reshape gradients correct!")


def test_reshape_backward_chain():
    """Test reshape in computation chain."""
    print("\n🔬 Testing reshape in chain...")
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Chain: reshape -> multiply
    y = x.reshape(4)  # flatten
    z = y * 2.0  # multiply by scalar
    
    z.backward(np.ones_like(z.data))
    
    # Gradient should flow through reshape and multiply
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, [[2.0, 2.0], [2.0, 2.0]])
    
    print("✅ Chained reshape gradients correct!")


def test_reshape_backward_3d_to_2d():
    """Test reshaping 3D to 2D (common in linear layers)."""
    print("\n🔬 Testing 3D to 2D reshape...")
    
    # Simulate: (batch=2, seq=3, features=4)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    original_shape = x.shape
    
    # Flatten batch and sequence: (6, 4)
    y = x.reshape(6, 4)
    assert y.shape == (6, 4)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    assert x.grad.shape == original_shape
    
    print("✅ 3D to 2D reshape gradients correct!")


def test_reshape_backward_add_batch_dim():
    """Test adding batch dimension."""
    print("\n🔬 Testing add batch dimension reshape...")
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Add batch dimension: (2, 2) -> (1, 2, 2)
    y = x.reshape(1, 2, 2)
    assert y.shape == (1, 2, 2)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    assert x.grad.shape == x.shape
    assert np.allclose(x.grad, [[1.0, 1.0], [1.0, 1.0]])
    
    print("✅ Add batch dimension reshape gradients correct!")


def test_module():
    """🧪 Module Test: ReshapeBackward Complete Test

    Run all ReshapeBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING RESHAPEBACKWARD MODULE TEST")
    print("="*60)
    
    test_reshape_backward_simple()
    test_reshape_backward_flatten()
    test_reshape_backward_expand()
    test_reshape_backward_batch()
    test_reshape_backward_chain()
    test_reshape_backward_3d_to_2d()
    test_reshape_backward_add_batch_dim()
    
    print("\n" + "="*60)
    print("🎉 ALL RESHAPEBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
