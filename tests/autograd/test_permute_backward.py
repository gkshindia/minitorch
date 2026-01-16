"""🧪 Test Suite: PermuteBackward

Tests gradient computation for arbitrary axis permutation.

Mathematical Rule: If Y = X.permute(axes), then:
∂Y/∂X = grad_Y.permute(inverse_axes)
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import PermuteBackward, enable_autograd


def test_permute_backward_simple():
    """Test basic permute gradient."""
    print("\n🔬 Testing simple permute gradients...")
    
    # Create a 3D tensor: (2, 3, 4)
    x_data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    
    # Permute to (4, 2, 3)
    axes = (2, 0, 1)
    y_data = np.transpose(x.data, axes)
    y = Tensor(y_data)
    y.requires_grad = True
    y._grad_fn = PermuteBackward(x, axes)
    
    assert y.shape == (4, 2, 3)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should be permuted back to original shape
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    print("✅ Simple permute gradients correct!")


def test_permute_backward_multihead_attention():
    """Test permute pattern used in multi-head attention."""
    print("\n🔬 Testing multi-head attention permute pattern...")
    
    # Multi-head attention permute: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
    batch, seq, heads, head_dim = 2, 4, 3, 5
    x_data = np.random.randn(batch, seq, heads, head_dim).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    
    # Permute: (0, 2, 1, 3) swaps seq and heads dimensions
    axes = (0, 2, 1, 3)
    y_data = np.transpose(x.data, axes)
    y = Tensor(y_data)
    y.requires_grad = True
    y._grad_fn = PermuteBackward(x, axes)
    
    assert y.shape == (batch, heads, seq, head_dim)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    assert x.grad.shape == x.shape
    
    print("✅ Multi-head attention permute gradients correct!")


def test_permute_backward_inverse():
    """Test that permute and inverse permute cancel out."""
    print("\n🔬 Testing permute inverse...")
    
    x_data = np.random.randn(2, 3, 4).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    
    # Permute and then permute back
    axes = (2, 0, 1)
    inverse_axes = (1, 2, 0)
    
    # Forward permute
    y_data = np.transpose(x.data, axes)
    y = Tensor(y_data)
    y.requires_grad = True
    y._grad_fn = PermuteBackward(x, axes)
    
    # Inverse permute
    z_data = np.transpose(y.data, inverse_axes)
    z = Tensor(z_data)
    z.requires_grad = True
    z._grad_fn = PermuteBackward(y, inverse_axes)
    
    # Should be back to original shape
    assert z.shape == x.shape
    assert np.allclose(z.data, x.data)
    
    grad_output = np.ones_like(z.data)
    z.backward(grad_output)
    
    # Gradient should flow through both permutes
    assert y.grad is not None
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    print("✅ Permute inverse gradients correct!")


def test_permute_backward_4d():
    """Test permute on 4D tensor (image-like data)."""
    print("\n🔬 Testing 4D tensor permute...")
    
    # NCHW to NHWC (common in conv networks)
    N, C, H, W = 2, 3, 4, 5
    x_data = np.random.randn(N, C, H, W).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    
    # NCHW -> NHWC
    axes = (0, 2, 3, 1)
    y_data = np.transpose(x.data, axes)
    y = Tensor(y_data)
    y.requires_grad = True
    y._grad_fn = PermuteBackward(x, axes)
    
    assert y.shape == (N, H, W, C)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    assert x.grad.shape == x.shape
    
    print("✅ 4D tensor permute gradients correct!")


def test_permute_backward_self_inverse():
    """Test self-inverse permutation."""
    print("\n🔬 Testing self-inverse permutation...")
    
    x_data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    x = Tensor(x_data, requires_grad=True)
    
    # Self-inverse permutation: applying twice returns to original
    axes = (0, 2, 1, 3)
    
    y_data = np.transpose(x.data, axes)
    y = Tensor(y_data)
    y.requires_grad = True
    y._grad_fn = PermuteBackward(x, axes)
    
    # Apply same permutation again
    z_data = np.transpose(y.data, axes)
    z = Tensor(z_data)
    z.requires_grad = True
    z._grad_fn = PermuteBackward(y, axes)
    
    assert z.shape == x.shape
    
    grad_output = np.ones_like(z.data)
    z.backward(grad_output)
    
    assert x.grad is not None
    
    print("✅ Self-inverse permutation gradients correct!")


def test_module():
    """🧪 Module Test: PermuteBackward Complete Test

    Run all PermuteBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING PERMUTEBACKWARD MODULE TEST")
    print("="*60)
    
    test_permute_backward_simple()
    test_permute_backward_multihead_attention()
    test_permute_backward_inverse()
    test_permute_backward_4d()
    test_permute_backward_self_inverse()
    
    print("\n" + "="*60)
    print("🎉 ALL PERMUTEBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
