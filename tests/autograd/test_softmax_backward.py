"""🧪 Test Suite: SoftmaxBackward

Tests gradient computation for softmax activation.

Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x))
Gradient: ∂L/∂x[i] = softmax[i] * (∂L/∂y[i] - sum(∂L/∂y * softmax))
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import SoftmaxBackward, enable_autograd


def softmax(x, axis=-1):
    """Helper: compute softmax."""
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def test_softmax_backward_simple():
    """Test basic softmax gradient."""
    print("\n🔬 Testing simple softmax gradients...")
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Forward: softmax
    softmax_data = softmax(x.data)
    y = Tensor(softmax_data)
    y.requires_grad = True
    y._grad_fn = SoftmaxBackward(x, y, dim=-1)
    
    # Verify softmax sums to 1
    assert np.allclose(np.sum(softmax_data), 1.0)
    
    # Backward with uniform gradient
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should sum to 0 (property of softmax)
    assert np.allclose(np.sum(x.grad), 0.0, atol=1e-6)
    
    print("✅ Simple softmax gradients correct!")


def test_softmax_backward_uniform():
    """Test softmax on uniform inputs."""
    print("\n🔬 Testing softmax on uniform inputs...")
    
    x = Tensor([1.0, 1.0, 1.0], requires_grad=True)
    
    softmax_data = softmax(x.data)
    y = Tensor(softmax_data)
    y.requires_grad = True
    y._grad_fn = SoftmaxBackward(x, y, dim=-1)
    
    # All outputs should be 1/3
    assert np.allclose(softmax_data, [1/3, 1/3, 1/3])
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient should sum to 0
    assert np.allclose(np.sum(x.grad), 0.0, atol=1e-6)
    
    print("✅ Softmax on uniform inputs gradients correct!")


def test_softmax_backward_batch():
    """Test softmax on batch."""
    print("\n🔬 Testing batch softmax...")
    
    # Batch of logits: (batch=3, classes=4)
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    
    softmax_data = softmax(x.data, axis=-1)
    y = Tensor(softmax_data)
    y.requires_grad = True
    y._grad_fn = SoftmaxBackward(x, y, dim=-1)
    
    # Each row should sum to 1
    row_sums = np.sum(softmax_data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0, 1.0])
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Each row's gradients should sum to 0
    grad_row_sums = np.sum(x.grad, axis=-1)
    assert np.allclose(grad_row_sums, [0.0, 0.0, 0.0], atol=1e-6)
    
    print("✅ Batch softmax gradients correct!")


def test_softmax_backward_cross_entropy():
    """Test softmax gradient in cross-entropy context."""
    print("\n🔬 Testing softmax with cross-entropy gradient...")
    
    logits = Tensor([2.0, 1.0, 0.1], requires_grad=True)
    
    softmax_data = softmax(logits.data)
    probs = Tensor(softmax_data)
    probs.requires_grad = True
    probs._grad_fn = SoftmaxBackward(logits, probs, dim=-1)
    
    # Simulate cross-entropy gradient: softmax - one_hot
    target_class = 0  # First class is correct
    one_hot = np.array([1.0, 0.0, 0.0])
    ce_grad = softmax_data - one_hot
    
    probs.backward(ce_grad)
    
    # Softmax backward applies the Jacobian to the gradient
    # For cross-entropy, this produces: softmax * (ce_grad - sum(ce_grad * softmax))
    sum_term = np.sum(ce_grad * softmax_data)
    expected_grad = softmax_data * (ce_grad - sum_term)
    assert np.allclose(logits.grad, expected_grad, rtol=1e-5)
    
    print("✅ Softmax with cross-entropy gradients correct!")


def test_softmax_backward_weighted():
    """Test softmax with weighted gradient."""
    print("\n🔬 Testing weighted softmax gradients...")
    
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    softmax_data = softmax(x.data)
    y = Tensor(softmax_data)
    y.requires_grad = True
    y._grad_fn = SoftmaxBackward(x, y, dim=-1)
    
    # Different gradient for each class
    grad_output = np.array([1.0, 2.0, 3.0])
    y.backward(grad_output)
    
    # Verify gradient computed
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    print("✅ Weighted softmax gradients correct!")


def test_softmax_backward_2d():
    """Test softmax on 2D (multiclass classification)."""
    print("\n🔬 Testing 2D softmax...")
    
    # Batch of logits for 4-class problem
    x = Tensor([[1.0, 2.0, 0.5, 0.1], 
                [2.0, 1.0, 3.0, 0.5]], requires_grad=True)
    
    softmax_data = softmax(x.data, axis=-1)
    y = Tensor(softmax_data)
    y.requires_grad = True
    y._grad_fn = SoftmaxBackward(x, y, dim=-1)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Each sample's gradients should sum to 0
    grad_sums = np.sum(x.grad, axis=-1)
    assert np.allclose(grad_sums, [0.0, 0.0], atol=1e-6)
    
    print("✅ 2D softmax gradients correct!")


def test_softmax_backward_numerical_stability():
    """Test softmax with large values."""
    print("\n🔬 Testing softmax numerical stability...")
    
    # Large values that could cause overflow
    x = Tensor([1000.0, 1000.1, 999.9], requires_grad=True)
    
    softmax_data = softmax(x.data)
    y = Tensor(softmax_data)
    y.requires_grad = True
    y._grad_fn = SoftmaxBackward(x, y, dim=-1)
    
    # Should not have NaN or Inf
    assert not np.any(np.isnan(softmax_data))
    assert not np.any(np.isinf(softmax_data))
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradients should also be finite
    assert not np.any(np.isnan(x.grad))
    assert not np.any(np.isinf(x.grad))
    
    print("✅ Softmax numerical stability correct!")


def test_module():
    """🧪 Module Test: SoftmaxBackward Complete Test

    Run all SoftmaxBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING SOFTMAXBACKWARD MODULE TEST")
    print("="*60)
    
    test_softmax_backward_simple()
    test_softmax_backward_uniform()
    test_softmax_backward_batch()
    test_softmax_backward_cross_entropy()
    test_softmax_backward_weighted()
    test_softmax_backward_2d()
    test_softmax_backward_numerical_stability()
    
    print("\n" + "="*60)
    print("🎉 ALL SOFTMAXBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
