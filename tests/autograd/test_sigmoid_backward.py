"""🧪 Test Suite: SigmoidBackward

Tests gradient computation for sigmoid activation.

Sigmoid: σ(x) = 1/(1 + exp(-x))
Derivative: σ'(x) = σ(x) * (1 - σ(x))
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import SigmoidBackward, enable_autograd


def sigmoid(x):
    """Helper: compute sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def test_sigmoid_backward_simple():
    """Test basic sigmoid gradient."""
    print("\n🔬 Testing simple sigmoid gradients...")
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Forward: sigmoid
    sigmoid_data = sigmoid(x.data)
    y = Tensor(sigmoid_data)
    y.requires_grad = True
    y._grad_fn = SigmoidBackward(x, y)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient: σ(x) * (1 - σ(x))
    expected_grad = sigmoid_data * (1 - sigmoid_data)
    assert np.allclose(x.grad, expected_grad, rtol=1e-5)
    
    print("✅ Simple sigmoid gradients correct!")


def test_sigmoid_backward_zero():
    """Test sigmoid at zero."""
    print("\n🔬 Testing sigmoid at zero...")
    
    x = Tensor([0.0], requires_grad=True)
    
    sigmoid_data = sigmoid(x.data)
    y = Tensor(sigmoid_data)
    y.requires_grad = True
    y._grad_fn = SigmoidBackward(x, y)
    
    # σ(0) = 0.5
    assert np.allclose(y.data, [0.5])
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # σ'(0) = 0.5 * 0.5 = 0.25
    assert np.allclose(x.grad, [0.25])
    
    print("✅ Sigmoid at zero gradient correct!")


def test_sigmoid_backward_extremes():
    """Test sigmoid at extreme values."""
    print("\n🔬 Testing sigmoid at extremes...")
    
    x = Tensor([-10.0, 10.0], requires_grad=True)
    
    sigmoid_data = sigmoid(x.data)
    y = Tensor(sigmoid_data)
    y.requires_grad = True
    y._grad_fn = SigmoidBackward(x, y)
    
    # σ(-10) ≈ 0, σ(10) ≈ 1
    assert y.data[0] < 0.01  # Very close to 0
    assert y.data[1] > 0.99  # Very close to 1
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradients at extremes should be very small
    assert x.grad[0] < 0.001
    assert x.grad[1] < 0.001
    
    print("✅ Sigmoid at extremes gradients correct!")


def test_sigmoid_backward_matrix():
    """Test sigmoid on matrix."""
    print("\n🔬 Testing matrix sigmoid...")
    
    x = Tensor([[-1.0, 0.0], [1.0, 2.0]], requires_grad=True)
    
    sigmoid_data = sigmoid(x.data)
    y = Tensor(sigmoid_data)
    y.requires_grad = True
    y._grad_fn = SigmoidBackward(x, y)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient: σ(x) * (1 - σ(x))
    expected_grad = sigmoid_data * (1 - sigmoid_data)
    assert np.allclose(x.grad, expected_grad, rtol=1e-5)
    
    print("✅ Matrix sigmoid gradients correct!")


def test_sigmoid_backward_weighted():
    """Test sigmoid with weighted gradient."""
    print("\n🔬 Testing weighted sigmoid gradients...")
    
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    
    sigmoid_data = sigmoid(x.data)
    y = Tensor(sigmoid_data)
    y.requires_grad = True
    y._grad_fn = SigmoidBackward(x, y)
    
    # Different gradient for each position
    grad_output = np.array([1.0, 2.0, 3.0])
    y.backward(grad_output)
    
    # Gradient: grad_output * σ(x) * (1 - σ(x))
    sigmoid_grad = sigmoid_data * (1 - sigmoid_data)
    expected_grad = grad_output * sigmoid_grad
    assert np.allclose(x.grad, expected_grad, rtol=1e-5)
    
    print("✅ Weighted sigmoid gradients correct!")


def test_sigmoid_backward_chain():
    """Test sigmoid in computation chain."""
    print("\n🔬 Testing sigmoid in chain...")
    
    x = Tensor([1.0, 2.0], requires_grad=True)
    
    # Chain: multiply -> sigmoid
    y = x * 2.0  # [2, 4]
    sigmoid_data = sigmoid(y.data)
    z = Tensor(sigmoid_data)
    z.requires_grad = True
    z._grad_fn = SigmoidBackward(y, z)
    
    z.backward(np.ones_like(z.data))
    
    # Gradient flows through sigmoid then multiply
    sigmoid_grad = sigmoid_data * (1 - sigmoid_data)
    expected_grad = 2.0 * sigmoid_grad
    assert np.allclose(x.grad, expected_grad, rtol=1e-5)
    
    print("✅ Chained sigmoid gradients correct!")


def test_sigmoid_backward_bce_simulation():
    """Test sigmoid in binary classification."""
    print("\n🔬 Testing sigmoid for binary classification...")
    
    logits = Tensor([0.5, -0.5, 2.0], requires_grad=True)
    
    # Sigmoid to get probabilities
    probs_data = sigmoid(logits.data)
    probs = Tensor(probs_data)
    probs.requires_grad = True
    probs._grad_fn = SigmoidBackward(logits, probs)
    
    # Simulate BCE gradient (p - y)
    targets = np.array([1.0, 0.0, 1.0])
    bce_grad = probs_data - targets
    
    probs.backward(bce_grad)
    
    # Verify gradient shape
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    
    print("✅ Sigmoid for binary classification gradients correct!")


def test_module():
    """🧪 Module Test: SigmoidBackward Complete Test

    Run all SigmoidBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING SIGMOIDBACKWARD MODULE TEST")
    print("="*60)
    
    test_sigmoid_backward_simple()
    test_sigmoid_backward_zero()
    test_sigmoid_backward_extremes()
    test_sigmoid_backward_matrix()
    test_sigmoid_backward_weighted()
    test_sigmoid_backward_chain()
    test_sigmoid_backward_bce_simulation()
    
    print("\n" + "="*60)
    print("🎉 ALL SIGMOIDBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
