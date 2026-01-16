"""🧪 Test Suite: ReLUBackward

Tests gradient computation for ReLU activation.

ReLU: f(x) = max(0, x)
Derivative: f'(x) = 1 if x > 0, else 0
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import ReLUBackward, enable_autograd


def test_relu_backward_simple():
    """Test basic ReLU gradient."""
    print("\n🔬 Testing simple ReLU gradients...")
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Forward: ReLU
    relu_data = np.maximum(0, x.data)
    y = Tensor(relu_data)
    y.requires_grad = True
    y._grad_fn = ReLUBackward(x)
    
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    assert np.allclose(y.data, expected)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient: 1 where x > 0, else 0
    expected_grad = [0.0, 0.0, 0.0, 1.0, 1.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Simple ReLU gradients correct!")


def test_relu_backward_all_positive():
    """Test ReLU on all positive inputs."""
    print("\n🔬 Testing ReLU on all positive...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    relu_data = np.maximum(0, x.data)
    y = Tensor(relu_data)
    y.requires_grad = True
    y._grad_fn = ReLUBackward(x)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # All gradients should be 1 (all inputs > 0)
    assert np.allclose(x.grad, [1.0, 1.0, 1.0, 1.0])
    
    print("✅ ReLU on all positive gradients correct!")


def test_relu_backward_all_negative():
    """Test ReLU on all negative inputs."""
    print("\n🔬 Testing ReLU on all negative...")
    
    x = Tensor([-1.0, -2.0, -3.0, -4.0], requires_grad=True)
    
    relu_data = np.maximum(0, x.data)
    y = Tensor(relu_data)
    y.requires_grad = True
    y._grad_fn = ReLUBackward(x)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # All gradients should be 0 (all inputs < 0)
    assert np.allclose(x.grad, [0.0, 0.0, 0.0, 0.0])
    
    print("✅ ReLU on all negative gradients correct!")


def test_relu_backward_matrix():
    """Test ReLU on matrix."""
    print("\n🔬 Testing matrix ReLU...")
    
    x = Tensor([[-1.0, 2.0], [3.0, -4.0], [0.0, 5.0]], requires_grad=True)
    
    relu_data = np.maximum(0, x.data)
    y = Tensor(relu_data)
    y.requires_grad = True
    y._grad_fn = ReLUBackward(x)
    
    expected = [[0.0, 2.0], [3.0, 0.0], [0.0, 5.0]]
    assert np.allclose(y.data, expected)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    expected_grad = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Matrix ReLU gradients correct!")


def test_relu_backward_weighted():
    """Test ReLU with weighted gradient."""
    print("\n🔬 Testing weighted ReLU gradients...")
    
    x = Tensor([-1.0, 2.0, 3.0], requires_grad=True)
    
    relu_data = np.maximum(0, x.data)
    y = Tensor(relu_data)
    y.requires_grad = True
    y._grad_fn = ReLUBackward(x)
    
    # Different gradient for each position
    grad_output = np.array([1.0, 2.0, 3.0])
    y.backward(grad_output)
    
    # Gradient: mask * grad_output
    expected_grad = [0.0, 2.0, 3.0]  # -1 masked, 2 and 3 get their weights
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Weighted ReLU gradients correct!")


def test_relu_backward_chain():
    """Test ReLU in computation chain."""
    print("\n🔬 Testing ReLU in chain...")
    
    x = Tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)
    
    # Chain: multiply -> ReLU
    y = x * 2.0  # [-4, -2, 2, 4]
    relu_data = np.maximum(0, y.data)
    z = Tensor(relu_data)
    z.requires_grad = True
    z._grad_fn = ReLUBackward(y)
    
    z.backward(np.ones_like(z.data))
    
    # Gradient flows through ReLU then multiply
    # Only positive values (1, 2) get gradient of 2
    expected_grad = [0.0, 0.0, 2.0, 2.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Chained ReLU gradients correct!")


def test_relu_backward_near_zero():
    """Test ReLU behavior near zero."""
    print("\n🔬 Testing ReLU near zero...")
    
    x = Tensor([-0.001, -0.0, 0.0, 0.001], requires_grad=True)
    
    relu_data = np.maximum(0, x.data)
    y = Tensor(relu_data)
    y.requires_grad = True
    y._grad_fn = ReLUBackward(x)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Gradient: 0 for negative/zero, 1 for positive
    expected_grad = [0.0, 0.0, 0.0, 1.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ ReLU near zero gradients correct!")


def test_module():
    """🧪 Module Test: ReLUBackward Complete Test

    Run all ReLUBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING RELUBACKWARD MODULE TEST")
    print("="*60)
    
    test_relu_backward_simple()
    test_relu_backward_all_positive()
    test_relu_backward_all_negative()
    test_relu_backward_matrix()
    test_relu_backward_weighted()
    test_relu_backward_chain()
    test_relu_backward_near_zero()
    
    print("\n" + "="*60)
    print("🎉 ALL RELUBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
