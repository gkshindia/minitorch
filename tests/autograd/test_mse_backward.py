"""🧪 Test Suite: MSEBackward

Tests gradient computation for Mean Squared Error Loss.

MSE: L = mean((predictions - targets)²)
Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import MSEBackward, enable_autograd


def test_mse_backward_simple():
    """Test basic MSE gradient."""
    print("\n🔬 Testing simple MSE gradients...")
    
    predictions = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    targets = Tensor([1.0, 3.0, 5.0])
    
    # Forward: MSE
    diff = predictions.data - targets.data
    mse = np.mean(diff ** 2)
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    # Backward
    loss.backward()
    
    # Gradient: 2 * (pred - target) / N
    N = predictions.data.size
    expected_grad = 2 * diff / N
    assert np.allclose(predictions.grad, expected_grad)
    
    print("✅ Simple MSE gradients correct!")


def test_mse_backward_perfect_prediction():
    """Test MSE when predictions match targets."""
    print("\n🔬 Testing MSE with perfect predictions...")
    
    predictions = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    targets = Tensor([1.0, 2.0, 3.0])
    
    diff = predictions.data - targets.data
    mse = np.mean(diff ** 2)
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    # Loss should be zero
    assert np.allclose(loss.data, 0.0)
    
    loss.backward()
    
    # Gradients should be zero (no error)
    assert np.allclose(predictions.grad, 0.0)
    
    print("✅ MSE with perfect predictions gradients correct!")


def test_mse_backward_single_sample():
    """Test MSE with single prediction."""
    print("\n🔬 Testing single sample MSE...")
    
    predictions = Tensor([5.0], requires_grad=True)
    targets = Tensor([3.0])
    
    diff = predictions.data - targets.data
    mse = np.mean(diff ** 2)
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    # MSE = (5-3)² = 4
    assert np.allclose(loss.data, 4.0)
    
    loss.backward()
    
    # Gradient: 2 * (5-3) / 1 = 4
    assert np.allclose(predictions.grad, [4.0])
    
    print("✅ Single sample MSE gradients correct!")


def test_mse_backward_batch():
    """Test MSE with batch of predictions."""
    print("\n🔬 Testing batch MSE...")
    
    # Batch of predictions: (batch_size=4, features=2)
    predictions = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    targets = Tensor([[1.5, 2.5], [2.5, 3.5], [5.5, 5.5], [7.5, 7.5]])
    
    diff = predictions.data - targets.data
    mse = np.mean(diff ** 2)
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    loss.backward()
    
    # Gradient: 2 * (pred - target) / N where N = total elements
    N = predictions.data.size
    expected_grad = 2 * diff / N
    assert np.allclose(predictions.grad, expected_grad)
    
    print("✅ Batch MSE gradients correct!")


def test_mse_backward_large_error():
    """Test MSE with large errors."""
    print("\n🔬 Testing MSE with large errors...")
    
    predictions = Tensor([10.0, 20.0], requires_grad=True)
    targets = Tensor([1.0, 2.0])
    
    diff = predictions.data - targets.data  # [9, 18]
    mse = np.mean(diff ** 2)  # (81 + 324) / 2 = 202.5
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    assert np.allclose(loss.data, 202.5)
    
    loss.backward()
    
    # Large errors should produce large gradients
    expected_grad = 2 * diff / 2  # [9, 18]
    assert np.allclose(predictions.grad, expected_grad)
    
    print("✅ MSE with large errors gradients correct!")


def test_mse_backward_regression():
    """Test MSE in regression context."""
    print("\n🔬 Testing MSE for regression...")
    
    # Simulate simple linear regression
    batch_size = 5
    predictions = Tensor(np.random.randn(batch_size), requires_grad=True)
    targets = Tensor(np.random.randn(batch_size))
    
    diff = predictions.data - targets.data
    mse = np.mean(diff ** 2)
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    loss.backward()
    
    # Verify gradient shape and values
    assert predictions.grad.shape == predictions.shape
    expected_grad = 2 * diff / batch_size
    assert np.allclose(predictions.grad, expected_grad)
    
    print("✅ MSE for regression gradients correct!")


def test_mse_backward_multidimensional():
    """Test MSE with multidimensional data."""
    print("\n🔬 Testing multidimensional MSE...")
    
    # Image-like data: (batch=2, height=3, width=3)
    predictions = Tensor(np.random.randn(2, 3, 3), requires_grad=True)
    targets = Tensor(np.random.randn(2, 3, 3))
    
    diff = predictions.data - targets.data
    mse = np.mean(diff ** 2)
    loss = Tensor(mse)
    loss.requires_grad = True
    loss._grad_fn = MSEBackward(predictions, targets)
    
    loss.backward()
    
    # Gradient computed over all elements
    N = predictions.data.size
    expected_grad = 2 * diff / N
    assert np.allclose(predictions.grad, expected_grad)
    
    print("✅ Multidimensional MSE gradients correct!")


def test_module():
    """🧪 Module Test: MSEBackward Complete Test

    Run all MSEBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING MSEBACKWARD MODULE TEST")
    print("="*60)
    
    test_mse_backward_simple()
    test_mse_backward_perfect_prediction()
    test_mse_backward_single_sample()
    test_mse_backward_batch()
    test_mse_backward_large_error()
    test_mse_backward_regression()
    test_mse_backward_multidimensional()
    
    print("\n" + "="*60)
    print("🎉 ALL MSEBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
