"""🧪 Test Suite: BCEBackward

Tests gradient computation for Binary Cross-Entropy Loss.

BCE: L = -[y*log(p) + (1-y)*log(1-p)]
Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import BCEBackward, enable_autograd, EPSILON


def test_bce_backward_simple():
    """Test basic BCE gradient."""
    print("\n🔬 Testing simple BCE gradients...")
    
    # Predictions should be probabilities in [0, 1]
    predictions = Tensor([0.7, 0.3, 0.9], requires_grad=True)
    targets = Tensor([1.0, 0.0, 1.0])
    
    # Forward: BCE
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    # Backward
    loss.backward()
    
    # Gradient: (p - y) / (p * (1-p) * N)
    N = predictions.data.size
    expected_grad = (p - y) / (p * (1 - p) * N)
    assert np.allclose(predictions.grad, expected_grad, rtol=1e-5)
    
    print("✅ Simple BCE gradients correct!")


def test_bce_backward_perfect_prediction():
    """Test BCE with perfect predictions."""
    print("\n🔬 Testing BCE with perfect predictions...")
    
    # Perfect predictions (but not exactly 0 or 1 due to numerical stability)
    predictions = Tensor([0.9999, 0.0001, 0.9999], requires_grad=True)
    targets = Tensor([1.0, 0.0, 1.0])
    
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    # Loss should be very small
    assert loss.data < 0.01
    
    loss.backward()
    
    # Gradients should be very small (predictions are correct)
    assert np.abs(predictions.grad).max() < 1.0
    
    print("✅ BCE with perfect predictions gradients correct!")


def test_bce_backward_worst_prediction():
    """Test BCE with worst predictions."""
    print("\n🔬 Testing BCE with worst predictions...")
    
    # Worst predictions: opposite of targets
    predictions = Tensor([0.1, 0.9, 0.1], requires_grad=True)
    targets = Tensor([1.0, 0.0, 1.0])
    
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    # Loss should be high
    assert loss.data > 1.0
    
    loss.backward()
    
    # Gradients should be large (predictions are wrong)
    assert np.abs(predictions.grad).max() > 1.0
    
    print("✅ BCE with worst predictions gradients correct!")


def test_bce_backward_batch():
    """Test BCE with batch of predictions."""
    print("\n🔬 Testing batch BCE...")
    
    batch_size = 10
    predictions = Tensor(np.random.uniform(0.1, 0.9, batch_size), requires_grad=True)
    targets = Tensor(np.random.randint(0, 2, batch_size).astype(float))
    
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    loss.backward()
    
    # Verify gradient shape
    assert predictions.grad.shape == predictions.shape
    
    print("✅ Batch BCE gradients correct!")


def test_bce_backward_edge_cases():
    """Test BCE with edge case probabilities."""
    print("\n🔬 Testing BCE edge cases...")
    
    # Test near 0 and 1 (clipping should prevent log(0))
    predictions = Tensor([0.001, 0.5, 0.999], requires_grad=True)
    targets = Tensor([0.0, 0.5, 1.0])
    
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    loss.backward()
    
    # Should not have NaN or Inf
    assert not np.any(np.isnan(predictions.grad))
    assert not np.any(np.isinf(predictions.grad))
    
    print("✅ BCE edge cases gradients correct!")


def test_bce_backward_binary_classification():
    """Test BCE in binary classification context."""
    print("\n🔬 Testing BCE for binary classification...")
    
    # Simulate binary classification with sigmoid outputs
    # After sigmoid, probabilities are in (0, 1)
    predictions = Tensor([0.8, 0.6, 0.3, 0.9, 0.4], requires_grad=True)
    targets = Tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    loss.backward()
    
    # Verify gradients point in right direction
    # When target=1 and pred<1, gradient should be positive (increase pred)
    # When target=0 and pred>0, gradient should be negative (decrease pred)
    N = predictions.data.size
    expected_grad = (p - y) / (p * (1 - p) * N)
    assert np.allclose(predictions.grad, expected_grad, rtol=1e-5)
    
    print("✅ BCE for binary classification gradients correct!")


def test_bce_backward_numerical_stability():
    """Test BCE numerical stability with clipping."""
    print("\n🔬 Testing BCE numerical stability...")
    
    # Test exact 0 and 1 (should be clipped)
    predictions = Tensor([0.0, 0.5, 1.0], requires_grad=True)
    targets = Tensor([0.0, 0.5, 1.0])
    
    eps = EPSILON
    p = np.clip(predictions.data, eps, 1 - eps)
    y = targets.data
    bce_per_sample = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    bce = np.mean(bce_per_sample)
    loss = Tensor(bce)
    loss.requires_grad = True
    loss._grad_fn = BCEBackward(predictions, targets)
    
    loss.backward()
    
    # Should not explode
    assert not np.any(np.isnan(predictions.grad))
    assert not np.any(np.isinf(predictions.grad))
    assert np.abs(predictions.grad).max() < 1e6
    
    print("✅ BCE numerical stability correct!")


def test_module():
    """🧪 Module Test: BCEBackward Complete Test

    Run all BCEBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING BCEBACKWARD MODULE TEST")
    print("="*60)
    
    test_bce_backward_simple()
    test_bce_backward_perfect_prediction()
    test_bce_backward_worst_prediction()
    test_bce_backward_batch()
    test_bce_backward_edge_cases()
    test_bce_backward_binary_classification()
    test_bce_backward_numerical_stability()
    
    print("\n" + "="*60)
    print("🎉 ALL BCEBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
