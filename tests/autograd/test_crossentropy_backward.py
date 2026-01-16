"""🧪 Test Suite: CrossEntropyBackward

Tests gradient computation for Cross-Entropy Loss.

CrossEntropy: L = -mean(log_softmax(logits)[targets])
Gradient: ∂L/∂logits = (softmax(logits) - one_hot(targets)) / batch_size

This is one of the most elegant gradients in ML!
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import CrossEntropyBackward, enable_autograd


def softmax(x):
    """Helper: compute softmax."""
    x_max = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def test_crossentropy_backward_simple():
    """Test basic cross-entropy gradient."""
    print("\n🔬 Testing simple cross-entropy gradients...")
    
    # Batch of 3 samples, 4 classes
    logits = Tensor([[2.0, 1.0, 0.1, 0.5], 
                     [0.5, 2.5, 0.2, 0.3],
                     [1.0, 0.5, 2.0, 0.1]], requires_grad=True)
    targets = Tensor(np.array([0, 1, 2]))  # Class indices
    
    # Forward: cross-entropy
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Compute softmax
    probs = softmax(logits.data)
    
    # Compute log-softmax and select target classes
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    # Backward
    loss.backward()
    
    # Gradient: (softmax - one_hot) / batch_size
    one_hot = np.zeros((batch_size, num_classes))
    one_hot[np.arange(batch_size), target_indices] = 1.0
    expected_grad = (probs - one_hot) / batch_size
    
    assert np.allclose(logits.grad, expected_grad, rtol=1e-5)
    
    print("✅ Simple cross-entropy gradients correct!")


def test_crossentropy_backward_perfect_prediction():
    """Test CE with perfect predictions."""
    print("\n🔬 Testing CE with perfect predictions...")
    
    # Logits strongly predict correct class
    logits = Tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]], requires_grad=True)
    targets = Tensor(np.array([0, 1]))
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    probs = softmax(logits.data)
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    # Loss should be very small
    assert loss.data < 0.01
    
    loss.backward()
    
    # Gradients should be very small (predictions are correct)
    assert np.abs(logits.grad).max() < 0.01
    
    print("✅ CE with perfect predictions gradients correct!")


def test_crossentropy_backward_uniform():
    """Test CE with uniform predictions."""
    print("\n🔬 Testing CE with uniform predictions...")
    
    # All logits equal (uniform distribution after softmax)
    logits = Tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]], requires_grad=True)
    targets = Tensor(np.array([0, 2]))
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    probs = softmax(logits.data)
    
    # All probabilities should be 1/3
    assert np.allclose(probs, 1.0/3.0, rtol=1e-5)
    
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    loss.backward()
    
    # Verify gradient computed
    assert logits.grad is not None
    
    print("✅ CE with uniform predictions gradients correct!")


def test_crossentropy_backward_wrong_prediction():
    """Test CE when prediction is wrong."""
    print("\n🔬 Testing CE with wrong predictions...")
    
    # Strongly predict wrong class
    logits = Tensor([[0.0, 100.0, 0.0], [100.0, 0.0, 0.0]], requires_grad=True)
    targets = Tensor(np.array([0, 1]))  # Correct classes are 0 and 1
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    probs = softmax(logits.data)
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    # Loss should be large
    assert loss.data > 10.0
    
    loss.backward()
    
    # Gradients should be large (predictions are wrong)
    assert np.abs(logits.grad).max() > 0.1
    
    print("✅ CE with wrong predictions gradients correct!")


def test_crossentropy_backward_multiclass():
    """Test CE with many classes."""
    print("\n🔬 Testing multiclass CE...")
    
    batch_size, num_classes = 8, 10
    logits = Tensor(np.random.randn(batch_size, num_classes), requires_grad=True)
    targets = Tensor(np.random.randint(0, num_classes, batch_size))
    
    probs = softmax(logits.data)
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    loss.backward()
    
    # Verify gradient shape
    assert logits.grad.shape == logits.shape
    
    # Each row's gradients should sum to 0 (property of softmax gradient)
    row_sums = np.sum(logits.grad, axis=1)
    assert np.allclose(row_sums, 0.0, atol=1e-5)
    
    print("✅ Multiclass CE gradients correct!")


def test_crossentropy_backward_elegant_gradient():
    """Test the elegant CE gradient formula."""
    print("\n🔬 Testing elegant CE gradient formula...")
    
    logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    targets = Tensor(np.array([2]))  # Target is class 2
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    probs = softmax(logits.data)
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    loss.backward()
    
    # The elegant formula: gradient = (softmax - one_hot) / batch_size
    one_hot = np.array([[0.0, 0.0, 1.0]])
    expected_grad = (probs - one_hot) / batch_size
    
    assert np.allclose(logits.grad, expected_grad, rtol=1e-5)
    
    # This is the beauty of cross-entropy with softmax!
    # The gradient is simply: predicted probability - actual label
    print("✅ Elegant CE gradient formula verified!")


def test_crossentropy_backward_numerical_stability():
    """Test CE numerical stability."""
    print("\n🔬 Testing CE numerical stability...")
    
    # Large logits that could cause overflow
    logits = Tensor([[100.0, 101.0, 99.0], [50.0, 51.0, 52.0]], requires_grad=True)
    targets = Tensor(np.array([1, 2]))
    
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    probs = softmax(logits.data)
    log_probs = np.log(probs + 1e-7)
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    
    loss = Tensor(ce_loss)
    loss.requires_grad = True
    loss._grad_fn = CrossEntropyBackward(logits, targets)
    
    loss.backward()
    
    # Should not have NaN or Inf
    assert not np.any(np.isnan(logits.grad))
    assert not np.any(np.isinf(logits.grad))
    
    print("✅ CE numerical stability correct!")


def test_module():
    """🧪 Module Test: CrossEntropyBackward Complete Test

    Run all CrossEntropyBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING CROSSENTROPYBACKWARD MODULE TEST")
    print("="*60)
    
    test_crossentropy_backward_simple()
    test_crossentropy_backward_perfect_prediction()
    test_crossentropy_backward_uniform()
    test_crossentropy_backward_wrong_prediction()
    test_crossentropy_backward_multiclass()
    test_crossentropy_backward_elegant_gradient()
    test_crossentropy_backward_numerical_stability()
    
    print("\n" + "="*60)
    print("🎉 ALL CROSSENTROPYBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
