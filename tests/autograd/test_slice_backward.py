"""🧪 Test Suite: SliceBackward

Tests gradient computation for tensor slicing/indexing operations.

Mathematical Rule: If Y = X[key], then:
- ∂Loss/∂X[key] = grad_output
- ∂Loss/∂X[other positions] = 0
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import SliceBackward, enable_autograd


def test_slice_backward_simple():
    """Test basic slicing gradient."""
    print("\n🔬 Testing simple slice gradients...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    
    # Slice: select elements 1:4
    y = x[1:4]
    assert np.allclose(y.data, [2.0, 3.0, 4.0])
    
    # Backward
    grad_output = np.array([1.0, 2.0, 3.0])
    y.backward(grad_output)
    
    # Only sliced positions should have gradients
    expected_grad = [0.0, 1.0, 2.0, 3.0, 0.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Simple slice gradients correct!")


def test_slice_backward_single_element():
    """Test single element selection."""
    print("\n🔬 Testing single element slice...")
    
    x = Tensor([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    
    # Select single element: x[2]
    y = x[2:3]  # Use slice to maintain dimension
    assert np.allclose(y.data, [30.0])
    
    grad_output = np.array([5.0])
    y.backward(grad_output)
    
    # Only index 2 should have gradient
    expected_grad = [0.0, 0.0, 5.0, 0.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Single element slice gradients correct!")


def test_slice_backward_2d():
    """Test 2D slicing."""
    print("\n🔬 Testing 2D slice gradients...")
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
    
    # Slice: select first two rows, last two columns
    y = x[0:2, 1:3]
    expected = [[2.0, 3.0], [5.0, 6.0]]
    assert np.allclose(y.data, expected)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Only sliced region should have gradients
    expected_grad = [[0.0, 1.0, 1.0], 
                     [0.0, 1.0, 1.0], 
                     [0.0, 0.0, 0.0]]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ 2D slice gradients correct!")


def test_slice_backward_step():
    """Test slicing with step."""
    print("\n🔬 Testing slice with step...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    
    # Slice with step: every other element
    y = x[::2]
    assert np.allclose(y.data, [1.0, 3.0, 5.0])
    
    grad_output = np.array([1.0, 2.0, 3.0])
    y.backward(grad_output)
    
    # Only selected positions should have gradients
    expected_grad = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Slice with step gradients correct!")


def test_slice_backward_negative_indices():
    """Test slicing with negative indices."""
    print("\n🔬 Testing negative index slicing...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    
    # Slice: last 3 elements
    y = x[-3:]
    assert np.allclose(y.data, [3.0, 4.0, 5.0])
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Last 3 positions should have gradients
    expected_grad = [0.0, 0.0, 1.0, 1.0, 1.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Negative index slice gradients correct!")


def test_slice_backward_batch():
    """Test slicing batch dimension."""
    print("\n🔬 Testing batch slice...")
    
    # Batch of data: (batch=4, features=3)
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    
    # Select first 2 samples from batch
    y = x[0:2, :]
    assert y.shape == (2, 3)
    
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # First 2 rows should have gradients, last 2 should be zero
    assert not np.allclose(x.grad[0:2], 0.0)
    assert np.allclose(x.grad[2:4], 0.0)
    
    print("✅ Batch slice gradients correct!")


def test_slice_backward_chain():
    """Test slicing in computation chain."""
    print("\n🔬 Testing slice in chain...")
    
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    
    # Chain: slice then multiply
    y = x[1:3]  # [2.0, 3.0]
    z = y * 2.0  # [4.0, 6.0]
    
    z.backward(np.ones_like(z.data))
    
    # Gradient should flow through multiply and slice
    expected_grad = [0.0, 2.0, 2.0, 0.0]
    assert np.allclose(x.grad, expected_grad)
    
    print("✅ Chained slice gradients correct!")


def test_module():
    """🧪 Module Test: SliceBackward Complete Test

    Run all SliceBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING SLICEBACKWARD MODULE TEST")
    print("="*60)
    
    test_slice_backward_simple()
    test_slice_backward_single_element()
    test_slice_backward_2d()
    test_slice_backward_step()
    test_slice_backward_negative_indices()
    test_slice_backward_batch()
    test_slice_backward_chain()
    
    print("\n" + "="*60)
    print("🎉 ALL SLICEBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
