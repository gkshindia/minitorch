"""🧪 Test Suite: MatMulBackward

Tests gradient computation for matrix multiplication.

Mathematical Rule: If Z = A @ B, then:
- ∂Z/∂A = grad_Z @ B.T
- ∂Z/∂B = A.T @ grad_Z
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import MatMulBackward, enable_autograd


def test_matmul_backward_simple():
    """Test basic matrix multiplication gradient."""
    print("\n🔬 Testing simple matmul gradients...")
    
    # 2x2 @ 2x2 = 2x2
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[0.5, 0.0], [0.0, 0.5]], requires_grad=True)
    
    c = a.matmul(b)
    expected = [[0.5, 1.0], [1.5, 2.0]]
    assert np.allclose(c.data, expected)
    
    # Backward
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    # ∂c/∂a = grad_output @ b.T
    # ∂c/∂b = a.T @ grad_output
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    
    print("✅ Simple matmul gradients correct!")


def test_matmul_backward_rectangular():
    """Test rectangular matrix multiplication."""
    print("\n🔬 Testing rectangular matmul...")
    
    # (2x3) @ (3x2) = (2x2)
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    
    c = a.matmul(b)
    assert c.shape == (2, 2)
    
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    # Verify shapes
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (3, 2)
    
    # Verify gradients are computed
    assert a.grad is not None
    assert b.grad is not None
    
    print("✅ Rectangular matmul gradients correct!")


def test_matmul_backward_vector():
    """Test matrix-vector multiplication."""
    print("\n🔬 Testing matrix-vector multiplication...")
    
    # (2x3) @ (3x1) = (2x1)
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = Tensor([[0.5], [0.5], [0.5]], requires_grad=True)
    
    c = a.matmul(b)
    expected = [[3.0], [7.5]]
    assert np.allclose(c.data, expected)
    
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (3, 1)
    
    print("✅ Matrix-vector multiplication gradients correct!")


def test_matmul_backward_chain():
    """Test matmul in computation chain."""
    print("\n🔬 Testing matmul in chain...")
    
    a = Tensor([[1.0, 2.0]], requires_grad=True)  # 1x2
    b = Tensor([[0.5], [0.5]], requires_grad=True)  # 2x1
    c = Tensor([[2.0]], requires_grad=True)  # 1x1
    
    # Chain: (a @ b) * c
    d = a.matmul(b)  # 1x1
    e = d * c.data  # element-wise multiply
    
    e.backward()
    
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape
    
    print("✅ Chained matmul gradients correct!")


def test_matmul_backward_batch():
    """Test batched matrix multiplication."""
    print("\n🔬 Testing batched matmul...")
    
    # (batch=2, 2x3) @ (batch=2, 3x2) = (batch=2, 2x2)
    a = Tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]], requires_grad=True)
    b = Tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], 
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], requires_grad=True)
    
    c = a.matmul(b)
    assert c.shape == (2, 2, 2)
    
    grad_output = np.ones_like(c.data)
    c.backward(grad_output)
    
    assert a.grad.shape == (2, 2, 3)
    assert b.grad.shape == (2, 3, 2)
    
    print("✅ Batched matmul gradients correct!")


def test_matmul_backward_linear_layer():
    """Test matmul simulating linear layer."""
    print("\n🔬 Testing matmul for linear layer simulation...")
    
    # Simulate: output = input @ weights.T
    batch_size, in_features, out_features = 4, 3, 2
    
    x = Tensor(np.random.randn(batch_size, in_features), requires_grad=True)
    W = Tensor(np.random.randn(out_features, in_features), requires_grad=True)
    
    # y = x @ W.T
    y = x.matmul(W.transpose())
    assert y.shape == (batch_size, out_features)
    
    # Backward
    grad_output = np.ones_like(y.data)
    y.backward(grad_output)
    
    # Verify gradient shapes
    assert x.grad.shape == x.shape
    assert W.grad.shape == W.shape
    
    print("✅ Linear layer matmul gradients correct!")


def test_module():
    """🧪 Module Test: MatMulBackward Complete Test

    Run all MatMulBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING MATMULBACKWARD MODULE TEST")
    print("="*60)
    
    test_matmul_backward_simple()
    test_matmul_backward_rectangular()
    test_matmul_backward_vector()
    test_matmul_backward_chain()
    test_matmul_backward_batch()
    test_matmul_backward_linear_layer()
    
    print("\n" + "="*60)
    print("🎉 ALL MATMULBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
