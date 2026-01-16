"""🧪 Test Suite: EmbeddingBackward

Tests gradient computation for embedding lookup operation.

Mathematical Rule: If Y = Embedding[indices], then:
∂Loss/∂Embedding[i] = sum of all gradients where index==i
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.autograd import EmbeddingBackward, enable_autograd


def test_embedding_backward_simple():
    """Test basic embedding gradient."""
    print("\n🔬 Testing simple embedding gradients...")
    
    # Embedding table: 5 words, 3 dimensions
    vocab_size, embed_dim = 5, 3
    weight = Tensor(np.random.randn(vocab_size, embed_dim), requires_grad=True)
    
    # Lookup indices: [0, 2, 1]
    indices = Tensor(np.array([0, 2, 1]))
    
    # Forward: lookup
    output_data = weight.data[indices.data.astype(int)]
    output = Tensor(output_data)
    output.requires_grad = True
    output._grad_fn = EmbeddingBackward(weight, indices)
    
    assert output.shape == (3, embed_dim)
    
    # Backward
    grad_output = np.ones_like(output.data)
    output.backward(grad_output)
    
    # Only accessed embeddings should have gradients
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape
    
    # Check that accessed indices have non-zero gradients
    assert not np.allclose(weight.grad[0], 0.0)
    assert not np.allclose(weight.grad[1], 0.0)
    assert not np.allclose(weight.grad[2], 0.0)
    
    print("✅ Simple embedding gradients correct!")


def test_embedding_backward_repeated_indices():
    """Test embedding with repeated indices (gradient accumulation)."""
    print("\n🔬 Testing repeated indices gradient accumulation...")
    
    vocab_size, embed_dim = 4, 2
    weight = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    
    # Indices with repetition: index 1 appears twice
    indices = Tensor(np.array([1, 1, 2]))
    
    output_data = weight.data[indices.data.astype(int)]
    output = Tensor(output_data)
    output.requires_grad = True
    output._grad_fn = EmbeddingBackward(weight, indices)
    
    grad_output = np.ones((3, embed_dim))
    output.backward(grad_output)
    
    # Index 1 should have accumulated gradient (appears twice)
    # Expected: weight.grad[1] = 2.0 (accumulated from two lookups)
    assert np.allclose(weight.grad[1], [2.0, 2.0])
    
    # Index 2 appears once
    assert np.allclose(weight.grad[2], [1.0, 1.0])
    
    print("✅ Repeated indices gradient accumulation correct!")


def test_embedding_backward_batch():
    """Test embedding with batch of sequences."""
    print("\n🔬 Testing batch embedding...")
    
    vocab_size, embed_dim = 10, 4
    batch_size, seq_len = 2, 3
    
    weight = Tensor(np.random.randn(vocab_size, embed_dim), requires_grad=True)
    
    # Batch of sequences: (batch_size, seq_len)
    indices = Tensor(np.array([[1, 2, 3], [4, 5, 1]]))
    
    # Lookup: (batch_size, seq_len, embed_dim)
    indices_flat = indices.data.astype(int).flatten()
    output_data = weight.data[indices_flat].reshape(batch_size, seq_len, embed_dim)
    output = Tensor(output_data)
    output.requires_grad = True
    output._grad_fn = EmbeddingBackward(weight, indices)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    
    grad_output = np.ones_like(output.data)
    output.backward(grad_output)
    
    # Index 1 appears twice (in different sequences)
    assert np.allclose(weight.grad[1], [2.0, 2.0, 2.0, 2.0])
    
    print("✅ Batch embedding gradients correct!")


def test_embedding_backward_weighted():
    """Test embedding with weighted gradients."""
    print("\n🔬 Testing weighted embedding gradients...")
    
    vocab_size, embed_dim = 3, 2
    weight = Tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    
    indices = Tensor(np.array([0, 1, 2]))
    
    output_data = weight.data[indices.data.astype(int)]
    output = Tensor(output_data)
    output.requires_grad = True
    output._grad_fn = EmbeddingBackward(weight, indices)
    
    # Different gradients for each position
    grad_output = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    output.backward(grad_output)
    
    # Each embedding should receive its corresponding gradient
    assert np.allclose(weight.grad[0], [1.0, 1.0])
    assert np.allclose(weight.grad[1], [2.0, 2.0])
    assert np.allclose(weight.grad[2], [3.0, 3.0])
    
    print("✅ Weighted embedding gradients correct!")


def test_embedding_backward_sparse():
    """Test that unused embeddings get zero gradient."""
    print("\n🔬 Testing sparse embedding gradients...")
    
    vocab_size, embed_dim = 10, 3
    weight = Tensor(np.random.randn(vocab_size, embed_dim), requires_grad=True)
    
    # Only use indices 0, 2, 5 (others should have zero gradient)
    indices = Tensor(np.array([0, 2, 5]))
    
    output_data = weight.data[indices.data.astype(int)]
    output = Tensor(output_data)
    output.requires_grad = True
    output._grad_fn = EmbeddingBackward(weight, indices)
    
    grad_output = np.ones_like(output.data)
    output.backward(grad_output)
    
    # Used indices should have non-zero gradients
    assert not np.allclose(weight.grad[0], 0.0)
    assert not np.allclose(weight.grad[2], 0.0)
    assert not np.allclose(weight.grad[5], 0.0)
    
    # Unused indices should have zero gradients
    for i in [1, 3, 4, 6, 7, 8, 9]:
        assert np.allclose(weight.grad[i], 0.0)
    
    print("✅ Sparse embedding gradients correct!")


def test_module():
    """🧪 Module Test: EmbeddingBackward Complete Test

    Run all EmbeddingBackward tests to ensure gradient computation is correct.
    """
    print("\n" + "="*60)
    print("🧪 RUNNING EMBEDDINGBACKWARD MODULE TEST")
    print("="*60)
    
    test_embedding_backward_simple()
    test_embedding_backward_repeated_indices()
    test_embedding_backward_batch()
    test_embedding_backward_weighted()
    test_embedding_backward_sparse()
    
    print("\n" + "="*60)
    print("🎉 ALL EMBEDDINGBACKWARD TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_module()
