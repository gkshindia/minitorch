"""
Test suite for the Embedding class.

Tests cover:
1. Initialization and shape validation
2. Forward pass with various input shapes
3. Lookup functionality
4. Edge cases (boundary indices, empty inputs)
5. Error handling (out-of-range indices)
6. Parameter management
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.embeddings import Embedding


class TestEmbeddingInitialization:
    """Test Embedding initialization."""
    
    def test_init_shape(self, small_vocab_size, small_embed_dim):
        """Test that embedding weight has correct shape."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        assert emb.weight.shape == (small_vocab_size, small_embed_dim)
    
    def test_init_attributes(self, vocab_size, embed_dim):
        """Test that attributes are set correctly."""
        emb = Embedding(vocab_size, embed_dim)
        assert emb.vocab_size == vocab_size
        assert emb.embed_dim == embed_dim
    
    def test_init_weights_not_zero(self, small_vocab_size, small_embed_dim):
        """Test that weights are initialized with non-zero values."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        assert not np.allclose(emb.weight.data, 0.0)
    
    def test_init_weights_range(self, small_vocab_size, small_embed_dim):
        """Test that initialized weights are within reasonable bounds."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        # Xavier/Glorot uniform: limit = sqrt(6 / (vocab + embed))
        limit = np.sqrt(6.0 / (small_vocab_size + small_embed_dim))
        assert np.all(emb.weight.data >= -limit - 0.01)  # Small epsilon for numerical stability
        assert np.all(emb.weight.data <= limit + 0.01)


class TestEmbeddingForward:
    """Test Embedding forward pass."""
    
    def test_forward_single_index(self, small_vocab_size, small_embed_dim):
        """Test forward pass with a single index."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([3]))
        
        output = emb.forward(indices)
        
        assert output.shape == (1, small_embed_dim)
        np.testing.assert_array_equal(output.data, emb.weight.data[3:4])
    
    def test_forward_multiple_indices(self, small_vocab_size, small_embed_dim):
        """Test forward pass with multiple indices."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([0, 2, 5, 1]))
        
        output = emb.forward(indices)
        
        assert output.shape == (4, small_embed_dim)
        np.testing.assert_array_equal(output.data[0], emb.weight.data[0])
        np.testing.assert_array_equal(output.data[1], emb.weight.data[2])
        np.testing.assert_array_equal(output.data[2], emb.weight.data[5])
        np.testing.assert_array_equal(output.data[3], emb.weight.data[1])
    
    def test_forward_batch_2d(self, small_vocab_size, small_embed_dim):
        """Test forward pass with 2D batch of indices."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        batch_size, seq_len = 2, 3
        indices = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        
        output = emb.forward(indices)
        
        assert output.shape == (batch_size, seq_len, small_embed_dim)
        np.testing.assert_array_equal(output.data[0, 0], emb.weight.data[1])
        np.testing.assert_array_equal(output.data[1, 2], emb.weight.data[6])
    
    def test_forward_repeated_indices(self, small_vocab_size, small_embed_dim):
        """Test forward pass with repeated indices."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([2, 2, 2]))
        
        output = emb.forward(indices)
        
        assert output.shape == (3, small_embed_dim)
        # All outputs should be identical
        np.testing.assert_array_equal(output.data[0], output.data[1])
        np.testing.assert_array_equal(output.data[1], output.data[2])
        np.testing.assert_array_equal(output.data[0], emb.weight.data[2])
    
    def test_forward_boundary_indices(self, small_vocab_size, small_embed_dim):
        """Test forward pass with boundary indices (0 and vocab_size-1)."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([0, small_vocab_size - 1]))
        
        output = emb.forward(indices)
        
        assert output.shape == (2, small_embed_dim)
        np.testing.assert_array_equal(output.data[0], emb.weight.data[0])
        np.testing.assert_array_equal(output.data[1], emb.weight.data[small_vocab_size - 1])
    
    def test_call_method(self, small_vocab_size, small_embed_dim):
        """Test __call__ method works like forward."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([1, 3, 5]))
        
        output_forward = emb.forward(indices)
        output_call = emb(indices)
        
        np.testing.assert_array_equal(output_forward.data, output_call.data)


class TestEmbeddingEdgeCases:
    """Test Embedding edge cases."""
    
    def test_zero_index(self, small_vocab_size, small_embed_dim):
        """Test that index 0 works correctly."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([0]))
        
        output = emb.forward(indices)
        
        assert output.shape == (1, small_embed_dim)
        np.testing.assert_array_equal(output.data[0], emb.weight.data[0])
    
    def test_float_indices_converted_to_int(self, small_vocab_size, small_embed_dim):
        """Test that float indices are converted to integers."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([1.9, 2.1, 3.8]))
        
        output = emb.forward(indices)
        
        # Should truncate to [1, 2, 3]
        assert output.shape == (3, small_embed_dim)
        np.testing.assert_array_equal(output.data[0], emb.weight.data[1])
        np.testing.assert_array_equal(output.data[1], emb.weight.data[2])
        np.testing.assert_array_equal(output.data[2], emb.weight.data[3])


class TestEmbeddingErrors:
    """Test Embedding error handling."""
    
    def test_out_of_range_high(self, small_vocab_size, small_embed_dim):
        """Test that indices >= vocab_size raise error."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([small_vocab_size]))
        
        with pytest.raises(ValueError, match="Index out of range"):
            emb.forward(indices)
    
    def test_out_of_range_negative(self, small_vocab_size, small_embed_dim):
        """Test that negative indices raise error."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([-1]))
        
        with pytest.raises(ValueError, match="Index out of range"):
            emb.forward(indices)
    
    def test_out_of_range_in_batch(self, small_vocab_size, small_embed_dim):
        """Test that out-of-range index in batch raises error."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        indices = Tensor(np.array([1, 2, small_vocab_size + 5]))
        
        with pytest.raises(ValueError, match="Index out of range"):
            emb.forward(indices)


class TestEmbeddingParameters:
    """Test Embedding parameter management."""
    
    def test_parameters_returns_weight(self, small_vocab_size, small_embed_dim):
        """Test that parameters() returns the weight tensor."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        
        params = emb.parameters()
        
        assert len(params) == 1
        assert params[0] is emb.weight
    
    def test_parameters_modifiable(self, small_vocab_size, small_embed_dim):
        """Test that parameters can be modified."""
        emb = Embedding(small_vocab_size, small_embed_dim)
        original_weight = emb.weight.data.copy()
        
        # Modify the weight
        emb.weight.data[:] = 0.5
        
        assert not np.allclose(emb.weight.data, original_weight)
        assert np.allclose(emb.weight.data, 0.5)


class TestEmbeddingRepr:
    """Test Embedding string representation."""
    
    def test_repr(self, vocab_size, embed_dim):
        """Test __repr__ method."""
        emb = Embedding(vocab_size, embed_dim)
        
        repr_str = repr(emb)
        
        assert "Embedding" in repr_str
        assert str(vocab_size) in repr_str
        assert str(embed_dim) in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
