"""
Test suite for the EmbeddingLayer class.

Tests cover:
1. Initialization with different positional encoding types
2. Forward pass with 1D and 2D inputs
3. Learned positional encoding
4. Sinusoidal positional encoding
5. No positional encoding
6. Embedding scaling
7. Parameter management
8. Edge cases and error handling
"""
import numpy as np
import pytest
import math
from core.tensor import Tensor
from core.embeddings import EmbeddingLayer


class TestEmbeddingLayerInitialization:
    """Test EmbeddingLayer initialization."""
    
    def test_init_learned_pos_encoding(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test initialization with learned positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        
        assert layer.vocab_size == small_vocab_size
        assert layer.embed_dim == small_embed_dim
        assert layer.max_seq_len == small_seq_len
        assert layer.pos_encoding_type == 'learned'
        assert layer.pos_encoding is not None
    
    def test_init_sinusoidal_pos_encoding(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test initialization with sinusoidal positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='sinusoidal'
        )
        
        assert layer.pos_encoding_type == 'sinusoidal'
        assert layer.pos_encoding is not None
        assert layer.pos_encoding.shape == (small_seq_len, small_embed_dim)
    
    def test_init_no_pos_encoding(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test initialization without positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding=None
        )
        
        assert layer.pos_encoding_type is None
        assert layer.pos_encoding is None
    
    def test_init_with_scaling(self, small_vocab_size, small_embed_dim):
        """Test initialization with embedding scaling enabled."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            scale_embeddings=True
        )
        
        assert layer.scale_embeddings is True
    
    def test_init_without_scaling(self, small_vocab_size, small_embed_dim):
        """Test initialization with embedding scaling disabled."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            scale_embeddings=False
        )
        
        assert layer.scale_embeddings is False
    
    def test_init_invalid_pos_encoding(self, small_vocab_size, small_embed_dim):
        """Test that invalid pos_encoding type raises error."""
        with pytest.raises(ValueError, match="Unknown pos_encoding"):
            EmbeddingLayer(
                small_vocab_size,
                small_embed_dim,
                pos_encoding='invalid'
            )


class TestEmbeddingLayerForwardLearned:
    """Test EmbeddingLayer forward pass with learned positional encoding."""
    
    def test_forward_1d_input(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test forward pass with 1D input."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        tokens = Tensor(np.array([1, 2, 3, 4]))
        
        output = layer.forward(tokens)
        
        # 1D input should be squeezed back to (seq_len, embed_dim)
        assert output.shape == (4, small_embed_dim)
    
    def test_forward_2d_input(self, small_vocab_size, small_embed_dim, small_seq_len, batch_size):
        """Test forward pass with 2D batch input."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        tokens = Tensor(np.random.randint(0, small_vocab_size, (batch_size, 8)))
        
        output = layer.forward(tokens)
        
        assert output.shape == (batch_size, 8, small_embed_dim)
    
    def test_forward_adds_positional_info(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test that positional encoding is added."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        tokens = Tensor(np.array([[0, 0, 0]]))  # Same token repeated
        
        output = layer.forward(tokens)
        
        # Even though tokens are the same, positions should differ
        assert not np.allclose(output.data[0, 0], output.data[0, 1])
        assert not np.allclose(output.data[0, 1], output.data[0, 2])
    
    def test_call_method(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test __call__ method works like forward."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        tokens = Tensor(np.array([1, 2, 3]))
        
        output_forward = layer.forward(tokens)
        output_call = layer(tokens)
        
        np.testing.assert_array_equal(output_forward.data, output_call.data)


class TestEmbeddingLayerForwardSinusoidal:
    """Test EmbeddingLayer forward pass with sinusoidal positional encoding."""
    
    def test_forward_1d_input(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test forward pass with 1D input and sinusoidal encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='sinusoidal'
        )
        tokens = Tensor(np.array([1, 2, 3]))
        
        output = layer.forward(tokens)
        
        assert output.shape == (3, small_embed_dim)
    
    def test_forward_2d_input(self, small_vocab_size, small_embed_dim, small_seq_len, batch_size):
        """Test forward pass with 2D batch input and sinusoidal encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='sinusoidal'
        )
        tokens = Tensor(np.random.randint(0, small_vocab_size, (batch_size, 8)))
        
        output = layer.forward(tokens)
        
        assert output.shape == (batch_size, 8, small_embed_dim)
    
    def test_sinusoidal_consistent_across_batches(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test that sinusoidal encoding is consistent across different batches."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='sinusoidal'
        )
        
        # Use same tokens in different batch sizes
        tokens1 = Tensor(np.array([[1, 2, 3]]))
        tokens2 = Tensor(np.array([[1, 2, 3], [1, 2, 3]]))
        
        output1 = layer.forward(tokens1)
        output2 = layer.forward(tokens2)
        
        # First batch item should be identical
        np.testing.assert_array_almost_equal(output1.data[0], output2.data[0])


class TestEmbeddingLayerForwardNoPosition:
    """Test EmbeddingLayer forward pass without positional encoding."""
    
    def test_forward_1d_input_no_pos(self, small_vocab_size, small_embed_dim):
        """Test forward pass with 1D input and no positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None
        )
        tokens = Tensor(np.array([1, 2, 3]))
        
        output = layer.forward(tokens)
        
        assert output.shape == (3, small_embed_dim)
    
    def test_forward_2d_input_no_pos(self, small_vocab_size, small_embed_dim, batch_size):
        """Test forward pass with 2D input and no positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None
        )
        tokens = Tensor(np.random.randint(0, small_vocab_size, (batch_size, 8)))
        
        output = layer.forward(tokens)
        
        assert output.shape == (batch_size, 8, small_embed_dim)
    
    def test_no_positional_difference(self, small_vocab_size, small_embed_dim):
        """Test that same tokens at different positions give same embeddings."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None
        )
        tokens = Tensor(np.array([[2, 2, 2]]))
        
        output = layer.forward(tokens)
        
        # Without positional encoding, same tokens should have identical embeddings
        np.testing.assert_array_almost_equal(output.data[0, 0], output.data[0, 1])
        np.testing.assert_array_almost_equal(output.data[0, 1], output.data[0, 2])


class TestEmbeddingLayerScaling:
    """Test EmbeddingLayer embedding scaling."""
    
    def test_scaling_enabled(self, small_vocab_size, small_embed_dim):
        """Test that scaling is applied when enabled."""
        layer_no_scale = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None,
            scale_embeddings=False
        )
        layer_with_scale = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None,
            scale_embeddings=True
        )
        
        # Set same weights
        layer_with_scale.token_embedding.weight.data[:] = layer_no_scale.token_embedding.weight.data
        
        tokens = Tensor(np.array([1, 2, 3]))
        
        output_no_scale = layer_no_scale.forward(tokens)
        output_with_scale = layer_with_scale.forward(tokens)
        
        # Scaled output should be sqrt(embed_dim) times larger
        scale_factor = math.sqrt(small_embed_dim)
        expected_scaled = output_no_scale.data * scale_factor
        
        np.testing.assert_array_almost_equal(output_with_scale.data, expected_scaled)
    
    def test_scaling_magnitude(self, small_vocab_size, small_embed_dim):
        """Test that scaling increases magnitude by sqrt(embed_dim)."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None,
            scale_embeddings=True
        )
        
        # Set known weights
        layer.token_embedding.weight.data[:] = 1.0
        
        tokens = Tensor(np.array([0]))
        output = layer.forward(tokens)
        
        # Each value should be approximately sqrt(embed_dim)
        expected = math.sqrt(small_embed_dim)
        np.testing.assert_array_almost_equal(output.data, expected)


class TestEmbeddingLayerParameters:
    """Test EmbeddingLayer parameter management."""
    
    def test_parameters_learned_pos(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test parameters with learned positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        
        params = layer.parameters()
        
        # Should have token embedding weight and positional embedding
        assert len(params) == 2
        assert params[0] is layer.token_embedding.weight
        assert params[1] is layer.pos_encoding.position_embeddings
    
    def test_parameters_sinusoidal_pos(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test parameters with sinusoidal positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='sinusoidal'
        )
        
        params = layer.parameters()
        
        # Should only have token embedding weight (sinusoidal is fixed)
        assert len(params) == 1
        assert params[0] is layer.token_embedding.weight
    
    def test_parameters_no_pos(self, small_vocab_size, small_embed_dim):
        """Test parameters without positional encoding."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None
        )
        
        params = layer.parameters()
        
        # Should only have token embedding weight
        assert len(params) == 1
        assert params[0] is layer.token_embedding.weight


class TestEmbeddingLayerRepr:
    """Test EmbeddingLayer string representation."""
    
    def test_repr_learned(self, vocab_size, embed_dim):
        """Test __repr__ with learned positional encoding."""
        layer = EmbeddingLayer(vocab_size, embed_dim, pos_encoding='learned')
        
        repr_str = repr(layer)
        
        assert "EmbeddingLayer" in repr_str
        assert str(vocab_size) in repr_str
        assert str(embed_dim) in repr_str
        assert "learned" in repr_str
    
    def test_repr_sinusoidal(self, vocab_size, embed_dim):
        """Test __repr__ with sinusoidal positional encoding."""
        layer = EmbeddingLayer(vocab_size, embed_dim, pos_encoding='sinusoidal')
        
        repr_str = repr(layer)
        
        assert "EmbeddingLayer" in repr_str
        assert "sinusoidal" in repr_str
    
    def test_repr_none(self, vocab_size, embed_dim):
        """Test __repr__ without positional encoding."""
        layer = EmbeddingLayer(vocab_size, embed_dim, pos_encoding=None)
        
        repr_str = repr(layer)
        
        assert "EmbeddingLayer" in repr_str
        assert "None" in repr_str


class TestEmbeddingLayerVariableLength:
    """Test EmbeddingLayer with variable sequence lengths."""
    
    def test_short_sequence(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test with sequence shorter than max_seq_len."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        short_seq = small_seq_len // 2
        tokens = Tensor(np.random.randint(0, small_vocab_size, (2, short_seq)))
        
        output = layer.forward(tokens)
        
        assert output.shape == (2, short_seq, small_embed_dim)
    
    def test_single_token(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test with single token."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        tokens = Tensor(np.array([5]))
        
        output = layer.forward(tokens)
        
        assert output.shape == (1, small_embed_dim)
    
    def test_max_length_sequence(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test with sequence at max_seq_len."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        tokens = Tensor(np.random.randint(0, small_vocab_size, (2, small_seq_len)))
        
        output = layer.forward(tokens)
        
        assert output.shape == (2, small_seq_len, small_embed_dim)


class TestEmbeddingLayerIntegration:
    """Integration tests for EmbeddingLayer."""
    
    def test_learned_vs_sinusoidal_difference(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test that learned and sinusoidal encodings produce different results."""
        layer_learned = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        layer_sinusoidal = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='sinusoidal'
        )
        
        # Set same token embeddings
        layer_sinusoidal.token_embedding.weight.data[:] = layer_learned.token_embedding.weight.data
        
        tokens = Tensor(np.array([[1, 2, 3]]))
        
        output_learned = layer_learned.forward(tokens)
        output_sinusoidal = layer_sinusoidal.forward(tokens)
        
        # Should be different due to different positional encodings
        assert not np.allclose(output_learned.data, output_sinusoidal.data)
    
    def test_consistent_token_embeddings(self, small_vocab_size, small_embed_dim):
        """Test that token embeddings are looked up consistently."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            pos_encoding=None
        )
        
        # Same token should give same embedding
        tokens1 = Tensor(np.array([[3]]))
        tokens2 = Tensor(np.array([[3]]))
        
        output1 = layer.forward(tokens1)
        output2 = layer.forward(tokens2)
        
        np.testing.assert_array_equal(output1.data, output2.data)
    
    def test_batch_independence(self, small_vocab_size, small_embed_dim, small_seq_len):
        """Test that batch items are processed independently."""
        layer = EmbeddingLayer(
            small_vocab_size,
            small_embed_dim,
            max_seq_len=small_seq_len,
            pos_encoding='learned'
        )
        
        # Process separately
        tokens1 = Tensor(np.array([[1, 2, 3]]))
        tokens2 = Tensor(np.array([[1, 2, 3]]))
        
        output1 = layer.forward(tokens1)
        output2 = layer.forward(tokens2)
        
        # Should get same results
        np.testing.assert_array_almost_equal(output1.data, output2.data)
        
        # Process as batch
        tokens_batch = Tensor(np.array([[1, 2, 3], [1, 2, 3]]))
        output_batch = layer.forward(tokens_batch)
        
        # Each batch item should match individual processing
        np.testing.assert_array_almost_equal(output_batch.data[0], output1.data[0])
        np.testing.assert_array_almost_equal(output_batch.data[1], output2.data[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
