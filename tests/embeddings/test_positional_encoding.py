"""
Test suite for the PositionalEncoding class.

Tests cover:
1. Initialization and shape validation
2. Forward pass with various sequence lengths
3. Batch handling
4. Variable sequence length handling
5. Error handling (dimension mismatch, too long sequences)
6. Parameter management
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.embeddings import PositionalEncoding


class TestPositionalEncodingInitialization:
    """Test PositionalEncoding initialization."""
    
    def test_init_shape(self, small_seq_len, small_embed_dim):
        """Test that position embeddings have correct shape."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        assert pos_enc.position_embeddings.shape == (small_seq_len, small_embed_dim)
    
    def test_init_attributes(self, max_seq_len, embed_dim):
        """Test that attributes are set correctly."""
        pos_enc = PositionalEncoding(max_seq_len, embed_dim)
        assert pos_enc.max_seq_len == max_seq_len
        assert pos_enc.embed_dim == embed_dim
    
    def test_init_weights_not_zero(self, small_seq_len, small_embed_dim):
        """Test that position embeddings are initialized with non-zero values."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        assert not np.allclose(pos_enc.position_embeddings.data, 0.0)
    
    def test_init_weights_range(self, small_seq_len, small_embed_dim):
        """Test that initialized weights are within reasonable bounds."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        # Limit = sqrt(2 / embed_dim)
        limit = np.sqrt(2.0 / small_embed_dim)
        assert np.all(pos_enc.position_embeddings.data >= -limit - 0.01)
        assert np.all(pos_enc.position_embeddings.data <= limit + 0.01)


class TestPositionalEncodingForward:
    """Test PositionalEncoding forward pass."""
    
    def test_forward_basic(self, small_seq_len, small_embed_dim, batch_size):
        """Test basic forward pass."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(batch_size, small_seq_len, small_embed_dim))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (batch_size, small_seq_len, small_embed_dim)
    
    def test_forward_adds_position_info(self, small_seq_len, small_embed_dim, batch_size):
        """Test that forward pass adds positional information."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.zeros((batch_size, small_seq_len, small_embed_dim)))
        
        output = pos_enc.forward(x)
        
        # Output should not be all zeros (position info added)
        assert not np.allclose(output.data, 0.0)
        # Each position in sequence should have same positional encoding across batch
        np.testing.assert_array_almost_equal(output.data[0], output.data[1])
    
    def test_forward_consistent_positions(self, small_seq_len, small_embed_dim):
        """Test that same positions get same encodings across different batches."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        
        x1 = Tensor(np.random.randn(2, small_seq_len, small_embed_dim))
        x2 = Tensor(np.random.randn(3, small_seq_len, small_embed_dim))
        
        output1 = pos_enc.forward(x1)
        output2 = pos_enc.forward(x2)
        
        # Position encodings should be the same (subtract input to get pos encoding)
        pos_encoding_1 = output1.data[0] - x1.data[0]
        pos_encoding_2 = output2.data[0] - x2.data[0]
        np.testing.assert_array_almost_equal(pos_encoding_1, pos_encoding_2)
    
    def test_forward_variable_seq_len(self, small_seq_len, small_embed_dim, batch_size):
        """Test forward pass with shorter sequence than max."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        short_seq_len = small_seq_len // 2
        x = Tensor(np.random.randn(batch_size, short_seq_len, small_embed_dim))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (batch_size, short_seq_len, small_embed_dim)
    
    def test_forward_single_position(self, small_seq_len, small_embed_dim):
        """Test forward pass with sequence length of 1."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(2, 1, small_embed_dim))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (2, 1, small_embed_dim)
    
    def test_forward_max_seq_len(self, small_seq_len, small_embed_dim, batch_size):
        """Test forward pass with maximum sequence length."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(batch_size, small_seq_len, small_embed_dim))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (batch_size, small_seq_len, small_embed_dim)
    
    def test_call_method(self, small_seq_len, small_embed_dim, batch_size):
        """Test __call__ method works like forward."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(batch_size, small_seq_len, small_embed_dim))
        
        output_forward = pos_enc.forward(x)
        output_call = pos_enc(x)
        
        np.testing.assert_array_equal(output_forward.data, output_call.data)


class TestPositionalEncodingBroadcasting:
    """Test PositionalEncoding broadcasting behavior."""
    
    def test_same_encoding_across_batch(self, small_seq_len, small_embed_dim):
        """Test that positional encoding is the same for all items in batch."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        batch_size = 4
        
        # Use zeros to easily see the positional encoding
        x = Tensor(np.zeros((batch_size, small_seq_len, small_embed_dim)))
        output = pos_enc.forward(x)
        
        # All batch items should have identical positional encodings
        for i in range(1, batch_size):
            np.testing.assert_array_almost_equal(output.data[0], output.data[i])
    
    def test_different_encoding_across_positions(self, small_seq_len, small_embed_dim):
        """Test that different positions get different encodings."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.zeros((1, small_seq_len, small_embed_dim)))
        
        output = pos_enc.forward(x)
        
        # Different positions should have different encodings
        for i in range(small_seq_len - 1):
            assert not np.allclose(output.data[0, i], output.data[0, i + 1])


class TestPositionalEncodingErrors:
    """Test PositionalEncoding error handling."""
    
    def test_seq_len_too_long(self, small_seq_len, small_embed_dim, batch_size):
        """Test that sequence longer than max_seq_len raises error."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(batch_size, small_seq_len + 10, small_embed_dim))
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            pos_enc.forward(x)
    
    def test_wrong_input_dimensions(self, small_seq_len, small_embed_dim):
        """Test that 2D input raises error."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(small_seq_len, small_embed_dim))
        
        with pytest.raises(ValueError, match="Expected 3D input"):
            pos_enc.forward(x)
    
    def test_embed_dim_mismatch(self, small_seq_len, small_embed_dim, batch_size):
        """Test that wrong embedding dimension raises error."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(batch_size, small_seq_len, small_embed_dim + 5))
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            pos_enc.forward(x)
    
    def test_4d_input_raises_error(self, small_seq_len, small_embed_dim):
        """Test that 4D input raises error."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        x = Tensor(np.random.randn(2, 2, small_seq_len, small_embed_dim))
        
        with pytest.raises(ValueError, match="Expected 3D input"):
            pos_enc.forward(x)


class TestPositionalEncodingParameters:
    """Test PositionalEncoding parameter management."""
    
    def test_parameters_returns_embeddings(self, small_seq_len, small_embed_dim):
        """Test that parameters() returns the position embeddings."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        
        params = pos_enc.parameters()
        
        assert len(params) == 1
        assert params[0] is pos_enc.position_embeddings
    
    def test_parameters_modifiable(self, small_seq_len, small_embed_dim):
        """Test that parameters can be modified."""
        pos_enc = PositionalEncoding(small_seq_len, small_embed_dim)
        original_embeddings = pos_enc.position_embeddings.data.copy()
        
        # Modify the embeddings
        pos_enc.position_embeddings.data[:] = 0.5
        
        assert not np.allclose(pos_enc.position_embeddings.data, original_embeddings)
        assert np.allclose(pos_enc.position_embeddings.data, 0.5)


class TestPositionalEncodingRepr:
    """Test PositionalEncoding string representation."""
    
    def test_repr(self, max_seq_len, embed_dim):
        """Test __repr__ method."""
        pos_enc = PositionalEncoding(max_seq_len, embed_dim)
        
        repr_str = repr(pos_enc)
        
        assert "PositionalEncoding" in repr_str
        assert str(max_seq_len) in repr_str
        assert str(embed_dim) in repr_str


class TestPositionalEncodingDifferentSizes:
    """Test PositionalEncoding with various sizes."""
    
    def test_small_dimensions(self):
        """Test with very small dimensions."""
        pos_enc = PositionalEncoding(max_seq_len=4, embed_dim=4)
        x = Tensor(np.random.randn(2, 4, 4))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (2, 4, 4)
    
    def test_large_dimensions(self):
        """Test with larger dimensions."""
        pos_enc = PositionalEncoding(max_seq_len=1024, embed_dim=512)
        x = Tensor(np.random.randn(2, 100, 512))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (2, 100, 512)
    
    def test_odd_embed_dim(self):
        """Test with odd embedding dimension."""
        pos_enc = PositionalEncoding(max_seq_len=10, embed_dim=7)
        x = Tensor(np.random.randn(2, 5, 7))
        
        output = pos_enc.forward(x)
        
        assert output.shape == (2, 5, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
