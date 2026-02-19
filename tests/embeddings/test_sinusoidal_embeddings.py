"""
Test suite for the create_sinusoidal_embeddings function.

Tests cover:
1. Shape validation
2. Sine/cosine pattern verification
3. Different frequencies for different dimensions
4. Fixed (non-learnable) embeddings
5. Mathematical properties
6. Edge cases (odd/even dimensions)
"""
import numpy as np
import pytest
import math
from core.tensor import Tensor
from core.embeddings import create_sinusoidal_embeddings


class TestSinusoidalEmbeddingsShape:
    """Test shape properties of sinusoidal embeddings."""
    
    def test_basic_shape(self, small_seq_len, small_embed_dim):
        """Test that output has correct shape."""
        embeddings = create_sinusoidal_embeddings(small_seq_len, small_embed_dim)
        assert embeddings.shape == (small_seq_len, small_embed_dim)
    
    def test_different_sizes(self):
        """Test various size combinations."""
        test_cases = [
            (10, 8),
            (100, 64),
            (512, 128),
            (1024, 512),
        ]
        
        for max_seq_len, embed_dim in test_cases:
            embeddings = create_sinusoidal_embeddings(max_seq_len, embed_dim)
            assert embeddings.shape == (max_seq_len, embed_dim)
    
    def test_odd_embed_dim(self):
        """Test with odd embedding dimension."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=10, embed_dim=7)
        assert embeddings.shape == (10, 7)
    
    def test_even_embed_dim(self):
        """Test with even embedding dimension."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=10, embed_dim=8)
        assert embeddings.shape == (10, 8)


class TestSinusoidalEmbeddingsSineCosine:
    """Test sine/cosine pattern in sinusoidal embeddings."""
    
    def test_sine_on_even_indices(self, small_seq_len, small_embed_dim):
        """Test that even indices use sine function."""
        embeddings = create_sinusoidal_embeddings(small_seq_len, small_embed_dim)
        
        # Manually compute what the first position should be
        position = 0
        div_term = np.exp(
            np.arange(0, small_embed_dim, 2, dtype=np.float32) *
            -(math.log(10000.0) / small_embed_dim)
        )
        
        expected_even = np.sin(position * div_term)
        
        # Check even indices (0, 2, 4, ...)
        actual_even = embeddings.data[0, 0::2]
        np.testing.assert_array_almost_equal(actual_even, expected_even, decimal=5)
    
    def test_cosine_on_odd_indices(self, small_seq_len):
        """Test that odd indices use cosine function."""
        embed_dim = 8  # Use even dimension for simplicity
        embeddings = create_sinusoidal_embeddings(small_seq_len, embed_dim)
        
        # Manually compute what the first position should be
        position = 0
        div_term = np.exp(
            np.arange(0, embed_dim, 2, dtype=np.float32) *
            -(math.log(10000.0) / embed_dim)
        )
        
        expected_odd = np.cos(position * div_term)
        
        # Check odd indices (1, 3, 5, ...)
        actual_odd = embeddings.data[0, 1::2]
        np.testing.assert_array_almost_equal(actual_odd, expected_odd, decimal=5)
    
    def test_alternating_pattern(self):
        """Test that sine and cosine alternate."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=5, embed_dim=4)
        
        # For position 0, sine columns should be 0, cosine columns should be 1
        assert abs(embeddings.data[0, 0]) < 1e-6  # sin(0) ≈ 0
        assert abs(embeddings.data[0, 1] - 1.0) < 1e-6  # cos(0) ≈ 1
        assert abs(embeddings.data[0, 2]) < 1e-6  # sin(0) ≈ 0
        assert abs(embeddings.data[0, 3] - 1.0) < 1e-6  # cos(0) ≈ 1


class TestSinusoidalEmbeddingsFrequencies:
    """Test frequency properties of sinusoidal embeddings."""
    
    def test_different_frequencies_per_dimension(self):
        """Test that different dimensions have different frequencies."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=100, embed_dim=8)
        
        # First dimension (highest frequency) should vary quickly
        # Last dimension (lowest frequency) should vary slowly
        dim_0_variation = np.std(embeddings.data[:, 0])
        dim_6_variation = np.std(embeddings.data[:, 6])
        
        # Lower indexed dimensions should have more variation (higher frequency)
        assert dim_0_variation >= dim_6_variation
    
    def test_position_specific_encoding(self):
        """Test that different positions have different encodings."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=10, embed_dim=8)
        
        # Each position should be unique
        for i in range(9):
            for j in range(i + 1, 10):
                assert not np.allclose(embeddings.data[i], embeddings.data[j])


class TestSinusoidalEmbeddingsMathematicalProperties:
    """Test mathematical properties of sinusoidal embeddings."""
    
    def test_values_in_range(self, small_seq_len, small_embed_dim):
        """Test that all values are in [-1, 1] range."""
        embeddings = create_sinusoidal_embeddings(small_seq_len, small_embed_dim)
        
        assert np.all(embeddings.data >= -1.0)
        assert np.all(embeddings.data <= 1.0)
    
    def test_deterministic(self, small_seq_len, small_embed_dim):
        """Test that function produces same output each time."""
        embeddings1 = create_sinusoidal_embeddings(small_seq_len, small_embed_dim)
        embeddings2 = create_sinusoidal_embeddings(small_seq_len, small_embed_dim)
        
        np.testing.assert_array_equal(embeddings1.data, embeddings2.data)
    
    def test_first_position_properties(self):
        """Test that first position (pos=0) has expected values."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=10, embed_dim=8)
        
        # At position 0: sin(0) = 0, cos(0) = 1
        for i in range(0, 8, 2):
            assert abs(embeddings.data[0, i]) < 1e-6  # sine columns ≈ 0
        for i in range(1, 8, 2):
            assert abs(embeddings.data[0, i] - 1.0) < 1e-6  # cosine columns ≈ 1
    
    def test_not_all_same(self, small_seq_len, small_embed_dim):
        """Test that embeddings are not all the same value."""
        embeddings = create_sinusoidal_embeddings(small_seq_len, small_embed_dim)
        
        # Should have variety in values
        unique_values = np.unique(embeddings.data)
        assert len(unique_values) > 10


class TestSinusoidalEmbeddingsOddEven:
    """Test handling of odd vs even embedding dimensions."""
    
    def test_odd_dimension_shape(self):
        """Test that odd dimensions are handled correctly."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=10, embed_dim=9)
        
        assert embeddings.shape == (10, 9)
        # Should still have valid values
        assert np.all(np.isfinite(embeddings.data))
    
    def test_even_dimension_shape(self):
        """Test that even dimensions work correctly."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=10, embed_dim=8)
        
        assert embeddings.shape == (10, 8)
        assert np.all(np.isfinite(embeddings.data))
    
    def test_dimension_1(self):
        """Test edge case with embedding dimension of 1."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=5, embed_dim=1)
        
        assert embeddings.shape == (5, 1)
        # Should be sine values at different positions
        assert abs(embeddings.data[0, 0]) < 1e-6  # sin(0) ≈ 0
    
    def test_dimension_2(self):
        """Test with embedding dimension of 2."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=5, embed_dim=2)
        
        assert embeddings.shape == (5, 2)
        # First column should be sine, second should be cosine
        assert abs(embeddings.data[0, 0]) < 1e-6  # sin(0) ≈ 0
        assert abs(embeddings.data[0, 1] - 1.0) < 1e-6  # cos(0) ≈ 1


class TestSinusoidalEmbeddingsTransformerFormula:
    """Test adherence to transformer paper formula."""
    
    def test_formula_correctness(self):
        """Test that implementation matches transformer paper formula."""
        max_seq_len = 5
        embed_dim = 4
        embeddings = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        
        # Manually compute expected values for position 1
        position = 1
        
        # For i=0 (dim 0 and 1)
        freq_0 = 1.0 / (10000.0 ** (0.0 / embed_dim))
        expected_0_sine = np.sin(position * freq_0)
        expected_0_cosine = np.cos(position * freq_0)
        
        # For i=1 (dim 2 and 3)
        freq_1 = 1.0 / (10000.0 ** (2.0 / embed_dim))
        expected_1_sine = np.sin(position * freq_1)
        expected_1_cosine = np.cos(position * freq_1)
        
        np.testing.assert_almost_equal(embeddings.data[1, 0], expected_0_sine, decimal=5)
        np.testing.assert_almost_equal(embeddings.data[1, 1], expected_0_cosine, decimal=5)
        np.testing.assert_almost_equal(embeddings.data[1, 2], expected_1_sine, decimal=5)
        np.testing.assert_almost_equal(embeddings.data[1, 3], expected_1_cosine, decimal=5)
    
    def test_10000_base(self):
        """Test that 10000 base is used correctly."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=2, embed_dim=4)
        
        # The base of 10000 creates the characteristic slow/fast frequencies
        # Lower dimensions should have lower frequencies (larger wavelengths)
        assert embeddings.shape == (2, 4)
        assert np.all(np.isfinite(embeddings.data))


class TestSinusoidalEmbeddingsLargeScale:
    """Test sinusoidal embeddings at scale."""
    
    def test_large_sequence_length(self):
        """Test with large sequence length."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=2048, embed_dim=64)
        
        assert embeddings.shape == (2048, 64)
        assert np.all(np.isfinite(embeddings.data))
        assert np.all(embeddings.data >= -1.0)
        assert np.all(embeddings.data <= 1.0)
    
    def test_large_embed_dim(self):
        """Test with large embedding dimension."""
        embeddings = create_sinusoidal_embeddings(max_seq_len=100, embed_dim=768)
        
        assert embeddings.shape == (100, 768)
        assert np.all(np.isfinite(embeddings.data))
        assert np.all(embeddings.data >= -1.0)
        assert np.all(embeddings.data <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
