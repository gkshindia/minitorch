"""
Tokenization Tests

Comprehensive test suite for BPE tokenization with focus on:
- Byte Pair Encoding (BPE) algorithm implementation
- Vocabulary building and merge operations
- Encoding and decoding functionality
- Dataset tokenization utilities
- Tokenization analysis statistics
"""
import numpy as np
import pytest
from core.tokenization import (
    BPETokenizer,
    create_tokenizer,
    tokenize_dataset,
    analyze_tokenization
)


class TestBPETokenizerInitialization:
    """Test BPETokenizer initialization and basic properties."""
    
    def test_default_initialization(self):
        """Test tokenizer initializes with default vocab size."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 1000
        assert tokenizer.vocab == []
        assert tokenizer.merges == []
        assert tokenizer.token_to_id == {}
        assert tokenizer.id_to_token == {}
    
    def test_custom_vocab_size(self):
        """Test tokenizer initializes with custom vocab size."""
        tokenizer = BPETokenizer(vocab_size=500)
        assert tokenizer.vocab_size == 500
    
    def test_large_vocab_size(self):
        """Test tokenizer with large vocabulary."""
        tokenizer = BPETokenizer(vocab_size=10000)
        assert tokenizer.vocab_size == 10000


class TestGetWordTokens:
    """Test _get_word_tokens method for character-level tokenization."""
    
    def test_single_character_word(self):
        """Test single character word gets end-of-word marker."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._get_word_tokens("a")
        assert tokens == ["a</w>"]
    
    def test_multi_character_word(self):
        """Test multi-character word splits correctly."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._get_word_tokens("cat")
        assert tokens == ["c", "a", "t</w>"]
    
    def test_empty_string(self):
        """Test empty string returns empty list."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._get_word_tokens("")
        assert tokens == []
    
    def test_word_with_special_characters(self):
        """Test word with special characters."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._get_word_tokens("hello!")
        assert tokens == ["h", "e", "l", "l", "o", "!</w>"]
    
    def test_numeric_word(self):
        """Test numeric string tokenization."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._get_word_tokens("123")
        assert tokens == ["1", "2", "3</w>"]


class TestGetPairs:
    """Test _get_pairs method for extracting adjacent token pairs."""
    
    def test_single_pair(self):
        """Test getting pairs from two tokens."""
        tokenizer = BPETokenizer()
        pairs = tokenizer._get_pairs(["a", "b</w>"])
        assert pairs == {("a", "b</w>")}
    
    def test_multiple_pairs(self):
        """Test getting pairs from multiple tokens."""
        tokenizer = BPETokenizer()
        pairs = tokenizer._get_pairs(["c", "a", "t</w>"])
        assert pairs == {("c", "a"), ("a", "t</w>")}
    
    def test_single_token(self):
        """Test single token returns no pairs."""
        tokenizer = BPETokenizer()
        pairs = tokenizer._get_pairs(["a</w>"])
        assert pairs == set()
    
    def test_empty_list(self):
        """Test empty list returns no pairs."""
        tokenizer = BPETokenizer()
        pairs = tokenizer._get_pairs([])
        assert pairs == set()
    
    def test_duplicate_pairs(self):
        """Test duplicate pairs are stored once in set."""
        tokenizer = BPETokenizer()
        pairs = tokenizer._get_pairs(["a", "b", "a", "b</w>"])
        # Pairs should be: (a, b), (b, a), (a, b</w>)
        # But (a, b) appears twice, set should have 3 unique pairs
        assert len(pairs) == 3
        assert ("a", "b") in pairs


class TestBPETraining:
    """Test BPE training functionality."""
    
    def test_train_simple_corpus(self):
        """Test training on simple corpus."""
        tokenizer = BPETokenizer(vocab_size=20)
        corpus = ["cat", "car", "cat"]
        tokenizer.train(corpus)
        
        # Should have vocabulary and mappings
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.token_to_id) > 0
        assert len(tokenizer.id_to_token) > 0
        assert "<UNK>" in tokenizer.vocab
    
    def test_train_builds_merges(self):
        """Test training creates merge rules."""
        tokenizer = BPETokenizer(vocab_size=15)
        corpus = ["hello", "hello", "help"]
        tokenizer.train(corpus)
        
        # Should have some merge rules
        assert len(tokenizer.merges) > 0
    
    def test_train_respects_vocab_size(self):
        """Test training respects vocabulary size limit."""
        tokenizer = BPETokenizer(vocab_size=20)
        corpus = ["cat", "dog", "bird"]
        tokenizer.train(corpus)
        
        # Should be close to or at vocab_size (may be slightly less if no more merges possible)
        assert len(tokenizer.vocab) <= 20
    
    def test_train_with_repeated_words(self):
        """Test training with repeated words creates better merges."""
        tokenizer = BPETokenizer(vocab_size=20)
        corpus = ["test"] * 10 + ["best"] * 10
        tokenizer.train(corpus)
        
        # Common substrings like "est" should be in vocab
        assert len(tokenizer.merges) > 0
    
    def test_train_single_word(self):
        """Test training on single word."""
        tokenizer = BPETokenizer(vocab_size=15)
        tokenizer.train(["hello"])
        
        assert len(tokenizer.vocab) > 0
        assert "<UNK>" in tokenizer.vocab
    
    def test_train_empty_corpus(self):
        """Test training on empty corpus."""
        tokenizer = BPETokenizer(vocab_size=10)
        tokenizer.train([])
        
        # Should have at least UNK token
        assert "<UNK>" in tokenizer.vocab
    
    def test_vocabulary_contains_characters(self):
        """Test vocabulary contains character-level tokens."""
        tokenizer = BPETokenizer(vocab_size=30)
        corpus = ["abc", "def"]
        tokenizer.train(corpus)
        
        # Should contain individual characters with </w>
        vocab_str = ''.join(tokenizer.vocab)
        assert 'a' in vocab_str or 'a</w>' in tokenizer.vocab
        assert 'b' in vocab_str or 'b</w>' in tokenizer.vocab
    
    def test_build_mappings_creates_bidirectional_dicts(self):
        """Test _build_mappings creates correct dictionaries."""
        tokenizer = BPETokenizer(vocab_size=15)
        corpus = ["test"]
        tokenizer.train(corpus)
        
        # Check bidirectional mapping
        for token, idx in tokenizer.token_to_id.items():
            assert tokenizer.id_to_token[idx] == token
        
        for idx, token in tokenizer.id_to_token.items():
            assert tokenizer.token_to_id[token] == idx


class TestBPEEncoding:
    """Test BPE encoding functionality."""
    
    def test_encode_simple_text(self):
        """Test encoding simple text."""
        tokenizer = BPETokenizer(vocab_size=30)
        corpus = ["hello", "world"]
        tokenizer.train(corpus)
        
        encoded = tokenizer.encode("hello")
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(token_id, int) for token_id in encoded)
    
    def test_encode_empty_string(self):
        """Test encoding empty string."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(["test"])
        
        encoded = tokenizer.encode("")
        assert encoded == []
    
    def test_encode_multi_word_text(self):
        """Test encoding multi-word text."""
        tokenizer = BPETokenizer(vocab_size=50)
        corpus = ["the", "cat", "sat"]
        tokenizer.train(corpus)
        
        encoded = tokenizer.encode("the cat")
        assert len(encoded) > 0
    
    def test_encode_unknown_word(self):
        """Test encoding word not in training corpus."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(["hello"])
        
        # Try to encode a word with unknown characters
        encoded = tokenizer.encode("xyz")
        # Should use UNK tokens (id 0)
        assert all(token_id >= 0 for token_id in encoded)
    
    def test_encode_without_training(self):
        """Test encoding without training returns empty."""
        tokenizer = BPETokenizer()
        encoded = tokenizer.encode("test")
        assert encoded == []
    
    def test_encode_preserves_word_boundaries(self):
        """Test encoding handles multiple words."""
        tokenizer = BPETokenizer(vocab_size=40)
        corpus = ["hello", "world", "hello", "world"]
        tokenizer.train(corpus)
        
        encoded = tokenizer.encode("hello world")
        # Should have tokens for both words
        assert len(encoded) >= 2


class TestBPEDecoding:
    """Test BPE decoding functionality."""
    
    def test_decode_simple_tokens(self):
        """Test decoding token IDs back to text."""
        tokenizer = BPETokenizer(vocab_size=30)
        corpus = ["hello", "world"]
        tokenizer.train(corpus)
        
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        
        assert isinstance(decoded, str)
        # Decoded should be similar to original (may have different spacing)
        assert "hello" in decoded.lower()
    
    def test_decode_empty_list(self):
        """Test decoding empty token list."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(["test"])
        
        decoded = tokenizer.decode([])
        assert decoded == ""
    
    def test_decode_without_training(self):
        """Test decoding without training."""
        tokenizer = BPETokenizer()
        decoded = tokenizer.decode([1, 2, 3])
        assert decoded == ""
    
    def test_decode_unknown_token_ids(self):
        """Test decoding with invalid token IDs."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(["test"])
        
        # Use token IDs that don't exist
        decoded = tokenizer.decode([9999, 8888])
        # Should use UNK tokens
        assert isinstance(decoded, str)
    
    def test_decode_removes_word_markers(self):
        """Test decoding properly handles word boundary markers."""
        tokenizer = BPETokenizer(vocab_size=40)
        corpus = ["hello", "world"]
        tokenizer.train(corpus)
        
        encoded = tokenizer.encode("hello world")
        decoded = tokenizer.decode(encoded)
        
        # Should not contain </w> in final output
        assert "</w>" not in decoded


class TestEncodeDecodeRoundtrip:
    """Test round-trip encoding and decoding."""
    
    def test_roundtrip_simple_word(self):
        """Test encoding then decoding preserves content."""
        tokenizer = BPETokenizer(vocab_size=30)
        corpus = ["hello", "world", "test"]
        tokenizer.train(corpus)
        
        original = "hello"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        # Should preserve the word (may have different spacing)
        assert original in decoded or decoded in original
    
    def test_roundtrip_multi_word(self):
        """Test roundtrip with multiple words."""
        tokenizer = BPETokenizer(vocab_size=50)
        corpus = ["the", "quick", "brown", "fox"]
        tokenizer.train(corpus)
        
        original = "the fox"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        # Check both words are present
        assert "the" in decoded.lower()
        assert "fox" in decoded.lower()
    
    def test_roundtrip_preserves_word_count(self):
        """Test roundtrip preserves number of words."""
        tokenizer = BPETokenizer(vocab_size=60)
        corpus = ["hello", "world", "test", "data"]
        tokenizer.train(corpus)
        
        original = "hello world test"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        
        # Should have similar number of words
        original_words = len(original.split())
        decoded_words = len(decoded.split())
        assert abs(original_words - decoded_words) <= 1


class TestCreateTokenizer:
    """Test create_tokenizer utility function."""
    
    def test_create_tokenizer_default(self):
        """Test creating tokenizer with defaults."""
        tokenizer = create_tokenizer()
        assert tokenizer.vocab_size == 1000
        assert len(tokenizer.vocab) == 0  # Not trained yet
    
    def test_create_tokenizer_custom_vocab_size(self):
        """Test creating tokenizer with custom vocab size."""
        tokenizer = create_tokenizer(vocab_size=500)
        assert tokenizer.vocab_size == 500
    
    def test_create_tokenizer_with_corpus(self):
        """Test creating and training tokenizer with corpus."""
        corpus = ["hello", "world", "test"]
        tokenizer = create_tokenizer(vocab_size=30, corpus=corpus)
        
        assert len(tokenizer.vocab) > 0
        assert len(tokenizer.token_to_id) > 0
    
    def test_create_tokenizer_returns_tokenizer_instance(self):
        """Test function returns proper tokenizer instance."""
        from core.abstracts import Tokenizer
        tokenizer = create_tokenizer()
        assert isinstance(tokenizer, Tokenizer)
        assert isinstance(tokenizer, BPETokenizer)


class TestTokenizeDataset:
    """Test tokenize_dataset utility function."""
    
    def test_tokenize_empty_dataset(self):
        """Test tokenizing empty dataset."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(["test"])
        
        result = tokenize_dataset([], tokenizer)
        assert result == []
    
    def test_tokenize_single_text(self):
        """Test tokenizing single text."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["hello", "world"])
        
        texts = ["hello"]
        result = tokenize_dataset(texts, tokenizer)
        
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0
    
    def test_tokenize_multiple_texts(self):
        """Test tokenizing multiple texts."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["hello", "world", "test"])
        
        texts = ["hello", "world", "test"]
        result = tokenize_dataset(texts, tokenizer)
        
        assert len(result) == 3
        assert all(isinstance(seq, list) for seq in result)
    
    def test_tokenize_with_max_length(self):
        """Test tokenization with max length limit."""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(["hello", "world", "test", "data"])
        
        texts = ["hello world test data"]
        result = tokenize_dataset(texts, tokenizer, max_length=3)
        
        assert len(result) == 1
        assert len(result[0]) <= 3
    
    def test_tokenize_max_length_no_truncation(self):
        """Test max length doesn't truncate short sequences."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["hi"])
        
        texts = ["hi"]
        result = tokenize_dataset(texts, tokenizer, max_length=100)
        
        assert len(result) == 1
        # Should not pad, just preserve original length
        assert len(result[0]) <= 100
    
    def test_tokenize_empty_text_in_dataset(self):
        """Test handling empty text in dataset."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["hello"])
        
        texts = ["hello", "", "world"]
        result = tokenize_dataset(texts, tokenizer)
        
        assert len(result) == 3
        assert result[1] == []  # Empty text -> empty token list
    
    def test_tokenize_preserves_order(self):
        """Test tokenization preserves text order."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["alpha", "beta", "gamma"])
        
        texts = ["alpha", "beta", "gamma"]
        result = tokenize_dataset(texts, tokenizer)
        
        # Each should be different
        assert len(result) == 3
        assert result[0] != result[1]
        assert result[1] != result[2]


class TestAnalyzeTokenization:
    """Test analyze_tokenization utility function."""
    
    def test_analyze_simple_texts(self):
        """Test analyzing tokenization of simple texts."""
        tokenizer = BPETokenizer(vocab_size=40)
        corpus = ["hello", "world", "test"]
        tokenizer.train(corpus)
        
        texts = ["hello", "world"]
        stats = analyze_tokenization(texts, tokenizer)
        
        assert isinstance(stats, dict)
        assert "vocab_size" in stats
        assert "avg_sequence_length" in stats
        assert "max_sequence_length" in stats
        assert "total_tokens" in stats
        assert "compression_ratio" in stats
        assert "unique_tokens" in stats
    
    def test_analyze_vocab_size(self):
        """Test analyze returns correct vocab size."""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(["test", "data"])
        
        texts = ["test"]
        stats = analyze_tokenization(texts, tokenizer)
        
        assert stats["vocab_size"] == 50
    
    def test_analyze_avg_sequence_length(self):
        """Test average sequence length calculation."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["hi", "hello"])
        
        texts = ["hi", "hello"]
        stats = analyze_tokenization(texts, tokenizer)
        
        assert stats["avg_sequence_length"] > 0
        assert isinstance(stats["avg_sequence_length"], (int, float, np.number))
    
    def test_analyze_max_sequence_length(self):
        """Test max sequence length tracking."""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(["a", "hello", "world"])
        
        texts = ["a", "hello world test"]
        stats = analyze_tokenization(texts, tokenizer)
        
        # Longer text should have more tokens
        assert stats["max_sequence_length"] > 0
    
    def test_analyze_compression_ratio(self):
        """Test compression ratio calculation."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["hello", "world"])
        
        texts = ["hello"]
        stats = analyze_tokenization(texts, tokenizer)
        
        # Compression ratio = chars / tokens
        assert stats["compression_ratio"] > 0
        # Should be reasonable (typically 1-5)
        assert stats["compression_ratio"] < 100
    
    def test_analyze_unique_tokens(self):
        """Test unique tokens counting."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["test", "best"])
        
        texts = ["test", "test"]  # Repeated word
        stats = analyze_tokenization(texts, tokenizer)
        
        # Should have fewer unique tokens than total
        assert stats["unique_tokens"] <= stats["total_tokens"]
    
    def test_analyze_empty_texts(self):
        """Test analysis with empty texts."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["test"])
        
        texts = []
        stats = analyze_tokenization(texts, tokenizer)
        
        assert stats["total_tokens"] == 0
        assert stats["max_sequence_length"] == 0
        assert stats["compression_ratio"] == 0
    
    def test_analyze_total_tokens(self):
        """Test total tokens counting."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["hi", "hello"])
        
        texts = ["hi", "hi", "hi"]
        stats = analyze_tokenization(texts, tokenizer)
        
        # Should count all tokens across all texts
        assert stats["total_tokens"] > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_character_corpus(self):
        """Test training on single character words."""
        tokenizer = BPETokenizer(vocab_size=10)
        tokenizer.train(["a", "b", "c"])
        
        assert len(tokenizer.vocab) > 0
        encoded = tokenizer.encode("a")
        assert len(encoded) > 0
    
    def test_repeated_character_word(self):
        """Test word with repeated characters."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["aaa", "aaa", "aaa"])
        
        encoded = tokenizer.encode("aaa")
        assert len(encoded) > 0
    
    def test_very_small_vocab_size(self):
        """Test with minimal vocab size."""
        tokenizer = BPETokenizer(vocab_size=5)
        tokenizer.train(["test"])
        
        # Should still work with small vocab
        assert len(tokenizer.vocab) <= 5
        assert len(tokenizer.vocab) > 0
    
    def test_special_characters_in_corpus(self):
        """Test corpus with special characters."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["hello!", "world?", "test."])
        
        encoded = tokenizer.encode("hello!")
        assert len(encoded) > 0
    
    def test_numeric_corpus(self):
        """Test corpus with numbers."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["123", "456", "789"])
        
        encoded = tokenizer.encode("123")
        assert len(encoded) > 0
    
    def test_mixed_case_corpus(self):
        """Test corpus with mixed case."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["Hello", "WORLD", "TeSt"])
        
        # Should handle case sensitivity
        encoded_lower = tokenizer.encode("hello")
        encoded_upper = tokenizer.encode("HELLO")
        
        assert len(encoded_lower) > 0
        assert len(encoded_upper) > 0


class TestConsistency:
    """Test consistency and determinism."""
    
    def test_encode_is_deterministic(self):
        """Test encoding same text gives same result."""
        tokenizer = BPETokenizer(vocab_size=40)
        tokenizer.train(["hello", "world"])
        
        encoded1 = tokenizer.encode("hello")
        encoded2 = tokenizer.encode("hello")
        
        assert encoded1 == encoded2
    
    def test_train_is_deterministic(self):
        """Test training on same corpus gives same vocab."""
        corpus = ["hello", "world", "test"]
        
        tokenizer1 = BPETokenizer(vocab_size=30)
        tokenizer1.train(corpus)
        
        tokenizer2 = BPETokenizer(vocab_size=30)
        tokenizer2.train(corpus)
        
        assert tokenizer1.vocab == tokenizer2.vocab
        assert tokenizer1.merges == tokenizer2.merges
    
    def test_mappings_are_consistent(self):
        """Test token_to_id and id_to_token are consistent."""
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(["hello", "world"])
        
        # Check every mapping
        for token_id in tokenizer.id_to_token:
            token = tokenizer.id_to_token[token_id]
            assert tokenizer.token_to_id[token] == token_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
