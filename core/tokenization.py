import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter

from core.abstracts import Tokenizer

KB_TO_BYTES = 1024


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer that learns subword units
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """
        1. Split word into characters
        2. Add </w> marker to last character
        3. Return list of tokens
        """

        if not word:
            return []
        
        tokens = list(word)
        tokens[-1] += '</w>'
        return tokens
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs from word tokens
        1. Iterate through adjacent tokens
        2. Create pairs of consecutive tokens
        3. Return set of unique pairs
        """
        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def _build_mappings(self):
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
    
    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """
        Train BPE on corpus to learn merge rules
        1. Build initial character vocabulary
        2. Count word frequencies in corpus
        3. Iteratively merge most frequent pairs
        4. Build final vocabulary and mappings
        
        - Start with character-level tokens using _get_word_tokens()
        - Use Counter to track word frequencies
        - Count all pairs, merge most frequent, repeat until vocab_size reached
        - Don't forget to call _build_mappings() at the end

        """

        if vocab_size:
            self.vocab_size = vocab_size

        word_freq = Counter(corpus)

        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        self.vocab = sorted(list(vocab))

        if '<UNK>' not in self.vocab:
            self.vocab = ['<UNK>'] + self.vocab

        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pair_counts = Counter()

            for word, freq in word_freq.items():
                tokens = word_tokens[word]
                pairs = self._get_pairs(tokens)
                for pair in pairs:
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]

            for word in word_tokens:
                tokens = word_tokens[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]):
                        new_tokens.append(best_pair[0] + best_pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                word_tokens[word] = new_tokens

            merged_token = best_pair[0] + best_pair[1]
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        self._build_mappings()
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        1. Start with character-level tokens
        2. Apply each merge rule in order
        3. Continue until no more merges possible

        For each merge pair, scan through tokens and replace adjacent pairs

        """

        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == merge_pair[0] and
                    tokens[i + 1] == merge_pair[1]):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        1. Split text into words
        2. Convert each word to character tokens
        3. Apply BPE merges
        4. Convert to token IDs

        - Use text.split() for simple word splitting
        - Use _get_word_tokens() to get character-level tokens for each word
        - Use _apply_merges() to apply learned merge rules
        - Use token_to_id dictionary with 0 (UNK) as default
        """
        if not self.vocab:
            return []

        words = text.split()
        all_tokens = []

        for word in words:
            word_tokens = self._get_word_tokens(word)

            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))

        return token_ids
    
    def decode(self, tokens: List[int]) -> str:
        """
        1. Convert IDs to tokens
        2. Join tokens together
        3. Clean up word boundaries and markers

        - Use id_to_token dictionary with '<UNK>' as default
        - Join all tokens into single string with ''.join()
        - Replace '</w>' markers with spaces for word boundaries

        """

        if not self.id_to_token:
            return ""

        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, '<UNK>')
            token_strings.append(token)

        text = ''.join(token_strings)
        text = text.replace('</w>', ' ')
        text = ' '.join(text.split())

        return text


def create_tokenizer(vocab_size: int = 1000, corpus: List[str] = None) -> Tokenizer:

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    if corpus:
        tokenizer.train(corpus, vocab_size)

    return tokenizer


def tokenize_dataset(texts: List[str], tokenizer: Tokenizer, max_length: int = None) -> List[List[int]]:
    """
    Tokenize a dataset with optional length limits.

    1. Encode each text with the tokenizer
    2. Apply max_length truncation if specified
    3. Return list of tokenized sequences

    - Handle empty texts gracefully (empty list is fine)
    - Truncate from the end if too long: tokens[:max_length]
    """
    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized


def analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """
    1. Tokenize all texts
    2. Compute sequence length statistics
    3. Calculate compression ratio
    4. Return analysis dictionary

    - Use np.mean() for average sequence length
    - Compression ratio = total_characters / total_tokens
    - Return dict with vocab_size, avg_sequence_length, max_sequence_length, etc.
    """
    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        'vocab_size': tokenizer.vocab_size,
        'avg_sequence_length': np.mean(tokenized_lengths),
        'max_sequence_length': max(tokenized_lengths) if tokenized_lengths else 0,
        'total_tokens': len(all_tokens),
        'compression_ratio': total_chars / len(all_tokens) if all_tokens else 0,
        'unique_tokens': len(set(all_tokens))
    }

    return stats

