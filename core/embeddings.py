import numpy as np
import math
from typing import List, Optional, Tuple

from core.tensor import Tensor

BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
MB_TO_BYTES = 1024 * 1024


class Embedding:
    """
    1. Initialize embedding matrix with random weights (vocab_size, embed_dim)
    2. Implement forward pass as matrix lookup using numpy indexing
    3. Handle batch dimensions correctly
    4. Return parameters for optimization

    - Use numpy advanced indexing for lookup: weight[indices]
    - Embedding matrix shape: (vocab_size, embed_dim)
    - Initialize with Xavier/Glorot uniform for stable gradients
    - Handle multi-dimensional indices correctly
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (vocab_size, embed_dim))
        )
    
    def forward(self, indices: Tensor) -> Tensor:
        if np.any(indices.data >= self.vocab_size) or np.any(indices.data < 0):
            raise ValueError(
                f"Index out of range. Expected 0 <= indices < {self.vocab_size}, "
                f"got min={np.min(indices.data)}, max={np.max(indices.data)}"
            )
        
        embedded = self.weight.data[indices.data.astype(int)]
        return Tensor(embedded)
    
    def __call__(self, indices: Tensor) -> Tensor:
        return self.forward(indices)

    def parameters(self) -> List[Tensor]:
        return [self.weight]

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"


class PositionalEncoding:
    """
    1. Create embedding matrix for positions: (max_seq_len, embed_dim)
    2. Forward pass: lookup position embeddings and add to input
    3. Handle different sequence lengths gracefully
    4. Return parameters for training

    - Position embeddings shape: (max_seq_len, embed_dim)
    - Use slice [:seq_len] to handle variable lengths
    - Add position encodings to input embeddings element-wise
    - Initialize with smaller values than token embeddings (they're additive)
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        limit = math.sqrt(2.0 / embed_dim)
        self.position_embeddings = Tensor(np.random.uniform(-limit, limit, (max_seq_len, embed_dim)))

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq, embed), got shape {x.shape}")

        batch_size, seq_len, embed_dim = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embed_dim}, got {embed_dim}"
            )

        # Slice position embeddings for this sequence length using Tensor slicing
        pos_embeddings = self.position_embeddings[:seq_len]  # (seq_len, embed_dim)

        # Reshape to add batch dimension: (1, seq_len, embed_dim)
        pos_data = pos_embeddings.data[np.newaxis, :, :]
        pos_embeddings_batched = Tensor(pos_data)

        # Add positional information
        result = x + pos_embeddings_batched

        return result
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self) -> List[Tensor]:
        return [self.position_embeddings]

    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"


def create_sinusoidal_embeddings(max_seq_len: int, embed_dim: int) -> Tensor:
    """
    1. Create position indices: [0, 1, 2, ..., max_seq_len-1]
    2. Create dimension indices for frequency calculation
    3. Apply sine to even dimensions, cosine to odd dimensions
    4. Use the transformer paper formula with 10000 base

    - Use np.arange to create position and dimension arrays
    - Calculate div_term using exponential for frequency scaling
    - Apply different formulas to even/odd dimensions
    - The 10000 base creates different frequencies for different dimensions
    """

    position = np.arange(max_seq_len, dtype=np.float32)[:, np.newaxis]

    # Create dimension indices for calculating frequencies
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) *
        -(math.log(10000.0) / embed_dim)
    )  # (embed_dim//2,)

    # Initialize the positional encoding matrix
    pe = np.zeros((max_seq_len, embed_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(position * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if embed_dim % 2 == 1:
        # Handle odd embed_dim by only filling available positions
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)

    return Tensor(pe)


class EmbeddingLayer:
    """
    1. Combine token embedding + positional encoding
    2. Support both learned and sinusoidal position encodings
    3. Handle variable sequence lengths gracefully
    4. Add optional embedding scaling (Transformer convention)

    - First apply token embedding, then add positional encoding
    - Support 'learned', 'sinusoidal', or None for pos_encoding
    - Handle both 2D (batch, seq) and 1D (seq) inputs gracefully
    - Scale embeddings by sqrt(embed_dim) if requested (transformer convention)

    """

    def __init__(self, 
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        pos_encoding: str = 'learned',
        scale_embeddings: bool = False
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding
        self.scale_embeddings = scale_embeddings

        self.token_embedding = Embedding(vocab_size, embed_dim)

        if pos_encoding == 'learned':
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding == 'sinusoidal':
            self.pos_encoding = create_sinusoidal_embeddings(max_seq_len, embed_dim)
        elif pos_encoding is None:
            self.pos_encoding = None
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}. Use 'learned', 'sinusoidal', or None")
    
    def forward(self, tokens: Tensor) -> Tensor:
        if len(tokens.shape) == 1:
            tokens = tokens.reshape(1, -1)
            squeeze_batch = True
        else:
            squeeze_batch = False
    
        token_embeds = self.token_embedding.forward(tokens)

        if self.scale_embeddings:
            scale_factor = math.sqrt(self.embed_dim)
            token_embeds = token_embeds * scale_factor
        
        if self.pos_encoding_type == 'learned':
            output = self.pos_encoding.forward(token_embeds)
        elif self.pos_encoding_type == 'sinusoidal':
            batch_size, seq_len, embed_dim = token_embeds.shape
            pos_embeddings = self.pos_encoding[:seq_len]

            pos_data = pos_embeddings.data[np.newaxis, :, :]
            pos_embeddings_batched = Tensor(pos_data)

            output = token_embeds + pos_embeddings_batched
        else:
            output = token_embeds
        if squeeze_batch:
            output = output[0]

        return output

    def __call__(self, tokens: Tensor) -> Tensor:
        return self.forward(tokens)

    def parameters(self) -> List[Tensor]:
        params = self.token_embedding.parameters()

        if self.pos_encoding_type == 'learned':
            params.extend(self.pos_encoding.parameters())

        return params

    def __repr__(self):
        return (f"EmbeddingLayer(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim}, "
                f"pos_encoding='{self.pos_encoding_type}')")