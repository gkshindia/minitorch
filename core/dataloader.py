import numpy as np
import random
import time
import sys
from typing import Optional, Tuple, List, Iterator, Union

from core.abstracts import DatasetAbstract
from core.tensor import Tensor


"""
TensorDataset takes multiple tensors and align them by their first dimension

Memory locality - All data is pre-loaded and stored contiguously in memory, enabling fast access patterns
Vectorized operations - no conversion overhead during training , as data are already in tensor format
Supervised learning
Batch friendly

TensorDataset transofrms arrays of data into a datast that serves samples
"""


class TensorDataset(DatasetAbstract):
    """
    Dataset wrapping tensors for supervised learning.

    Each sample is a tuple of tensors from the same index across all input tensors.
    All tensors must have the same size in their first dimension.

    1. Store all input tensors
    2. Validate they have same first dimension (number of samples)
    3. Return tuple of tensor slices for each index

    - Use *tensors to accept variable number of tensor arguments
    - Check all tensors have same length in dimension 0
    - Return tuple of tensor[idx] for all tensors

    """

    def __init__(self, *tensors):
        assert len(tensors) > 0, "Must provide at least 1 tensor"

        self.tensors = tensors

        first_size = len(tensors[0].data)
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"All tensors must have same size in first dimension"
                    f"Tensor 0: {first_size}, Tensor {i}: {len(tensor.data)}"
                )
    
    def __len__(self):
        return len(self.tensors[0].data)
    
    def __getitem__(self, idx):
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        return tuple(Tensor(tensor.data[idx] for tensor in self.tensors))


class DataLoader:
    """
    Data loader with batching and shuffling support

    1. Store dataset, batch_size, and shuffle settings
    2. Create iterator that groups samples into batches
    3. Handle shuffling by randomizing indices
    4. Collate individual samples into batch tensors

    Algo : 
    1. Create indices list: [0, 1, 2, ..., dataset_length-1]
    2. If shuffle=True: randomly shuffle the indices
    3. Group indices into chunks of batch_size
    4. For each chunk:
        a. Retrieve samples: [dataset[i] for i in chunk]
        b. Collate samples: stack individual tensors into batch tensors
        c. Yield the batch tensor tuple

    """

    def __init__(self, dataset: DatasetAbstract, batch_size: int, shuffle: bool = False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        """
        Collate individual samples into batch tensors
        1. Handle empty batch edge case
        2. Determine how many tensors per sample (e.g., 2 for features + labels)
        3. For each tensor position, extract all samples at that position
        4. Stack them using np.stack() to create batch dimension
        5. Wrap result in Tensor and return tuple

        """

        if len(batch) == 0:
            return ()
        
        num_tensors = len(batch[0])

        batched_tensors = []
        for tensor_idx in range(num_tensors):
            tensor_list = [sample[tensor_idx].data for sample in batch]

            batched_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batched_data))
        
        return tuple(batched_tensors)

        
    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            yield self._collate_batch(batch)


class RandomHorizontalFlip:
    """
    Randomly flip images horizontally with given probability
    """

    def __init__(self, p=0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Rpbability must be between 0 and 1, got {p}")
        self.p = p

    def __call__(self, x):
        
        if np.random.random() < self.p:
            # Flip along the width axis (last axis for HW format, second-to-last for HWC)
            # Using axis=-1 works for both (..., H, W) and (..., H, W, C)
            if isinstance(x, Tensor):
                return Tensor(np.flip(x.data, axis=-1).copy())
            else:
                return np.flip(x, axis=-1).copy()
            
        return x

class RandomCrop:
    """
    Randomly crop image after padding
    """
    def __init__(self, size, padding = 4):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, x):
        """
        1. Add zero-padding to all sides
        2. Choose random top-left corner for crop
        3. Extract crop of target size

        - Use np.pad for adding zeros
        - Handle both (C, H, W) and (H, W) formats
        - Random offsets should be in [0, 2*padding]
        """

        is_tensor = isinstance(x, Tensor)
        data = x.data if is_tensor else x

        target_h, target_w = self.size

        if len(data.shape) == 2:
            h, w = data.shape
            padded = np.pad(data, self.padding, mode='constant', constant_values=0)

            top = np.random.randint(0, 2 * self.padding + h - target_h + 1)
            left = np.random.randint(0, 2 * self.padding + w - target_w + 1)

            cropped = padded[top:top + target_h, left:left + target_w]
        elif len(data.shape) == 3:
            if data.shape[0] <= 4:  # Assume (C, H, W)
                c, h, w = data.shape
                padded = np.pad(data, 
                                ((0,0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
                top = np.random.randint(0, 2 * self.padding + 1)
                left = np.random.randint(0, 2 * self.padding + 1)
                cropped = padded[:, top:top + target_h, left:left + target_w]
            else:  # Assume (H, W, C)
                h, w, c = data.shape
                padded = np.pad(data, 
                                ((self.padding, self.padding), (self.padding, self.padding), (0,0)),
                                mode='constant', constant_values=0)
                
                top = np.random.randint(0, 2 * self.padding + 1)
                left = np.random.randint(0, 2 * self.padding + 1)
                cropped = padded[top:top + target_h, left:left + target_w, :]
        else:
            raise ValueError(f"Input must be 2D or 3D array/tensor, got {data.shape}")
        
        return Tensor(cropped) if is_tensor else cropped


class Compose:
    """
    Compose several transforms together into a pipeline
    Applies transforms in sequence, passing output of one as input to next
    """
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x