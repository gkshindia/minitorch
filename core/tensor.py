import numpy as np

#  constants for memory calculations
BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MT_TO_BYTES = 1024 * 1024

"""
Tensor Class Architecture

Tensor Class Structure:
┌─────────────────────────────────┐
│ Core Attributes:                │
│ • data: np.array (the numbers)  │
│ • shape: tuple (dimensions)     │
│ • size: int (total elements)    │
│ • dtype: type (float32)         │
├─────────────────────────────────┤
│ Arithmetic Operations:          │
│ • __add__, __sub__, __mul__     │
│ • __truediv__, matmul()         │
├─────────────────────────────────┤
│ Shape Operations:               │
│ • reshape(), transpose()        │
│ • sum(), mean(), max()          │
│ • __getitem__ (indexing)        │
├─────────────────────────────────┤
│ Utility Methods:                │
│ • __repr__(), __str__()         │
│ • numpy(), memory_footprint()   │
└─────────────────────────────────┘

Core operations include arithmetic and shape manipulations, while utility methods aid in representation and memory calculations.

Operation Types:
┌─────────────────┬─────────────────┬─────────────────┐
│ Element-wise    │ Matrix Ops      │ Shape Ops       │
├─────────────────┼─────────────────┼─────────────────┤
│ + Addition      │ @ Matrix Mult   │ .reshape()      │
│ - Subtraction   │ .transpose()    │ .sum()          │
│ * Multiplication│                 │ .mean()         │
│ / Division      │                 │ .max()          │
└─────────────────┴─────────────────┴─────────────────┘

Broadcasting Examples:
┌─────────────────────────────────────────────────────────┐
│ Scalar + Vector:                                        │
│    5    + [1, 2, 3] → [5, 5, 5] + [1, 2, 3] = [6, 7, 8] │
│                                                         │
│ Matrix + Vector (row-wise):                             │
│ [[1, 2]]   [10]   [[1, 2]]   [[10, 10]]   [[11, 12]]    │
│ [[3, 4]] + [10] = [[3, 4]] + [[10, 10]] = [[13, 14]]    │
└─────────────────────────────────────────────────────────┘

"""

class Tensor:
    
    def __init__(self, data):

        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        return f"Tensor({self.data})"

    def numpy(self):
        return self.data

    def memory_footprint(self):
        return self.data.nbytes

    def __add__(self, other):
        """
        Add two tensors element-wise with broadcasting support.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """
        Multiply two tensors element-wise (NOT matrix multiplication).
        Element-wise multiplication is *, not matrix multiplication (@)
        """
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        """
        Divide two tensors element-wise.
        """
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        """
        Matrix multiplication of two tensors.

        1. Validate other is a Tensor (raise TypeError if not)
        2. Check for scalar cases (0D tensors) - use element-wise multiply
        3. For 2D+ matrices: validate inner dimensions match (shape[-1] == shape[-2])
        4. For 2D matrices: use explicit nested loops (educational)
        5. For batched (3D+): use np.matmul for correctness
        6. Return result wrapped in Tensor

        - Inner dimensions must match: (M, K) @ (K, N) = (M, N)
        - For 2D case: use np.dot(a[i, :], b[:, j]) for each output element
        - Raise ValueError with clear message if shapes incompatible
        """

        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor for matrix multipleication, got {type(other)}")
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(self.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape) >= 2 or len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}"
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )
        
        a = self.data
        b = other.data

        # Handle 2d matrices with explicit loops
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape

            result_data = np.zeros((M, N), dtype=a.dtype)
            # Each output element is a dot product of a row from A and a column from B
            for i in range(M):
                for j in range(N):
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        
        else:
            # For batched operations (3D+), use np.matmul for correctness
            result_data = np.matmul(a, b)
        
        return Tensor(result_data)

    def __matmul__(self, other):
        return self.matmul(other)

    def __getitem__(self, key):
        """
        Enable indexing and slicing operations on Tensors
        NumPy's indexing already handles all complex cases (slicing, fancy indexing)
        """
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        """
        Reshape tensor to new dimensions

        EXAMPLE:
        >>> t = Tensor([1, 2, 3, 4, 5, 6])
        >>> reshaped = t.reshape(2, 3)
        >>> print(reshaped.data)
        [[1. 2. 3.]
         [4. 5. 6.]]
        >>> auto = t.reshape(2, -1)  # Infers -1 as 3 -> total_elements (6) / known_dimension (2) = 3
        >>> print(auto.shape)
        (2, 3)

        """

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")

            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim

            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} ≠ {target_size}"
            )

        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)
    
    def transpose(self, dim0=None, dim1=None):
        """
        APPROACH:
        1. If no dims specified: swap last two dimensions (most common case)
        2. For 1D tensors: return copy (no transpose needed)
        3. If both dims specified: swap those specific dimensions
        4. Use np.transpose with axes list to perform the swap
        5. Return result wrapped in new Tensor

        - Create axes list: [0, 1, 2, ...] then swap positions
        - For default: axes[-2], axes[-1] = axes[-1], axes[-2]
        - Use np.transpose(self.data, axes)

        """

        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        
        return Tensor(transposed_data)
    
    def sum(self, axis=None, keepdims=False):
        """
        1. Use np.sum with axis and keepdims parameters
        2. axis=None sums all elements (scalar result)
        3. axis=N sums along dimension N
        4. keepdims=True preserves original number of dimensions
        5. Return result wrapped in Tensor

        """
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
    
    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
    
    def max(self, axis=None, keepdims=False):
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

