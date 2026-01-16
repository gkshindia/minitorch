import numpy as np

from core.abstracts import FunctionAbstract
from core.tensor import Tensor


EPSILON = 1e-7


"""
Addition (z = a + b):
    ∂z/∂a = 1    ∂z/∂b = 1

    a ──┐           grad_a ←──┐
        ├─[+]─→ z          ├─[+]←── grad_z
    b ──┘           grad_b ←──┘

Multiplication (z = a * b):
    ∂z/∂a = b    ∂z/∂b = a

    a ──┐           grad_a = grad_z * b
        ├─[×]─→ z
    b ──┘           grad_b = grad_z * a

Matrix Multiplication (Z = A @ B):
    ∂Z/∂A = grad_Z @ B.T
    ∂Z/∂B = A.T @ grad_Z

    A ──┐           grad_A = grad_Z @ B.T
        ├─[@]─→ Z
    B ──┘           grad_B = A.T @ grad_Z
"""


class AddBackward(FunctionAbstract):
    """
    Gradient computation for tensor addition.

    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1

    **Key Insight:** Addition distributes gradients equally to both inputs.
    The gradient flowing backward is passed unchanged to each input.

    **Broadcasting Handling:** When input shapes differ due to broadcasting,
    we sum gradients appropriately to match original tensor shapes.

    """

    def apply(self, grad_output):
        """

        **Mathematical Foundation:**
        - ∂(a+b)/∂a = 1 → grad_a = grad_output
        - ∂(a+b)/∂b = 1 → grad_b = grad_output

        1. Extract input tensors from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if it requires gradients, set grad_a = grad_output
        4. For second input (b): if it requires gradients, set grad_b = grad_output
        5. Return tuple (grad_a, grad_b)

        - Addition distributes gradients equally (derivative of a+b w.r.t. both is 1)
        - Check isinstance(tensor, Tensor) and tensor.requires_grad before computing
        - Return None for inputs that don't require gradients
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
        
        return grad_a, grad_b


class MulBackward(FunctionAbstract):
    """
    Gradient computation for tensor multiplication.

    **Mathematical Rule:** If z = a * b, then ∂z/∂a = b and ∂z/∂b = a

    **Key Insight:** Each input's gradient equals the gradient output
    multiplied by the OTHER input's value (product rule).

    **Applications:** Used in weight scaling, attention mechanisms,
    and anywhere element-wise multiplication occurs.
    """

    def apply(self, grad_output):
        """

        **Mathematical Foundation:**
        - ∂(a*b)/∂a = b → grad_a = grad_output * b
        - ∂(a*b)/∂b = a → grad_b = grad_output * a

        1. Extract input tensors a, b from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if requires_grad, compute grad_a = grad_output * b
        4. For second input (b): if requires_grad, compute grad_b = grad_output * a
        5. Handle both Tensor and scalar cases for b
        6. Return tuple (grad_a, grad_b)

        - Product rule: each input's gradient equals grad_output times the OTHER input
        - Check if b is a Tensor or scalar before accessing .data
        - Use b.data if Tensor, or b directly if scalar
        
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b
        
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data

        return grad_a, grad_b
    

class SubBackward(FunctionAbstract):
    """
    Gradient computation for tensor subtraction.

    **Mathematical Rule:** If z = a - b, then ∂z/∂a = 1 and ∂z/∂b = -1
    """

    def apply(self, grad_output):
        """
        
        1. Extract input tensors from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if requires_grad, set grad_a = grad_output
        4. For second input (b): if requires_grad, set grad_b = -grad_output (note the negative!)
        5. Return tuple (grad_a, grad_b)

        - ∂(a-b)/∂a = 1 (gradient flows unchanged to first operand)
        - ∂(a-b)/∂b = -1 (gradient is negated for second operand)
        - The negative sign is crucial for correct gradient flow

        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
        
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output
        
        return grad_a, grad_b
    

class DivBackward(FunctionAbstract):
    """
    Gradient computation for tensor division.

    **Mathematical Rule:** If z = a / b, then:
    - ∂z/∂a = 1/b
    - ∂z/∂b = -a/b²
    """
      

    def apply(self, grad_output):
        """
        
        1. Extract input tensors from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a): if requires_grad, compute grad_a = grad_output / b
        4. For second input (b): if requires_grad, compute grad_b = -grad_output * a / (b²)
        5. Handle both Tensor and scalar cases for b
        6. Return tuple (grad_a, grad_b)

        - Quotient rule: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
        - Use b.data if Tensor, or b directly if scalar
        - b² means b.data ** 2 for tensors

        """
        
        a, b = self.saved_tensors
        grad_a = grad_b = None
            
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output / b.data
            else:
                grad_a = grad_output / b
        
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output * a.data / (b.data ** 2)
        
        return grad_a, grad_b
        
    
class MatMulBackward(FunctionAbstract):
    """
    Gradient computation for matrix multiplication.

    **Mathematical Rule:** If Z = A @ B, then:
    - ∂Z/∂A = grad_Z @ B.T
    - ∂Z/∂B = A.T @ grad_Z
    """

    def apply(self, grad_output):
        """
        1. Extract input tensors a, b from self.saved_tensors
        2. Initialize grad_a and grad_b to None
        3. For first input (a):
           - Transpose b: use np.swapaxes(b.data, -2, -1) for batched tensors
           - Compute grad_a = grad_output @ b_T using np.matmul
        4. For second input (b):
           - Transpose a: use np.swapaxes(a.data, -2, -1) for batched tensors
           - Compute grad_b = a_T @ grad_output using np.matmul
        5. Return tuple (grad_a, grad_b)

        - Matrix multiplication gradients involve transposing one input
        - Use np.swapaxes(array, -2, -1) to transpose last two dimensions
        - This preserves batch dimensions for 3D+ tensors
        - Use np.matmul for the actual matrix multiplication

        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data, -2, -1)
            else:
                b_T = b.data.T
            grad_a = np.matmul(grad_output, b_T)
        
        if isinstance(b, Tensor) and b.requires_grad:
            if a.data.ndim >= 2:
                a_T = np.swapaxes(a.data, -2, -1)
            else:
                a_T = a.data.T
            grad_b = np.matmul(a_T, grad_output)
        
        return grad_a, grad_b


class TransposeBackward(FunctionAbstract):
    """
    Gradient computation for transpose operation.

    **Mathematical Rule:** If Y = X.T, then:
    - ∂Y/∂X = grad_Y.T

    **Key Insight:** The gradient of transpose is just transpose the gradient!
    This is because transpose is a linear operation that just rearranges elements.

    **Applications:** Used in attention (K.T for scores), weight gradients (W.T),
    and any operation that needs to swap matrix dimensions.
    """

    def __init__(self, tensor, dim0, dim1):
        """
        Args:
            tensor: Input tensor
            dim0: First dimension to swap (None for default)
            dim1: Second dimension to swap (None for default)
        """
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        """
        Compute gradient for transpose.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - ∂(X.T)/∂X = grad_output.T
        - Just transpose the gradient back!

        APPROACH:
        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
           - Check if default transpose (last two dims) or specific dims
           - For default: swap last two dimensions of grad_output
           - For specific dims: swap the specified dimensions back
        4. Return tuple (grad_x,)

        HINTS:
        - Transpose gradient is simply transposing the gradient back
        - Use np.transpose(grad_output, axes) to specify axis order
        - For default transpose, swap axes[-2] and axes[-1]
        - Return as single-element tuple: (grad_x,)
        """
        ### BEGIN SOLUTION
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Transpose gradient using the same dims
            if self.dim0 is None and self.dim1 is None:
                # Default: transpose last two dimensions
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = np.transpose(grad_output, axes)
            else:
                # Specific dimensions: swap them back
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = np.transpose(grad_output, axes)

        return (grad_x,)
    

class PermuteBackward(FunctionAbstract):
    """
    Gradient computation for arbitrary axis permutation (general transpose).

    **Mathematical Rule:** If Y = X.permute(axes), then:
    - ∂Y/∂X = grad_Y.permute(inverse_axes)

    **Example:** If axes = (0, 2, 1, 3), the inverse is (0, 2, 1, 3) (self-inverse).
    More generally, if axes = (2, 0, 1), the inverse is (1, 2, 0).

    **Applications:** Multi-head attention uses (0, 2, 1, 3) to rearrange heads.
    """

    def __init__(self, tensor, axes):
        super().__init__(tensor)
        self.axes = axes
        # Compute inverse permutation: if axes[i] = j, then inverse_axes[j] = i
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        """
        Compute gradient for permutation.

        The gradient is permuted back using the inverse permutation.

        **Mathematical Foundation:**
        - ∂(X.permute(axes))/∂X = grad_output.permute(inverse_axes)

        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
           - Permute grad_output using self.inverse_axes
           - Use np.transpose(grad_output, self.inverse_axes)
        4. Return tuple (grad_x,)

        - Inverse permutation is precomputed in __init__ using np.argsort
        - Simply apply np.transpose with inverse_axes
        - Return as single-element tuple: (grad_x,)
        """
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Permute gradient back to original axis order
            grad_x = np.transpose(grad_output, self.inverse_axes)

        return (grad_x,)
    

class EmbeddingBackward(FunctionAbstract):
    """
    Gradient computation for embedding lookup operation.

    **Mathematical Rule:** If Y = Embedding[indices], then:
    - ∂Loss/∂Embedding[i] = sum of all gradients where index==i

    **Key Insight:** Embedding lookup is a gather operation. The backward
    is a scatter operation that accumulates gradients to the embedding weights.

    **Applications:** Word embeddings, positional embeddings, token embeddings
    in transformers.
    """

    def __init__(self, weight, indices):
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_output):
        """
        Compute gradient for embedding lookup.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for weight tensor

        **Mathematical Foundation:**
        - ∂(Embedding[indices])/∂Embedding = scatter gradients to selected rows
        - Multiple indices can point to same embedding → gradients accumulate

        1. Extract weight tensor from self.saved_tensors
        2. Initialize grad_weight to None
        3. If weight requires gradients:
           - Create zeros array: grad_weight = np.zeros_like(weight.data)
           - Flatten indices: indices_flat = self.indices.data.astype(int).flatten()
           - Reshape grad_output: match flattened indices with embedding dimension
           - Use np.add.at to accumulate gradients: np.add.at(grad_weight, indices_flat, grad_output_reshaped)
        4. Return tuple (grad_weight,)


        - Embedding lookup is a gather operation; backward is scatter
        - np.add.at accumulates gradients for repeated indices
        - Reshape grad_output to match: (num_indices, embedding_dim)
        - Return as single-element tuple: (grad_weight,)
        """
        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:

            grad_weight = np.zeros_like(weight.data)

            # Scatter gradients back to embedding weights
            # np.add.at accumulates gradients for repeated indices
            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight,)


class SliceBackward(FunctionAbstract):
    """
    Gradient computation for tensor slicing/indexing operations.

    **Mathematical Rule:** If Y = X[key], then:
    - ∂Loss/∂X[key] = grad_output
    - ∂Loss/∂X[other positions] = 0

    **Key Insight:** Slicing is a masking operation. The backward
    places gradients back into the original tensor positions, with
    zeros everywhere else.

    **Applications:** Positional encodings, sequence slicing, batch selection,
    attention masking in transformers.

    """

    def __init__(self, tensor, key):
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        """
        Compute gradient for slicing operation.

        **Mathematical Foundation:**
        - Slicing extracts a subset of elements
        - Backward scatters gradients back to original positions
        - Unsliced positions receive zero gradient

        1. Extract input tensor from self.saved_tensors
        2. Initialize grad_input to None
        3. If tensor requires gradients:
           - Create zeros array: grad_input = np.zeros(self.original_shape)
           - Place gradients back: grad_input[self.key] = grad_output
        4. Return tuple (grad_input,)

        - Create zero gradient array with original tensor shape
        - Use fancy indexing: grad_input[self.key] = grad_output
        - This automatically handles all slice types (single index, ranges, tuples)
        - Return as single-element tuple: (grad_input,)
        """

        tensor, = self.saved_tensors
        grad_input = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:

            grad_input = np.zeros(self.original_shape, dtype=np.float32)

            # Place gradients back into the sliced positions
            # This is the inverse of the forward slicing operation
            grad_input[self.key] = grad_output

        return (grad_input,)


class ReshapeBackward(FunctionAbstract):
    """
    Gradient computation for reshape operation.

    **Mathematical Rule:** If Y = X.reshape(new_shape), then:
    - ∂Y/∂X = grad_Y.reshape(X.shape)

    **Key Insight:** Reshape just rearranges the same elements.
    The gradient is simply reshaped back to the original shape!

    **Applications:** Flattening tensors for linear layers, reshaping
    between convolutional and dense layers.
    """

    def __init__(self, tensor, original_shape):
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        """
        Compute gradient for reshape.

        **Mathematical Foundation:**
        - ∂(X.reshape(...))/∂X = grad_output.reshape(X.shape)
        - Just reshape the gradient back!
        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
           - Reshape grad_output back to original shape
           - Use grad_output.reshape(self.original_shape)
        4. Return tuple (grad_x,)

        - Reshape just rearranges elements, doesn't change values
        - Simply reshape gradient back to original shape
        - Use .reshape() method on grad_output numpy array
        - Return as single-element tuple: (grad_x,)
        """

        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Reshape gradient back to original shape
            grad_x = grad_output.reshape(self.original_shape)

        return (grad_x,)


class SumBackward(FunctionAbstract):
    """
    Gradient computation for tensor sum.

    **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i

    **Key Insight:** Sum distributes the gradient equally to all input elements.
    The gradient is broadcast from the reduced output back to input shape.

    **Applications:** Used in loss functions, mean operations, and
    anywhere tensor reduction occurs.
    """

    def apply(self, grad_output):
        """
        Compute gradients for sum operation.
        **Mathematical Foundation:**
        - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output

        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Create ones array: np.ones_like(tensor.data)
           - Multiply by grad_output: ones * grad_output
           - Return as tuple: (grad_tensor,)
        3. Else return (None,)

        - Sum distributes gradient equally to all elements
        - Use np.ones_like(tensor.data) to create gradient template
        - Multiply ones by grad_output (broadcasting handles scalar/tensor)
        - Return as single-element tuple: (grad_result,)
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Gradient is 1 for all elements, scaled by grad_output
            return np.ones_like(tensor.data) * grad_output,
        return None,
    

class ReLUBackward(FunctionAbstract):
    """
    Gradient computation for ReLU activation.

    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """

        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Compute ReLU mask: (tensor.data > 0).astype(np.float32)
           - Multiply grad_output by mask: grad_output * relu_grad
           - Return as tuple: (result,)
        3. Else return (None,)

        - ReLU derivative: 1 if x > 0, else 0
        - Use boolean mask: tensor.data > 0
        - Convert to float32 for gradient computation
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # ReLU gradient: 1 if x > 0, else 0
            relu_grad = (tensor.data > 0).astype(np.float32)
            return grad_output * relu_grad,
        return None,


class SigmoidBackward(FunctionAbstract):
    """
    Gradient computation for sigmoid activation.

    Sigmoid: σ(x) = 1/(1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))
    """

    def __init__(self, input_tensor, output_tensor):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """

        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Use saved output: σ(x) = self.output_data
           - Compute sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
           - Multiply by grad_output: grad_output * sigmoid_grad
           - Return as tuple: (result,)
        3. Else return (None,)

        - Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
        - Output is already computed and saved in self.output_data
        - This avoids recomputing sigmoid during backward pass
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # σ'(x) = σ(x) * (1 - σ(x))
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return grad_output * sigmoid_grad,
        return None,


class SoftmaxBackward(FunctionAbstract):
    """
    Gradient computation for softmax activation.

    Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x))
    Derivative: ∂softmax/∂x[i] = softmax[i] * (δ[i,j] - softmax[j])

    For gradient computation:
    grad_x[i] = softmax[i] * (grad_y[i] - sum(grad_y * softmax))

    **Key Insight:** The gradient depends on all elements of softmax due to
    the normalization, not just the element being differentiated.
    """

    def __init__(self, input_tensor, output_tensor, dim=-1):

        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_output):
        """
        Compute gradient for softmax.

        Mathematical formula:
        ∂L/∂x[i] = softmax[i] * (∂L/∂y[i] - sum_j(∂L/∂y[j] * softmax[j]))

        This can be vectorized as:
        grad_x = softmax * (grad_y - sum(grad_y * softmax, keepdims=True))

        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Compute sum term: np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)
           - Compute gradient: self.output_data * (grad_output - sum_term)
           - Return as tuple: (grad_x,)
        3. Else return (None,)

        - Softmax gradient depends on all elements due to normalization
        - Use keepdims=True in np.sum to maintain dimensions for broadcasting
        - Vectorized formula: softmax * (grad_output - sum(grad_output * softmax))
        """

        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Compute sum(grad_output * softmax) along the softmax dimension
            sum_term = np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)

            # Softmax gradient: softmax * (grad_output - sum_term)
            grad_x = self.output_data * (grad_output - sum_term)

            return (grad_x,)
        return (None,)
    

class GELUBackward(FunctionAbstract):
    """
    Gradient computation for GELU activation.

    GELU: f(x) = x * Φ(x) where Φ is the CDF of standard normal
    Approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    **Key Insight:** GELU is smoother than ReLU, providing non-zero gradients
    for negative values, which helps training deep networks.
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for GELU.

        Mathematical formula (using approximation):
        ∂gelu/∂x ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * (...)

        Simplified: We compute the derivative numerically or use the formula.

        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
           - Compute tanh approximation components
           - Compute sech² (derivative of tanh)
           - Apply GELU derivative formula
           - Multiply by grad_output
        3. Else return (None,)

        - GELU is smoother than ReLU, providing gradients for negative values
        - Use tanh approximation for numerical stability
        - Formula: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            # GELU derivative approximation
            # Using the tanh approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            x_cubed = x ** 3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out ** 2

            # Derivative: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145 * x ** 2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x * sech_squared * d_tanh_arg

            return (grad_output * gelu_grad,)
        return (None,)


class MSEBackward(FunctionAbstract):
    """
    Gradient computation for Mean Squared Error Loss.

    MSE: L = mean((predictions - targets)²)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """

        1. Extract predictions tensor from self.saved_tensors
        2. If predictions requires gradients:
           - Compute difference: predictions.data - self.targets_data
           - Apply MSE derivative: 2 * difference / N
           - Multiply by grad_output: grad * grad_output
           - Return as tuple: (result,)
        3. Else return (None,)

        - MSE derivative: ∂MSE/∂pred = 2 * (pred - target) / N
        - N = self.num_samples (total number of elements)
        - Multiply by grad_output for chain rule
        """
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            # Gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples

            return grad * grad_output,
        return None,


class BCEBackward(FunctionAbstract):
    """
    Gradient computation for Binary Cross-Entropy Loss.

    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        1. Extract predictions tensor from self.saved_tensors
        2. If predictions requires gradients:
           - Clip predictions: p = np.clip(predictions.data, eps, 1-eps)
           - Get targets: y = self.targets_data
           - Apply BCE derivative: (p - y) / (p * (1-p) * N)
           - Multiply by grad_output
           - Return as tuple: (result,)
        3. Else return (None,)

        - BCE derivative: ∂BCE/∂p = (p - y) / (p * (1-p)) per sample
        - Clip predictions to avoid log(0) instability
        - Divide by N for mean loss
        """
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.targets_data

            # Gradient: (p - y) / (p * (1-p) * N)
            grad = (p - y) / (p * (1 - p) * self.num_samples)

            return grad * grad_output,
        return None,


class CrossEntropyBackward(FunctionAbstract):
    """
    Gradient computation for Cross-Entropy Loss.

    CrossEntropy: L = -mean(log_softmax(logits)[targets])

    The gradient with respect to logits is remarkably elegant:
    ∂L/∂logits = (softmax(logits) - one_hot(targets)) / N

    This is one of the most beautiful results in machine learning:
    - The gradient is simply the difference between predictions and targets
    - It naturally scales with how wrong we are
    - It's numerically stable when computed via softmax
    """

    def __init__(self, logits, targets):
        """Initialize with logits and target class indices."""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        """
        1. Extract logits tensor from self.saved_tensors
        2. If logits requires gradients:
           - Compute stable softmax: subtract max, exponentiate, normalize
           - Create one-hot encoding of targets
           - Apply CE derivative: (softmax - one_hot) / batch_size
           - Multiply by grad_output
           - Return as tuple: (result,)
        3. Else return (None,)

        - CE gradient: (softmax(logits) - one_hot(targets)) / batch_size
        - This is one of the most elegant gradients in ML!
        - Use stable softmax: subtract max before exp
        - Create one_hot: zeros array, set target indices to 1.0
        """

        logits, = self.saved_tensors

        if isinstance(logits, Tensor) and logits.requires_grad:
            # Compute softmax probabilities
            # Using stable softmax: subtract max for numerical stability
            logits_data = logits.data
            max_logits = np.max(logits_data, axis=1, keepdims=True)
            exp_logits = np.exp(logits_data - max_logits)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Create one-hot encoding of targets
            one_hot = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
            one_hot[np.arange(self.batch_size), self.targets_data] = 1.0

            # Gradient: (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size

            return grad * grad_output,
        return None,


def enable_autograd(quiet=False):
    """
    Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    Call this once to activate gradients globally.

    **Args:**
        quiet (bool): If True, suppress status messages. Default: False.

    **What it does:**
    - Replaces Tensor operations with gradient-tracking versions
    - Adds backward() method for reverse-mode differentiation
    - Enables computation graph building
    - Maintains full backward compatibility

    **After calling this:**
    - Tensor operations will track computation graphs
    - backward() method becomes available
    - Gradients will flow through operations
    - requires_grad=True enables tracking per tensor

    **Example:**
    ```python
    enable_autograd()  # Call once
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    y.backward()
    print(x.grad)  # [3.0]
    ```
    """

    # Educational Note: hasattr() is LEGITIMATE here because:
    # 1. This is a runtime monkey-patch system (meta-programming)
    # 2. We're checking if a class has been dynamically modified
    # 3. _autograd_enabled is a marker attribute we add at runtime
    # This is the CORRECT use of hasattr() for dynamic class modification
    if hasattr(Tensor, '_autograd_enabled'):
        # Silently return if already enabled - no need to warn
        return

    # ===== STEP 1: Add gradient infrastructure to Tensor =====
    # Store original __init__ to extend it
    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        """Extended Tensor init that supports gradient tracking."""
        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None

    # Replace __init__ with gradient-aware version
    Tensor.__init__ = gradient_aware_init

    # Store original operations
    # These are guaranteed to exist from Module 01 (Tensor class)
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__

    # These methods are also guaranteed from Module 01 - trust Single Tensor Class
    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    # Helper to safely check requires_grad (handles tensors created before enable_autograd)
    def _get_requires_grad(tensor):
        """Safely get requires_grad, defaulting to False for pre-autograd tensors."""
        return getattr(tensor, 'requires_grad', False) if isinstance(tensor, Tensor) else False

    def _ensure_grad_attrs(tensor):
        """Ensure tensor has gradient attributes (for tensors created before enable_autograd)."""
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, 'requires_grad'):
                tensor.requires_grad = False
            if not hasattr(tensor, 'grad'):
                tensor.grad = None

    # Enhanced operations that track gradients
    def tracked_add(self, other):
        """
        Addition with gradient tracking.

        Enhances the original __add__ method to build computation graphs
        when requires_grad=True for any input.
        """
        # Ensure self has gradient attributes
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_add(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result

    def tracked_mul(self, other):
        """
        Multiplication with gradient tracking.

        Enhances the original __mul__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed for consistency
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        # Call original operation
        result = _original_mul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)

        return result

    def tracked_matmul(self, other):
        """
        Matrix multiplication with gradient tracking.

        Enhances the original matmul method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        # Call original matmul from Module 01
        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MatMulBackward(self, other)

        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        """
        Transpose with gradient tracking.

        Enhances the original transpose method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original transpose from Module 01
        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)

        return result

    def tracked_reshape(self, *shape):
        """
        Reshape with gradient tracking.

        Enhances the original reshape method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)
        original_shape = self.shape

        # Call original reshape from Module 01
        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)

        return result

    def tracked_sub(self, other):
        """
        Subtraction with gradient tracking.

        Enhances the original __sub__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_sub(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other)

        return result

    def tracked_div(self, other):
        """
        Division with gradient tracking.

        Enhances the original __truediv__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Convert scalar to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        # Call original operation
        result = _original_div(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other)

        return result

    def tracked_getitem(self, key):
        """
        Indexing/slicing with gradient tracking.

        Enhances the original __getitem__ method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original __getitem__ from Module 01
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)

        return result

    def sum_op(self, axis=None, keepdims=False):
        """
        Sum operation with gradient tracking.

        Creates a new sum method that builds computation graphs
        when requires_grad=True.
        """
        _ensure_grad_attrs(self)

        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SumBackward(self)

        return result

    def backward(self, gradient=None):
        """
        Compute gradients via backpropagation.

        This is the key method that makes training possible!
        It implements reverse-mode automatic differentiation.

        **Algorithm:**
        1. Initialize gradient if not provided (for scalar outputs)
        2. Accumulate gradient in self.grad
        3. If this tensor has a _grad_fn, call it to propagate gradients
        4. Recursively call backward() on parent tensors

        **Example:**
        ```python
        x = Tensor([2.0], requires_grad=True)
        y = x * 3
        y.backward()  # Computes gradients for x
        print(x.grad)  # [3.0]
        ```
        """
        # Ensure gradient attributes exist
        _ensure_grad_attrs(self)

        # Only compute gradients if required
        if not _get_requires_grad(self):
            return

        # Initialize gradient if not provided (for scalar outputs)
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f"backward() called on non-scalar tensor without gradient argument.\n"
                    f"  Tensor shape: {self.shape}\n"
                    f"  Issue: For non-scalar outputs, you must provide the gradient from the next layer.\n"
                    f"  Fix: Call backward(gradient) with the gradient tensor from the loss function."
                )

        # Initialize or accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # Convert scalar gradient to numpy array if needed
        if not isinstance(gradient, np.ndarray):
            gradient = np.array(gradient, dtype=np.float32)

        # Handle broadcasting: sum gradient to match self.data shape
        # This happens when operations broadcast tensors (e.g., adding bias to batch)
        if gradient.shape != self.grad.shape:
            # Step 1: Remove extra leading dimensions added during forward pass
            # Example: gradient (batch_size, features) → self.grad (features,)
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)

            # Step 2: Sum over dimensions that were size-1 in original tensor
            # Example: bias with shape (1,) broadcast to (batch_size,) during forward
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)

        self.grad += gradient

        # Propagate gradients through computation graph
        # _grad_fn is set by autograd enhancement when tensor is created from an operation
        grad_fn = getattr(self, '_grad_fn', None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad)

    def zero_grad(self):
        """
        Reset gradients to zero.

        Call this before each backward pass to prevent gradient accumulation
        from previous iterations.
        """
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__sub__ = tracked_sub
    Tensor.__mul__ = tracked_mul
    Tensor.__truediv__ = tracked_div
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Patch activations and losses to track gradients
    try:
        from tinytorch.core.activations import Sigmoid, ReLU, Softmax, GELU
        from tinytorch.core.losses import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss

        # Store original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking."""
            result_data = 1.0 / (1.0 + np.exp(-x.data))
            result = Tensor(result_data)

            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)

            return result

        def tracked_relu_forward(self, x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)

            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)

            return result

        def tracked_softmax_forward(self, x, dim=-1):
            """Softmax with gradient tracking."""
            # Call original forward to get result using Tensor operations
            result = _original_softmax_forward(self, x, dim=dim)

            # Attach the correct gradient function
            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SoftmaxBackward(x, result, dim)

            return result

        def tracked_gelu_forward(self, x):
            """GELU with gradient tracking."""
            # Call original forward to get result
            result = _original_gelu_forward(self, x)

            # Attach the correct gradient function
            if x.requires_grad:
                result.requires_grad = True
                result._grad_fn = GELUBackward(x)

            return result

        def tracked_bce_forward(self, predictions, targets):
            """Binary cross-entropy with gradient tracking."""
            # Compute BCE loss
            eps = EPSILON
            clamped_preds = np.clip(predictions.data, eps, 1 - eps)
            log_preds = np.log(clamped_preds)
            log_one_minus_preds = np.log(1 - clamped_preds)
            bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)
            bce_loss = np.mean(bce_per_sample)

            result = Tensor(bce_loss)

            if predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)

            return result

        def tracked_mse_forward(self, predictions, targets):
            """MSE loss with gradient tracking."""
            # Compute MSE loss
            diff = predictions.data - targets.data
            squared_diff = diff ** 2
            mse = np.mean(squared_diff)

            result = Tensor(mse)

            if predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)

            return result

        def tracked_ce_forward(self, logits, targets):
            """Cross-entropy loss with gradient tracking."""
            from tinytorch.core.losses import log_softmax

            # Compute log-softmax for numerical stability
            log_probs = log_softmax(logits, dim=-1)

            # Select log-probabilities for correct classes
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

            # Return negative mean
            ce_loss = -np.mean(selected_log_probs)

            result = Tensor(ce_loss)

            if logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)

            return result

        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward

    except ImportError:
        # Activations/losses not yet available (happens during module development)
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True

    if not quiet:
        print("✅ Autograd enabled! Tensors now track gradients.")
        print("   - Operations build computation graphs")
        print("   - backward() computes gradients")
        print("   - requires_grad=True enables tracking")

# Auto-enable when module is imported
# Always quiet to avoid cluttering user imports
import os
enable_autograd(quiet=True)