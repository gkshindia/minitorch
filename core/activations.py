import numpy as np
from typing import Optional
from core.abstracts import ActivationAbstract
from core.tensor import Tensor

TOLERANCE = 1e-10 # small tolerance for floating point comparisons in tests
__all__ = ["Sigmoid", "ReLU", "Tanh", "Gelu", "Softmax"]


class Sigmoid(ActivationAbstract):

    """
    Sigmoid activation: σ(x) = 1/(1 + e^(-x))
    Maps any real number to (0,1) range.
    Perfect for probabilityies and binary classification.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply signoid activation elementwise

        1. Apply sigmoid formula: 1 / (1 + exp(-x))
        2. Use np.exp for exponential
        3. Return result wrapped in new Tensor

        use np.exp(-x.data) for numerical stability

        For x ≥ 0:  Use σ(x) = 1 / (1 + e^(-x))
            Safe because e^(-x) is small

        For x < 0:  Use σ(x) = e^x / (1 + e^x)  
            Safe because e^x is small

        """
        # Clipping at ±500 ensures exp() stays within float64 range
        z = np.clip(x.data, -500, 500)
        result_data = np.zeros_like(z)

        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)
    
    def backward(self, grad_output):
        pass


class ReLU(ActivationAbstract):
    """
    ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.
    Most popular activation for hidden layers.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        1. Use np.maximum(0, x.data) for element-wise max with zero
        2. Return result wrapped in new Tensor

        np.maximum handles element-wise maximum automatically
        """

        result = np.maximum(0, x.data)
        return Tensor(result)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        pass


class Tanh(ActivationAbstract):
    """
    Tanh activation: f(x) = (e^x - e^(-x))/(e^x + e^(-x))

    Maps any real number to (-1, 1) range.
    Zero-centered alternative to sigmoid.
    """

    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        pass


class GELU(ActivationAbstract):
    """
    GELU activation: f(x) = x * Φ(x) ≈ x * Sigmoid(1.702 * x)

    The 1.702 constant comes from √(2/π) approximation

    Smooth approximation to ReLU, used in modern transformers.
    Where Φ(x) is the cumulative distribution function of standard normal.

    GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715x³)))

    GELU has no-zero gradients even for negative inputs. 

    GELU(x) = x · P(X ≤ x)  where X ~ N(0,1)

    Translation:
    "Multiply input by the probability that a random 
    normal variable is less than it"

    For x = 2:   P(X ≤ 2) ≈ 0.977  → keep most of it
    For x = 0:   P(X ≤ 0) = 0.5    → keep half
    For x = -2:  P(X ≤ -2) ≈ 0.023 → mostly suppress

    stochastic regularization . 
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        1. Use approximation: x * sigmoid(1.702 * x)
        2. Compute sigmoid part: 1 / (1 + exp(-1.702 * x))
        3. Multiply by x element-wise
        4. Return result wrapped in new Tensor

        """
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        result = x.data * sigmoid_part

        return Tensor(result)
    
    def backward(self, grad_output: Tensor) -> Tensor:
        pass
