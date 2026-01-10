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
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, grad_output):
        pass

