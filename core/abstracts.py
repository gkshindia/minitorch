from abc import ABC, abstractmethod
from core.tensor import Tensor


class ActivationAbstract(ABC):
    """Abstract base class for activation functions.
    
    All activation functions inherit from this class and must implement:
    - forward(): Apply activation to input tensor
    - backward(): Compute gradient for backpropagation
    - __call__(): Convenience method for forward pass
    
    Architecture:
    ┌─────────────────────────────────────┐
    │ Activation (Abstract Base)          │
    ├─────────────────────────────────────┤
    │ Methods:                            │
    │ • forward(x: Tensor) → Tensor      │
    │ • backward(grad: Tensor) → Tensor  │
    │ • __call__(x: Tensor) → Tensor     │
    ├─────────────────────────────────────┤
    │ Implementations:                    │
    │ • Sigmoid (S-curve, 0-1 range)     │
    │ • ReLU (max(0, x), sparse)         │
    │ • Tanh (S-curve, -1 to 1)          │
    │ • Gelu (smooth approximation)      │
    │ • Softmax (probability distribution)│
    └─────────────────────────────────────┘
    
    Use Cases:
    - Hidden layers: ReLU, Gelu (non-saturation, sparse gradients)
    - Output layer (binary): Sigmoid
    - Output layer (multiclass): Softmax
    - Recurrent layers: Tanh (bounded outputs)
    """
    
    def __init__(self):
        """Initialize activation function."""
        self.input_cache = None  # Store input for backward pass

    def parameters(self):
        """Return empty list(activations have no learnable parameters)"""
        return []
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply activation function to input.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor with same shape as input
            
        Raises:
            NotImplementedError: Subclass must implement
        """
        pass
    
    @abstractmethod
    def backward(self, grad_output: Tensor) -> Tensor:
        """Compute gradient for backpropagation.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
            
        Raises:
            NotImplementedError: Subclass must implement
        """
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        """Convenience method for forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return self.forward(x)
    
    def __repr__(self):
        """String representation of activation."""
        return f"{self.__class__.__name__}()"
