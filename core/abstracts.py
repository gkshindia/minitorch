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


class LayerAbstract(ABC):
    """Abstract base class for neural network layers.
    
    All layers inherit from this class and must implement:
    - forward(): Compute output from input tensor
    - backward(): Compute gradients for backpropagation
    - parameters(): Return learnable parameters
    
    Architecture:
    ┌─────────────────────────────────────┐
    │ Layer (Abstract Base)               │
    ├─────────────────────────────────────┤
    │ Methods:                            │
    │ • forward(x: Tensor) → Tensor      │
    │ • backward(grad: Tensor) → Tensor  │
    │ • parameters() → List[Tensor]      │
    ├─────────────────────────────────────┤
    │ Implementations:                    │
    │ • Linear                            │
    │ • Dropout (Regularization)         │
    └─────────────────────────────────────┘
    
    Use Cases:
    - Building blocks for neural networks
    - Feature extraction, transformation, and learning
    """
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer Compute output from input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
            
        Raises:
            NotImplementedError: Subclass must implement
        """
        pass

    def __call__(self, x, *args, **kwargs):
        """Convenience method for forward pass."""
        return self.forward(x, *args, **kwargs)
    
    def parameters(self):
        """Return list of learnable parameters."""
        return []
    
    def __repr__(self):
        """String representation of layer."""
        return f"{self.__class__.__name__}()"


class DatasetAbstract(ABC):
    """
    Provides the fundamental interface that all datasets must implement:
    - __len__(): Returns the total number of samples
    - __getitem__(idx): Returns the sample at given index
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class FunctionAbstract(ABC):
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.

    **Key Concepts:**
    - **saved_tensors**: Store inputs needed for backward pass
    - **apply()**: Compute gradients using chain rule
    - **next_functions**: Track computation graph connections
    """

    def __init__(self, *tensors):
        self.saved_tensors = tensors
        self.next_functions = []

        for t in tensors:
            if isinstance(t, Tensor) and t.requires_grad:
                if getattr(t, '_grad_fn', None) is not None:
                    self.next_functions.append(t._grad_fn)
    
    def apply(self, grad_output):
        raise NotImplementedError("Each function must implement an apply method.")


class OptimizerAbstract(ABC):
    """Abstract base class for optimizers.
    
    All optimizers inherit from this class and must implement:
    - step(): Update parameters based on gradients
    - zero_grad(): Clear gradients of all parameters
    
    Architecture:
    ┌─────────────────────────────────────┐
    │ Optimizer (Abstract Base)          │
    ├─────────────────────────────────────┤
    │ Methods:                            │
    │ • step()                           │
    │ • zero_grad()                      │
    ├─────────────────────────────────────┤
    │ Implementations:                    │
    │ • SGD                              │
    │ • Adam                             │
    │ • AdamW                            │
    └─────────────────────────────────────┘
    
    Use Cases:
    - Training neural networks by updating weights based on computed gradients.

        1. Store parameters as a list for iteration
        2. Validate that all parameters require gradients
        3. Initialize step counter for algorithms that need it
    """
    
    def __init__(self, parameters):
        """Initialize optimizer with model parameters.
        
        Args:
            parameters: Iterable of model parameters to optimize
        """
        if not isinstance(parameters, list):
            parameters = list(parameters)
        
        self.parameters = parameters
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        """Update parameters based on gradients.
        
        Raises:
            NotImplementedError: Subclass must implement
        """
        raise NotImplementedError("Subclass must implememnt step()")
    
    @abstractmethod
    def zero_grad(self):
        """Clear gradients of all parameters. Update parameters based on gradients
        
        Raises:
            NotImplementedError: Subclass must implement
        """
        for parameter in self.parameters:
            parameter.grad = None