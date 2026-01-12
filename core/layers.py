import numpy as np

from core.tensor import Tensor
from core.activations import ReLU, Sigmoid

from core.abstracts import LayerAbstract


# Constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0  # Xavier/Glorot initialization uses sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


class Linear(LayerAbstract):
    """
    Linear (fully connected) layer: y = xW + b

    This is the fundamental building block of neural networks.
    Applies a linear transformation to incoming data.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize linear layers with proper weight initialization

        1. Create weight matrix (in_features, out_features) with Xavier scaling
        2. Create bias vector (out_features,) initialized to zeros if bias=True
        3. Store as Tensor objects for use in forward pass

        - Xavier init: scale = sqrt(1/in_features)
        - Use np.random.randn() for normal distribution
        - bias=None when bias=False

        """

        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(XAVIER_SCALE_FACTOR / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        """
        1. Matrix multiply input with weights: xW
        2. Add bias if it exists
        3. Return result as new Tensor

        - Use tensor.matmul() for matrix multiplication
        - Handle bias=None case
        - Broadcasting automatically handles bias addition
        """

        output = x.matmul(self.weight)

        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def parameters(self):
        """
        Return list of trainable parameters.

        1. Start with weight (always present)
        2. Add bias if it exists
        3. Return as list for optimizer

        - Create list starting with self.weight
        - Check if self.bias is not None before appending
        - Return the complete list
        """

        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def __repr__(self):
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"


class Dropout(LayerAbstract):
        """
        Dropout layer for regularization.

        During training: randomly zeros elements with probability p, scales survivors by 1/(1-p)
        During inference: passes input through unchanged

        This prevents overfitting by forcing the network to not rely on specific neurons.
        """

        def __init__(self, p=0.5):
            """`
            1. Validate p is between 0.0 and 1.0 (inclusive)
            2. Raise ValueError if out of range
            3. Store p as instance attribute

            - Use DROPOUT_MIN_PROB and DROPOUT_MAX_PROB constants for validation
        - Check: DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB
        - Raise descriptive ValueError if inv

            """

            if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:

                raise ValueError(f"Dropout probability must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}, got {p}")
            
            self.p = p
        
        def forward(self, x, training=True):
            """
            Forward pass through dropout layer.

            During training: randomly zeros elements with probability p, scales survivors by 1/(1-p)
            During inference: passes input through unchanged

            This prevents overfitting by forcing the network to not rely on specific neurons.

            1. If training=False or p=0, return input unchanged
            2. If p=1, return zeros
            3. Otherwise: create random mask, apply it, scale by 1/(1-p)

            - Use np.random.random() < keep_prob for mask
            - Scale by 1/(1-p) to maintain expected value
            - training=False should return input unchanged

            """

            if not training or self.p == DROPOUT_MIN_PROB:
                return x
            
            if self.p == DROPOUT_MAX_PROB:
                return Tensor(np.zeros_like(x.data))
            
            keep_prob = 1.0 - self.p
            mask = np.random.random(x.data.shape) < keep_prob

            mask_tensor = Tensor(mask.astype(np.float32))
            scale = Tensor(np.array(1.0 / keep_prob, dtype=np.float32))
            output = x * mask_tensor * scale

            return output

        def parameters(self):
            """Dropout has no trainable parameters."""
            return []
        
        def __repr__(self):
            return f"Dropout(p={self.p})"


class Sequential(LayerAbstract):
    """
    Container for stacking multiple layers sequentially.

    Applies each layer in order during forward pass.
    Useful for building feedforward networks.
    """

    def __init__(self, *layers: Linear):
        """
        Initialize with layers to chain together
        """

        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward()
        return x
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
    
    def __repr__(self):
        layer_reprs = ",".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"

