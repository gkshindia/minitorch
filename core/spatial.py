import numpy as np

from core.tensor import Tensor

DEFAULT_KERNEL_SIZE = 3
DEFAULT_STRIDE = 1
DEFAULT_PADDING = 0


class Conv2d:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        1. Store hyperparameters (channels, kernel_size, stride, padding)
        2. Initialize weights using He initialization for ReLU compatibility
        3. Initialize bias (if enabled) to zeros
        4. Use proper shapes: weight (out_channels, in_channels, kernel_h, kernel_w)

        WEIGHT INITIALIZATION:
        - He init: std = sqrt(2 / (in_channels * kernel_h * kernel_w))
        - This prevents vanishing/exploding gradients with ReLU

        Convert kernel_size to tuple if it's an integer
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        kernel_h, kernel_w = self.kernel_size
        fan_in = in_channels * kernel_h * kernel_w
        std = np.sqrt(2.0 / fan_in)

        self.weight = Tensor(np.random.normal(0, std,
                           (out_channels, in_channels, kernel_h, kernel_w)))

        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        """
        1. Extract input dimensions and validate
        2. Calculate output dimensions
        3. Apply padding if needed
        4. Implement 6 nested loops for full convolution
        5. Add bias if present

        - Handle padding by creating padded input array
        - Watch array bounds in inner loops
        - Accumulate products for each output position
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels = self.out_channels
        kernel_h, kernel_w = self.kernel_size

        out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

        if self.padding > 0:
            padded_input = np.pad(x.data,
                                 ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                 mode='constant', constant_values=0)
        else:
            padded_input = x.data

        output = np.zeros((batch_size, out_channels, out_height, out_width))

        # Explicit 6-nested loop convolution to show complexity
        for b in range(batch_size):
            for out_ch in range(out_channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Calculate input region for this output position
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    input_val = padded_input[b, in_ch,
                                                           in_h_start + k_h,
                                                           in_w_start + k_w]
                                    weight_val = self.weight.data[out_ch, in_ch, k_h, k_w]
                                    conv_sum += input_val * weight_val

                        # Store result
                        output[b, out_ch, out_h, out_w] = conv_sum

        if self.bias is not None:
            # Broadcast bias across spatial dimensions
            for out_ch in range(out_channels):
                output[:, out_ch, :, :] += self.bias.data[out_ch]

        return Tensor(output)

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __call__(self, x):
        return self.forward(x)


class MaxPool2d:
    """
    2D Max Pooling layer for spatial dimension reduction.

    Applies maximum operation over spatial windows, preserving
    the strongest activations while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided (non-overlapping)
        3. Store padding parameter

        Default stride equals kernel_size for non-overlapping windows
        """
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Default stride equals kernel_size (non-overlapping)
        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding

    def forward(self, x):
        """
        1. Extract input dimensions
        2. Calculate output dimensions
        3. Apply padding if needed
        4. Implement nested loops for pooling windows
        5. Find maximum value in each window

        - Initialize max_val to negative infinity
        - Handle stride correctly when accessing input
        - No parameters to update (pooling has no weights)
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, channels, in_height, in_width = x.shape
        kernel_h, kernel_w = self.kernel_size

        out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

        if self.padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=-np.inf)
        else:
            padded_input = x.data

        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Calculate input region for this output position
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Find maximum in window
                        max_val = -np.inf
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded_input[b, c,
                                                       in_h_start + k_h,
                                                       in_w_start + k_w]
                                max_val = max(max_val, input_val)

                        output[b, c, out_h, out_w] = max_val

        return Tensor(output)

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class AvgPool2d:
    """
    2D Average Pooling layer for spatial dimension reduction.

    Applies average operation over spatial windows, smoothing
    features while reducing computational load.

    Args:
        kernel_size: Size of pooling window (int or tuple)
        stride: Stride of pooling operation (default: same as kernel_size)
        padding: Zero-padding added to input (default: 0)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        """
        1. Convert kernel_size to tuple if needed
        2. Set stride to kernel_size if not provided
        3. Store padding parameter
        """
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride

        self.padding = padding

    def forward(self, x):
        """
        1. Similar structure to MaxPool2d
        2. Instead of max, compute average of window
        3. Divide sum by window area for true average

        LOOP STRUCTURE:
        for batch in range(batch_size):
            for channel in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        # Compute average in window
                        window_sum = 0
                        for k_h in range(kernel_height):
                            for k_w in range(kernel_width):
                                window_sum += input[...]
                        avg_val = window_sum / (kernel_height * kernel_width)

        Remember to divide by window area to get true average
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, channels, in_height, in_width = x.shape
        kernel_h, kernel_w = self.kernel_size

        out_height = (in_height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_w) // self.stride + 1

        if self.padding > 0:
            padded_input = np.pad(x.data,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = x.data

        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h * self.stride
                        in_w_start = out_w * self.stride

                        # Compute sum in window
                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded_input[b, c,
                                                       in_h_start + k_h,
                                                       in_w_start + k_w]
                                window_sum += input_val

                        # Compute average
                        avg_val = window_sum / (kernel_h * kernel_w)

                        output[b, c, out_h, out_w] = avg_val

        # Return Tensor with gradient tracking (consistent with MaxPool2d)
        result = Tensor(output, requires_grad=x.requires_grad)
        return result

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class BatchNorm2d:
    """
    Batch Normalization for 2D spatial inputs (images).

    Normalizes activations across batch and spatial dimensions for each channel,
    then applies learnable scale (gamma) and shift (beta) parameters.

    Key behaviors:
    - Training: Uses batch statistics, updates running statistics
    - Eval: Uses frozen running statistics for consistent inference

    Args:
        num_features: Number of channels (C in NCHW format)
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics update (default: 0.1)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        1. Store hyperparameters (num_features, eps, momentum)
        2. Initialize gamma (scale) to ones - identity at start
        3. Initialize beta (shift) to zeros - no shift at start
        4. Initialize running_mean to zeros
        5. Initialize running_var to ones
        6. Set training mode to True initially
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters (requires_grad=True for training)
        # gamma (scale): initialized to 1 so output = normalized input initially
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        # beta (shift): initialized to 0 so no shift initially
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        # Running statistics (not trained, accumulated during training)
        # These are used during evaluation for consistent normalization
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, x):
        """
        1. Validate input shape (must be 4D: batch, channels, height, width)
        2. If training:
           a. Compute batch mean and variance per channel
           b. Normalize using batch statistics
           c. Update running statistics with momentum
        3. If eval:
           a. Use running mean and variance
           b. Normalize using frozen statistics
        4. Apply scale (gamma) and shift (beta)

        - Compute mean/var over axes (0, 2, 3) to get per-channel statistics
        - Reshape gamma/beta to (1, C, 1, 1) for broadcasting
        - Running stat update: running = (1 - momentum) * running + momentum * batch
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.shape}")

        batch_size, channels, height, width = x.shape

        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, got {channels}")

        if self.training:
            # Compute batch statistics per channel
            # Mean over batch and spatial dimensions: axes (0, 2, 3)
            batch_mean = np.mean(x.data, axis=(0, 2, 3))  # Shape: (C,)
            batch_var = np.var(x.data, axis=(0, 2, 3))    # Shape: (C,)

            # Update running statistics (exponential moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics (frozen during eval)
            mean = self.running_mean
            var = self.running_var

        mean_reshaped = mean.reshape(1, channels, 1, 1)
        var_reshaped = var.reshape(1, channels, 1, 1)

        x_normalized = (x.data - mean_reshaped) / np.sqrt(var_reshaped + self.eps)

        gamma_reshaped = self.gamma.data.reshape(1, channels, 1, 1)
        beta_reshaped = self.beta.data.reshape(1, channels, 1, 1)

        output = gamma_reshaped * x_normalized + beta_reshaped

        result = Tensor(output, requires_grad=x.requires_grad or self.gamma.requires_grad)

        return result

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, x):
        return self.forward(x)
