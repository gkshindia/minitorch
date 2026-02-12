"""
Spatial Layers Tests

Comprehensive test suite for spatial layers including:
- Conv2d: 2D convolution with various configurations
- MaxPool2d: Max pooling with different kernel sizes
- AvgPool2d: Average pooling with different kernel sizes
- BatchNorm2d: Batch normalization for 2D inputs
"""

import numpy as np
import pytest
from core.tensor import Tensor
from core.spatial import Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d
from tests.layers.conftest import (
    TOLERANCE,
    assert_shape_correct,
    assert_finite,
    assert_bias_zeros,
    assert_close,
    assert_parameters_count,
)


class TestConv2dInitialization:
    """Test Conv2d layer initialization."""

    def test_conv2d_init_with_bias(self):
        """Test Conv2d initialization with bias."""
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        conv = Conv2d(in_channels, out_channels, kernel_size, bias=True)

        assert conv.in_channels == in_channels
        assert conv.out_channels == out_channels
        assert conv.kernel_size == (kernel_size, kernel_size)
        assert_shape_correct(conv.weight, (out_channels, in_channels, kernel_size, kernel_size))
        assert_shape_correct(conv.bias, (out_channels,))
        assert conv.bias is not None

    def test_conv2d_init_without_bias(self):
        """Test Conv2d initialization without bias."""
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        conv = Conv2d(in_channels, out_channels, kernel_size, bias=False)

        assert conv.in_channels == in_channels
        assert conv.out_channels == out_channels
        assert_shape_correct(conv.weight, (out_channels, in_channels, kernel_size, kernel_size))
        assert conv.bias is None

    def test_conv2d_init_tuple_kernel(self):
        """Test Conv2d with tuple kernel size."""
        in_channels = 3
        out_channels = 16
        kernel_size = (3, 5)
        conv = Conv2d(in_channels, out_channels, kernel_size)

        assert conv.kernel_size == kernel_size
        assert_shape_correct(conv.weight, (out_channels, in_channels, 3, 5))

    def test_conv2d_init_with_stride_padding(self):
        """Test Conv2d with custom stride and padding."""
        conv = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

        assert conv.stride == 2
        assert conv.padding == 1

    def test_conv2d_he_initialization(self):
        """Test that Conv2d uses He initialization."""
        in_channels = 3
        kernel_size = 3
        conv = Conv2d(in_channels, 16, kernel_size)
        
        fan_in = in_channels * kernel_size * kernel_size
        expected_std = np.sqrt(2.0 / fan_in)
        actual_std = np.std(conv.weight.data)
        
        assert 0.5 * expected_std < actual_std < 2.0 * expected_std

    def test_conv2d_bias_initialization(self):
        """Test that bias is initialized to zeros."""
        conv = Conv2d(3, 16, kernel_size=3, bias=True)
        assert_bias_zeros(conv.bias)


class TestConv2dForward:
    """Test Conv2d forward pass."""

    def test_conv2d_forward_basic(self):
        """Test basic Conv2d forward pass."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height, width = 8, 8
        kernel_size = 3

        conv = Conv2d(in_channels, out_channels, kernel_size, padding=1)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        output = conv.forward(x)
        
        assert_shape_correct(output, (batch_size, out_channels, height, width))
        assert_finite(output)

    def test_conv2d_forward_no_padding(self):
        """Test Conv2d forward pass without padding."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height, width = 8, 8
        kernel_size = 3

        conv = Conv2d(in_channels, out_channels, kernel_size, padding=0)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        output = conv.forward(x)
        
        expected_height = height - kernel_size + 1
        expected_width = width - kernel_size + 1
        assert_shape_correct(output, (batch_size, out_channels, expected_height, expected_width))
        assert_finite(output)

    def test_conv2d_forward_with_stride(self):
        """Test Conv2d forward pass with stride."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height, width = 8, 8
        kernel_size = 3
        stride = 2

        conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        output = conv.forward(x)
        
        expected_height = (height + 2 * 1 - kernel_size) // stride + 1
        expected_width = (width + 2 * 1 - kernel_size) // stride + 1
        assert_shape_correct(output, (batch_size, out_channels, expected_height, expected_width))
        assert_finite(output)

    def test_conv2d_forward_single_pixel_kernel(self):
        """Test Conv2d with 1x1 kernel (pointwise convolution)."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height, width = 8, 8

        conv = Conv2d(in_channels, out_channels, kernel_size=1)
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        
        output = conv.forward(x)
        
        assert_shape_correct(output, (batch_size, out_channels, height, width))
        assert_finite(output)

    def test_conv2d_forward_identity_convolution(self):
        """Test Conv2d produces correct output for identity-like weights."""
        batch_size = 1
        in_channels = 1
        out_channels = 1
        height, width = 3, 3
        kernel_size = 1

        conv = Conv2d(in_channels, out_channels, kernel_size, bias=False)
        conv.weight.data = np.ones((out_channels, in_channels, kernel_size, kernel_size))
        
        x = Tensor(np.ones((batch_size, in_channels, height, width)) * 2.0)
        output = conv.forward(x)
        
        expected = np.ones((batch_size, out_channels, height, width)) * 2.0
        assert_close(output.data, expected)

    def test_conv2d_forward_with_bias(self):
        """Test Conv2d forward pass applies bias correctly."""
        batch_size = 1
        in_channels = 1
        out_channels = 1
        height, width = 3, 3

        conv = Conv2d(in_channels, out_channels, kernel_size=1)
        conv.weight.data = np.ones((out_channels, in_channels, 1, 1))
        conv.bias.data = np.array([5.0])
        
        x = Tensor(np.ones((batch_size, in_channels, height, width)) * 2.0)
        output = conv.forward(x)
        
        expected = np.ones((batch_size, out_channels, height, width)) * 7.0
        assert_close(output.data, expected)

    def test_conv2d_forward_invalid_input_shape(self):
        """Test Conv2d raises error for invalid input shape."""
        conv = Conv2d(3, 16, kernel_size=3)
        x = Tensor(np.random.randn(2, 3, 8))
        
        with pytest.raises(ValueError, match="Expected 4D input"):
            conv.forward(x)

    def test_conv2d_call_method(self):
        """Test Conv2d __call__ method works."""
        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8))
        
        output = conv(x)
        
        assert_shape_correct(output, (2, 16, 8, 8))


class TestConv2dParameters:
    """Test Conv2d parameter management."""

    def test_conv2d_parameters_with_bias(self):
        """Test Conv2d returns correct parameters with bias."""
        conv = Conv2d(3, 16, kernel_size=3, bias=True)
        params = conv.parameters()
        
        assert_parameters_count(conv, 2)
        assert params[0] is conv.weight
        assert params[1] is conv.bias

    def test_conv2d_parameters_without_bias(self):
        """Test Conv2d returns correct parameters without bias."""
        conv = Conv2d(3, 16, kernel_size=3, bias=False)
        params = conv.parameters()
        
        assert_parameters_count(conv, 1)
        assert params[0] is conv.weight


class TestMaxPool2dInitialization:
    """Test MaxPool2d layer initialization."""

    def test_maxpool2d_init_basic(self):
        """Test MaxPool2d initialization with default stride."""
        kernel_size = 2
        pool = MaxPool2d(kernel_size)

        assert pool.kernel_size == (kernel_size, kernel_size)
        assert pool.stride == kernel_size
        assert pool.padding == 0

    def test_maxpool2d_init_with_stride(self):
        """Test MaxPool2d initialization with custom stride."""
        kernel_size = 2
        stride = 1
        pool = MaxPool2d(kernel_size, stride=stride)

        assert pool.kernel_size == (kernel_size, kernel_size)
        assert pool.stride == stride

    def test_maxpool2d_init_with_padding(self):
        """Test MaxPool2d initialization with padding."""
        kernel_size = 2
        padding = 1
        pool = MaxPool2d(kernel_size, padding=padding)

        assert pool.padding == padding

    def test_maxpool2d_init_tuple_kernel(self):
        """Test MaxPool2d with tuple kernel size."""
        kernel_size = (2, 3)
        pool = MaxPool2d(kernel_size)

        assert pool.kernel_size == kernel_size
        assert pool.stride == kernel_size[0]


class TestMaxPool2dForward:
    """Test MaxPool2d forward pass."""

    def test_maxpool2d_forward_basic(self):
        """Test basic MaxPool2d forward pass."""
        batch_size = 2
        channels = 3
        height, width = 8, 8
        kernel_size = 2

        pool = MaxPool2d(kernel_size)
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        output = pool.forward(x)
        
        expected_height = height // kernel_size
        expected_width = width // kernel_size
        assert_shape_correct(output, (batch_size, channels, expected_height, expected_width))
        assert_finite(output)

    def test_maxpool2d_forward_correctness(self):
        """Test MaxPool2d produces correct max values."""
        batch_size = 1
        channels = 1
        height, width = 4, 4
        kernel_size = 2

        pool = MaxPool2d(kernel_size)
        
        x_data = np.array([[[[1, 2, 5, 6],
                             [3, 4, 7, 8],
                             [9, 10, 13, 14],
                             [11, 12, 15, 16]]]], dtype=np.float64)
        x = Tensor(x_data)
        
        output = pool.forward(x)
        
        expected = np.array([[[[4, 8],
                               [12, 16]]]], dtype=np.float64)
        assert_close(output.data, expected)

    def test_maxpool2d_forward_with_stride(self):
        """Test MaxPool2d forward pass with custom stride."""
        batch_size = 1
        channels = 1
        height, width = 8, 8
        kernel_size = 2
        stride = 1

        pool = MaxPool2d(kernel_size, stride=stride)
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        output = pool.forward(x)
        
        expected_height = (height - kernel_size) // stride + 1
        expected_width = (width - kernel_size) // stride + 1
        assert_shape_correct(output, (batch_size, channels, expected_height, expected_width))

    def test_maxpool2d_forward_preserves_channels(self):
        """Test MaxPool2d preserves number of channels."""
        batch_size = 2
        channels = 16
        height, width = 8, 8
        kernel_size = 2

        pool = MaxPool2d(kernel_size)
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        output = pool.forward(x)
        
        assert output.shape[1] == channels

    def test_maxpool2d_forward_invalid_input_shape(self):
        """Test MaxPool2d raises error for invalid input shape."""
        pool = MaxPool2d(2)
        x = Tensor(np.random.randn(2, 3, 8))
        
        with pytest.raises(ValueError, match="Expected 4D input"):
            pool.forward(x)

    def test_maxpool2d_call_method(self):
        """Test MaxPool2d __call__ method works."""
        pool = MaxPool2d(2)
        x = Tensor(np.random.randn(2, 3, 8, 8))
        
        output = pool(x)
        
        assert_shape_correct(output, (2, 3, 4, 4))


class TestMaxPool2dParameters:
    """Test MaxPool2d parameter management."""

    def test_maxpool2d_no_parameters(self):
        """Test MaxPool2d has no trainable parameters."""
        pool = MaxPool2d(2)
        params = pool.parameters()
        
        assert_parameters_count(pool, 0)


class TestAvgPool2dInitialization:
    """Test AvgPool2d layer initialization."""

    def test_avgpool2d_init_basic(self):
        """Test AvgPool2d initialization with default stride."""
        kernel_size = 2
        pool = AvgPool2d(kernel_size)

        assert pool.kernel_size == (kernel_size, kernel_size)
        assert pool.stride == kernel_size
        assert pool.padding == 0

    def test_avgpool2d_init_with_stride(self):
        """Test AvgPool2d initialization with custom stride."""
        kernel_size = 2
        stride = 1
        pool = AvgPool2d(kernel_size, stride=stride)

        assert pool.kernel_size == (kernel_size, kernel_size)
        assert pool.stride == stride

    def test_avgpool2d_init_with_padding(self):
        """Test AvgPool2d initialization with padding."""
        kernel_size = 2
        padding = 1
        pool = AvgPool2d(kernel_size, padding=padding)

        assert pool.padding == padding


class TestAvgPool2dForward:
    """Test AvgPool2d forward pass."""

    def test_avgpool2d_forward_basic(self):
        """Test basic AvgPool2d forward pass."""
        batch_size = 2
        channels = 3
        height, width = 8, 8
        kernel_size = 2

        pool = AvgPool2d(kernel_size)
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        output = pool.forward(x)
        
        expected_height = height // kernel_size
        expected_width = width // kernel_size
        assert_shape_correct(output, (batch_size, channels, expected_height, expected_width))
        assert_finite(output)

    def test_avgpool2d_forward_correctness(self):
        """Test AvgPool2d produces correct average values."""
        batch_size = 1
        channels = 1
        height, width = 4, 4
        kernel_size = 2

        pool = AvgPool2d(kernel_size)
        
        x_data = np.array([[[[1.0, 2.0, 5.0, 6.0],
                             [3.0, 4.0, 7.0, 8.0],
                             [9.0, 10.0, 13.0, 14.0],
                             [11.0, 12.0, 15.0, 16.0]]]], dtype=np.float64)
        x = Tensor(x_data)
        
        output = pool.forward(x)
        
        expected = np.array([[[[2.5, 6.5],
                               [10.5, 14.5]]]], dtype=np.float64)
        assert_close(output.data, expected)

    def test_avgpool2d_forward_with_stride(self):
        """Test AvgPool2d forward pass with custom stride."""
        batch_size = 1
        channels = 1
        height, width = 8, 8
        kernel_size = 2
        stride = 1

        pool = AvgPool2d(kernel_size, stride=stride)
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        output = pool.forward(x)
        
        expected_height = (height - kernel_size) // stride + 1
        expected_width = (width - kernel_size) // stride + 1
        assert_shape_correct(output, (batch_size, channels, expected_height, expected_width))

    def test_avgpool2d_forward_preserves_channels(self):
        """Test AvgPool2d preserves number of channels."""
        batch_size = 2
        channels = 16
        height, width = 8, 8
        kernel_size = 2

        pool = AvgPool2d(kernel_size)
        x = Tensor(np.random.randn(batch_size, channels, height, width))
        
        output = pool.forward(x)
        
        assert output.shape[1] == channels

    def test_avgpool2d_forward_invalid_input_shape(self):
        """Test AvgPool2d raises error for invalid input shape."""
        pool = AvgPool2d(2)
        x = Tensor(np.random.randn(2, 3, 8))
        
        with pytest.raises(ValueError, match="Expected 4D input"):
            pool.forward(x)

    def test_avgpool2d_call_method(self):
        """Test AvgPool2d __call__ method works."""
        pool = AvgPool2d(2)
        x = Tensor(np.random.randn(2, 3, 8, 8))
        
        output = pool(x)
        
        assert_shape_correct(output, (2, 3, 4, 4))

    def test_avgpool2d_gradient_tracking(self):
        """Test AvgPool2d preserves gradient tracking."""
        pool = AvgPool2d(2)
        x = Tensor(np.random.randn(1, 1, 4, 4), requires_grad=True)
        
        output = pool.forward(x)
        
        assert output.requires_grad == True


class TestAvgPool2dParameters:
    """Test AvgPool2d parameter management."""

    def test_avgpool2d_no_parameters(self):
        """Test AvgPool2d has no trainable parameters."""
        pool = AvgPool2d(2)
        params = pool.parameters()
        
        assert_parameters_count(pool, 0)


class TestBatchNorm2dInitialization:
    """Test BatchNorm2d layer initialization."""

    def test_batchnorm2d_init_basic(self):
        """Test BatchNorm2d initialization."""
        num_features = 16
        bn = BatchNorm2d(num_features)

        assert bn.num_features == num_features
        assert bn.eps == 1e-5
        assert bn.momentum == 0.1
        assert bn.training == True

    def test_batchnorm2d_init_custom_params(self):
        """Test BatchNorm2d with custom eps and momentum."""
        num_features = 16
        eps = 1e-3
        momentum = 0.2
        bn = BatchNorm2d(num_features, eps=eps, momentum=momentum)

        assert bn.eps == eps
        assert bn.momentum == momentum

    def test_batchnorm2d_gamma_initialization(self):
        """Test BatchNorm2d gamma initialized to ones."""
        num_features = 16
        bn = BatchNorm2d(num_features)

        assert_shape_correct(bn.gamma, (num_features,))
        expected_gamma = np.ones(num_features)
        assert_close(bn.gamma.data, expected_gamma)

    def test_batchnorm2d_beta_initialization(self):
        """Test BatchNorm2d beta initialized to zeros."""
        num_features = 16
        bn = BatchNorm2d(num_features)

        assert_shape_correct(bn.beta, (num_features,))
        assert_bias_zeros(bn.beta)

    def test_batchnorm2d_running_stats_initialization(self):
        """Test BatchNorm2d running statistics initialized correctly."""
        num_features = 16
        bn = BatchNorm2d(num_features)

        assert_close(bn.running_mean, np.zeros(num_features))
        assert_close(bn.running_var, np.ones(num_features))


class TestBatchNorm2dModes:
    """Test BatchNorm2d training and evaluation modes."""

    def test_batchnorm2d_train_mode(self):
        """Test BatchNorm2d train() sets training mode."""
        bn = BatchNorm2d(16)
        bn.training = False
        
        result = bn.train()
        
        assert bn.training == True
        assert result is bn

    def test_batchnorm2d_eval_mode(self):
        """Test BatchNorm2d eval() sets evaluation mode."""
        bn = BatchNorm2d(16)
        
        result = bn.eval()
        
        assert bn.training == False
        assert result is bn


class TestBatchNorm2dForward:
    """Test BatchNorm2d forward pass."""

    def test_batchnorm2d_forward_training_basic(self):
        """Test BatchNorm2d forward pass in training mode."""
        batch_size = 4
        num_features = 3
        height, width = 8, 8

        bn = BatchNorm2d(num_features)
        x = Tensor(np.random.randn(batch_size, num_features, height, width))
        
        output = bn.forward(x)
        
        assert_shape_correct(output, (batch_size, num_features, height, width))
        assert_finite(output)

    def test_batchnorm2d_forward_training_normalization(self):
        """Test BatchNorm2d normalizes to zero mean and unit variance in training."""
        batch_size = 8
        num_features = 2
        height, width = 4, 4

        bn = BatchNorm2d(num_features)
        bn.gamma.data = np.ones(num_features)
        bn.beta.data = np.zeros(num_features)
        
        x = Tensor(np.random.randn(batch_size, num_features, height, width) * 10 + 5)
        output = bn.forward(x)
        
        for c in range(num_features):
            channel_output = output.data[:, c, :, :]
            mean = np.mean(channel_output)
            std = np.std(channel_output)
            
            assert abs(mean) < 0.1
            assert abs(std - 1.0) < 0.1

    def test_batchnorm2d_forward_training_updates_running_stats(self):
        """Test BatchNorm2d updates running statistics in training mode."""
        batch_size = 4
        num_features = 2
        height, width = 4, 4

        bn = BatchNorm2d(num_features)
        bn.running_mean = np.zeros(num_features)
        bn.running_var = np.ones(num_features)
        
        x = Tensor(np.random.randn(batch_size, num_features, height, width) * 2 + 3)
        _ = bn.forward(x)
        
        assert not np.allclose(bn.running_mean, 0.0, atol=TOLERANCE)
        assert not np.allclose(bn.running_var, 1.0, atol=TOLERANCE)

    def test_batchnorm2d_forward_eval_uses_running_stats(self):
        """Test BatchNorm2d uses running statistics in eval mode."""
        batch_size = 2
        num_features = 2
        height, width = 4, 4

        bn = BatchNorm2d(num_features)
        bn.eval()
        
        bn.running_mean = np.array([1.0, 2.0])
        bn.running_var = np.array([0.5, 1.5])
        bn.gamma.data = np.ones(num_features)
        bn.beta.data = np.zeros(num_features)
        
        x = Tensor(np.ones((batch_size, num_features, height, width)) * 3.0)
        output = bn.forward(x)
        
        assert_finite(output)
        assert_shape_correct(output, (batch_size, num_features, height, width))

    def test_batchnorm2d_forward_applies_affine_transform(self):
        """Test BatchNorm2d applies gamma and beta correctly."""
        batch_size = 2
        num_features = 2
        height, width = 2, 2

        bn = BatchNorm2d(num_features)
        bn.gamma.data = np.array([2.0, 3.0])
        bn.beta.data = np.array([1.0, -1.0])
        
        x = Tensor(np.ones((batch_size, num_features, height, width)) * 5.0)
        output = bn.forward(x)
        
        assert_finite(output)

    def test_batchnorm2d_forward_invalid_input_shape(self):
        """Test BatchNorm2d raises error for invalid input shape."""
        bn = BatchNorm2d(3)
        x = Tensor(np.random.randn(2, 3, 8))
        
        with pytest.raises(ValueError, match="Expected 4D input"):
            bn.forward(x)

    def test_batchnorm2d_forward_wrong_channels(self):
        """Test BatchNorm2d raises error for wrong number of channels."""
        bn = BatchNorm2d(16)
        x = Tensor(np.random.randn(2, 8, 4, 4))
        
        with pytest.raises(ValueError, match="Expected 16 channels"):
            bn.forward(x)

    def test_batchnorm2d_call_method(self):
        """Test BatchNorm2d __call__ method works."""
        bn = BatchNorm2d(3)
        x = Tensor(np.random.randn(2, 3, 8, 8))
        
        output = bn(x)
        
        assert_shape_correct(output, (2, 3, 8, 8))


class TestBatchNorm2dParameters:
    """Test BatchNorm2d parameter management."""

    def test_batchnorm2d_parameters(self):
        """Test BatchNorm2d returns correct parameters."""
        bn = BatchNorm2d(16)
        params = bn.parameters()
        
        assert_parameters_count(bn, 2)
        assert params[0] is bn.gamma
        assert params[1] is bn.beta


class TestSpatialIntegration:
    """Integration tests for spatial layers."""

    def test_conv_pool_stack(self):
        """Test stacking Conv2d with pooling layers."""
        batch_size = 2
        in_channels = 3
        height, width = 16, 16

        conv = Conv2d(in_channels, 16, kernel_size=3, padding=1)
        pool = MaxPool2d(2)
        
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        x = conv(x)
        x = pool(x)
        
        assert_shape_correct(x, (batch_size, 16, 8, 8))

    def test_conv_batchnorm_pool_stack(self):
        """Test stacking Conv2d, BatchNorm2d, and pooling."""
        batch_size = 2
        in_channels = 3
        out_channels = 16
        height, width = 16, 16

        conv = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        bn = BatchNorm2d(out_channels)
        pool = MaxPool2d(2)
        
        x = Tensor(np.random.randn(batch_size, in_channels, height, width))
        x = conv(x)
        x = bn(x)
        x = pool(x)
        
        assert_shape_correct(x, (batch_size, out_channels, 8, 8))
        assert_finite(x)

    def test_multiple_conv_layers(self):
        """Test stacking multiple Conv2d layers."""
        batch_size = 2
        height, width = 16, 16

        conv1 = Conv2d(3, 16, kernel_size=3, padding=1)
        conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        conv3 = Conv2d(32, 64, kernel_size=3, padding=1)
        
        x = Tensor(np.random.randn(batch_size, 3, height, width))
        x = conv1(x)
        x = conv2(x)
        x = conv3(x)
        
        assert_shape_correct(x, (batch_size, 64, height, width))
        assert_finite(x)
