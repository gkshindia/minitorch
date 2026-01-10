"""
Cross-Activation Comparison Tests

Tests comparing different activation functions to understand their
conceptual and mathematical differences:
- ReLU vs Sigmoid
- Tanh vs Sigmoid
- GELU vs ReLU
- GELU vs Sigmoid
"""
import numpy as np
import pytest
from core.tensor import Tensor
from core.activations import ReLU, Sigmoid, Tanh, GELU as Gelu
from tests.activations.conftest import TOLERANCE


class TestReLUSigmoidComparison:
    """Comparison tests between ReLU and Sigmoid."""
    
    def test_relu_unbounded_sigmoid_bounded(self):
        """ReLU has unbounded positive output, Sigmoid is bounded [0,1]."""
        relu = ReLU()
        sigmoid = Sigmoid()
        
        x = Tensor([10.0])
        
        # ReLU preserves magnitude
        relu_output = relu(x)
        assert relu_output.data[0] == 10.0
        
        # Sigmoid saturates
        sigmoid_output = sigmoid(x)
        assert sigmoid_output.data[0] <= 1.0
        assert sigmoid_output.data[0] > 0.9
    
    def test_relu_extreme_values_vs_sigmoid(self):
        """ReLU with extreme values vs Sigmoid saturation."""
        relu = ReLU()
        sigmoid = Sigmoid()
        
        large_x = Tensor([1000.0])
        
        # ReLU output grows with input
        relu_large = relu(large_x)
        assert relu_large.data[0] == 1000.0
        
        # Sigmoid stays bounded
        sigmoid_large = sigmoid(large_x)
        assert sigmoid_large.data[0] <= 1.0


class TestTanhSigmoidComparison:
    """Comparison tests between Tanh and Sigmoid."""
    
    def test_tanh_zero_centered_sigmoid_not(self):
        """Tanh is zero-centered, Sigmoid is not."""
        tanh = Tanh()
        sigmoid = Sigmoid()
        
        x = Tensor([0.0])
        
        # Tanh(0) = 0
        tanh_output = tanh(x)
        assert np.allclose(tanh_output.data, 0.0, atol=TOLERANCE)
        
        # Sigmoid(0) = 0.5
        sigmoid_output = sigmoid(x)
        assert np.allclose(sigmoid_output.data, 0.5, atol=TOLERANCE)
    
    def test_tanh_negative_range_sigmoid_positive_only(self):
        """Tanh can output negative, Sigmoid always positive."""
        tanh = Tanh()
        sigmoid = Sigmoid()
        
        x_neg = Tensor([-1.0])
        
        # Tanh can output negative
        tanh_result = tanh(x_neg)
        assert tanh_result.data[0] < 0.0
        
        # Sigmoid always positive
        sigmoid_result = sigmoid(x_neg)
        assert sigmoid_result.data[0] > 0.0
    
    def test_tanh_range_vs_sigmoid_range(self):
        """Tanh range [-1,1] vs Sigmoid range [0,1]."""
        tanh = Tanh()
        sigmoid = Sigmoid()
        
        x_large = Tensor([100.0])
        
        # Tanh approaches ±1
        assert tanh(x_large).data[0] <= 1.0
        
        x_small = Tensor([-100.0])
        assert tanh(x_small).data[0] >= -1.0
        
        # Sigmoid approaches 0 or 1
        assert sigmoid(x_large).data[0] <= 1.0
        assert sigmoid(x_small).data[0] >= 0.0


class TestGeluReLUComparison:
    """Comparison tests between GELU and ReLU."""
    
    def test_gelu_smooth_relu_sharp(self):
        """GELU is smooth at zero, ReLU has sharp kink."""
        gelu = Gelu()
        relu = ReLU()
        
        # Around zero
        x_near_zero = Tensor([-0.1, 0.0, 0.1])
        
        gelu_result = gelu(x_near_zero)
        relu_result = relu(x_near_zero)
        
        # GELU should be smooth
        assert gelu_result.data[0] < 0.0      # GELU smooth through zero
        assert gelu_result.data[1] < 0.01     # GELU(0) ≈ 0
        assert gelu_result.data[2] > 0.0      # GELU smooth through zero
        
        # ReLU has hard cutoff
        assert relu_result.data[0] == 0.0     # ReLU hard cutoff
        assert relu_result.data[1] == 0.0     # ReLU(0) = 0
        assert relu_result.data[2] == 0.1     # ReLU identity
    
    def test_gelu_allows_small_negative_relu_zeros(self):
        """GELU allows small negative values, ReLU zeros them."""
        gelu = Gelu()
        relu = ReLU()
        
        x_neg = Tensor([-1.0])
        
        gelu_result = gelu(x_neg)
        relu_result = relu(x_neg)
        
        # ReLU zeros negative
        assert relu_result.data[0] == 0.0
        
        # GELU allows small negative
        assert gelu_result.data[0] < 0.0
        assert gelu_result.data[0] > -0.2
    
    def test_gelu_approximates_relu_for_large_positive(self):
        """GELU ≈ ReLU for large positive values."""
        gelu = Gelu()
        relu = ReLU()
        
        x_large = Tensor([5.0, 10.0, 20.0])
        
        gelu_result = gelu(x_large)
        relu_result = relu(x_large)
        
        # Should be very similar
        assert np.allclose(gelu_result.data, relu_result.data, rtol=0.01)


class TestGeluSigmoidComparison:
    """Comparison tests between GELU and Sigmoid."""
    
    def test_gelu_unbounded_sigmoid_bounded(self):
        """GELU has unbounded output, Sigmoid bounded [0,1]."""
        gelu = Gelu()
        sigmoid = Sigmoid()
        
        x_large = Tensor([10.0])
        
        # GELU preserves large values
        gelu_result = gelu(x_large)
        assert np.allclose(gelu_result.data, 10.0, rtol=0.01)
        
        # Sigmoid saturates
        sigmoid_result = sigmoid(x_large)
        assert sigmoid_result.data[0] <= 1.0
    
    def test_gelu_negative_sigmoid_positive_only(self):
        """GELU can output negative, Sigmoid always positive."""
        gelu = Gelu()
        sigmoid = Sigmoid()
        
        x_neg = Tensor([-1.0])
        
        # GELU can output negative
        gelu_result = gelu(x_neg)
        assert gelu_result.data[0] < 0.0
        
        # Sigmoid always positive
        sigmoid_result = sigmoid(x_neg)
        assert sigmoid_result.data[0] > 0.0
    
    def test_gelu_zero_centered_sigmoid_not(self):
        """GELU is more zero-centered than Sigmoid."""
        gelu = Gelu()
        sigmoid = Sigmoid()
        
        x = Tensor([0.0])
        
        # GELU(0) ≈ 0
        gelu_result = gelu(x)
        assert np.allclose(gelu_result.data, 0.0, atol=0.01)
        
        # Sigmoid(0) = 0.5
        sigmoid_result = sigmoid(x)
        assert np.allclose(sigmoid_result.data, 0.5, atol=TOLERANCE)


class TestAllActivationsBatchConsistency:
    """Verify all activations handle batches consistently."""
    
    def test_all_activations_batch_consistency(self, batch_tensor):
        """All activations should process batches independently."""
        activations = [
            ("Sigmoid", Sigmoid()),
            ("ReLU", ReLU()),
            ("Tanh", Tanh()),
            ("GELU", Gelu())
        ]
        
        for name, activation in activations:
            result = activation(batch_tensor)
            
            # Verify shape preservation
            assert result.shape == batch_tensor.shape, f"{name} shape not preserved"
            
            # Verify independent processing
            for i in range(batch_tensor.shape[0]):
                sample = Tensor(batch_tensor.data[i])
                sample_result = activation(sample)
                assert np.allclose(result.data[i], sample_result.data, atol=TOLERANCE), \
                    f"{name} batch processing not independent"
    
    def test_all_activations_api_consistency(self):
        """__call__ and forward should be consistent for all."""
        activations = [
            ("Sigmoid", Sigmoid()),
            ("ReLU", ReLU()),
            ("Tanh", Tanh()),
            ("GELU", Gelu())
        ]
        
        x = Tensor([1.0, -2.0, 3.0])
        
        for name, activation in activations:
            result_forward = activation.forward(x)
            result_call = activation(x)
            
            assert np.allclose(result_forward.data, result_call.data, atol=TOLERANCE), \
                f"{name} forward vs call inconsistent"
    
    def test_all_activations_repr(self):
        """All activations should have meaningful repr."""
        activations = [
            ("Sigmoid", Sigmoid()),
            ("ReLU", ReLU()),
            ("Tanh", Tanh()),
            ("GELU", Gelu())
        ]
        
        for name, activation in activations:
            repr_str = repr(activation)
            assert name in repr_str, f"{name} not in repr: {repr_str}"
