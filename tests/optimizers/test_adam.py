"""
Unit tests for Adam optimizer.
"""

import numpy as np
from core.tensor import Tensor
from core.optimizers import Adam


def test_unit_adam_optimizer():
    """🔬 Test Adam optimizer implementation."""
    print("🔬 Unit Test: Adam Optimizer...")

    # Test basic Adam without weight decay
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = Adam([param], learning_rate=0.1)
    original_data = param.data.copy()

    optimizer.step()

    # First step calculations:
    # m = 0.9 * 0 + (1 - 0.9) * [0.1, 0.2] = [0.01, 0.02]
    # v = 0.999 * 0 + (1 - 0.999) * [0.01, 0.04] = [0.00001, 0.00004]
    # m_hat = [0.01, 0.02] / (1 - 0.9^1) = [0.1, 0.2]
    # v_hat = [0.00001, 0.00004] / (1 - 0.999^1) = [0.01, 0.04]
    # param = [1.0, 2.0] - 0.1 * [0.1, 0.2] / (sqrt([0.01, 0.04]) + 1e-8)
    
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    m = (1 - beta1) * param.grad.data
    v = (1 - beta2) * (param.grad.data ** 2)
    m_hat = m / (1 - beta1 ** 1)
    v_hat = v / (1 - beta2 ** 1)
    expected = original_data - 0.1 * m_hat / (np.sqrt(v_hat) + eps)
    
    assert np.allclose(param.data, expected, rtol=1e-5)
    assert optimizer.step_count == 1

    # Test Adam with multiple steps
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    param2.grad = Tensor([0.1, 0.2])
    
    optimizer2 = Adam([param2], learning_rate=0.1)
    
    # First step
    optimizer2.step()
    first_step_data = param2.data.copy()
    
    # Second step with same gradient
    param2.grad = Tensor([0.1, 0.2])
    optimizer2.step()
    
    # Verify that the parameter changed (momentum accumulation)
    assert not np.allclose(param2.data, first_step_data)
    assert optimizer2.step_count == 2

    # Test Adam with weight decay
    param3 = Tensor([1.0, 2.0], requires_grad=True)
    param3.grad = Tensor([0.1, 0.2])
    
    optimizer_wd = Adam([param3], learning_rate=0.1, weight_decay=0.01)
    original_data3 = param3.data.copy()
    optimizer_wd.step()
    
    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    # Then apply Adam update with this modified gradient
    grad_wd = np.array([0.11, 0.22])
    m_wd = (1 - beta1) * grad_wd
    v_wd = (1 - beta2) * (grad_wd ** 2)
    m_hat_wd = m_wd / (1 - beta1 ** 1)
    v_hat_wd = v_wd / (1 - beta2 ** 1)
    expected_wd = original_data3 - 0.1 * m_hat_wd / (np.sqrt(v_hat_wd) + eps)
    
    assert np.allclose(param3.data, expected_wd, rtol=1e-5)

    # Test custom betas
    param4 = Tensor([1.0, 2.0], requires_grad=True)
    param4.grad = Tensor([0.1, 0.2])
    
    optimizer_custom = Adam([param4], learning_rate=0.1, betas=(0.95, 0.9999))
    optimizer_custom.step()
    
    # Verify the optimizer used custom betas
    assert optimizer_custom.beta1 == 0.95
    assert optimizer_custom.beta2 == 0.9999
    assert optimizer_custom.step_count == 1

    print("✅ Adam optimizer works correctly!")
