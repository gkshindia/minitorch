import numpy as np
from typing import Dict, Any, Optional, List, Union

from core.abstracts import OptimizerAbstract
from core.tensor import Tensor
from core.autograd import enable_autograd
enable_autograd()


DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9  # Default momentum for SGD
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW


class SGD(OptimizerAbstract):
    """
    Stochastic Gradient Descent with momentum.

    SGD is the foundational optimization algorithm that moves parameters
    in the direction opposite to gradients. With momentum, it remembers
    previous updates to reduce oscillations and accelerate convergence.
    """

    def __init__(self, parameters: List[Tensor], learning_rate: float = DEFAULT_LEARNING_RATE_SGD, momentum: float = 0.0, weight_decay: float = 0.0):
        """
        1. Call parent constructor to set up parameters
        2. Store learning rate, momentum, and weight decay
        3. Initialize momentum buffers for each parameter

        - Momentum buffers should be initialized as None
        - They'll be created lazily on first step
        """

        super().__init__(parameters=parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.momentum_buffers = [None for _ in self.parameters]

    
    def has_momentum(self) -> bool:
        return self.momentum > 0
    
    def get_momentum_state(self) -> Optional[List]:
        if not self.has_momentum():
            return None
        return [buf.copy() if buf is not None else None for buf in self.momentum_buffers]
    
    def set_momentum_buffers(self, state: Optional[List]) -> None:
        if state is None or not self.has_momentum():
            return
        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"State length {len(state)} doesn;t match"
                f"optimizer parameters {len(self.momentum_buffers)}"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()
    
    def step(self):
        """
        Perform SGD update with step momentum

        1. For each parameter with gradients:
           a. Apply weight decay if specified
           b. Update momentum buffer
           c. Update parameter using momentum

        FORMULA:
        - With weight decay: grad = grad + weight_decay * param
        - Momentum: v = momentum * v_prev + grad
        - Update: param = param - lr * v

        HINTS:
        - Skip parameters without gradients
        - Initialize momentum buffers on first use
        - Use in-place operations to save memory
        """

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                grad_data = grad
            
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(param.data)
                
                # Update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data
                grad_data = self.momentum_buffers[i]
            
            param.data = param.data - self.learning_rate * grad_data

        self.step_count += 1
    
    def zero_grad(self):
        """
        Clear gradients of all parameters.
        
        Sets the gradient of each parameter to None.
        This should be called before each backward pass.
        """
        for param in self.parameters:
            param.grad = None



class Adam(OptimizerAbstract):
    """
    Adam optimizer with adaptive learning rates. 
    Adam computes individual adaptive learning rates for different parameters
    from estimates of first and second moments of the gradients.
    This makes it effective for problems with sparse gradients or noisy data.
    """

    def __init__(self, parameters: List[Tensor], learning_rate: float = DEFAULT_LEARNING_RATE_ADAM, betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2), eps: float = DEFAULT_EPS, weight_decay: float = 0.0):
        """
        1. Call parent constructor
        2. Store hyperparameters (lr, betas, eps, weight_decay)
        3. Initialize first and second moment buffers

        - lr: Learning rate (default: 0.001)
        - betas: Coefficients for computing running averages (default: (0.9, 0.999))
        - eps: Small constant for numerical stability (default: 1e-8)
        - weight_decay: L2 penalty coefficient (default: 0.0)
        """
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m_buffers = [None for _ in self.parameters]
        self.v_buffers = [None for _ in self.parameters]
    
    def step(self):
        """
        1. For each parameter with gradients:
           a. Apply weight decay if specified
           b. Update first moment estimate (momentum of gradient)
           c. Update second moment estimate (momentum of squared gradient)
           d. Compute bias-corrected moments
           e. Update parameter using adaptive learning rate

        FORMULAE:
            - m_t = β₁ * m_{t-1} + (1-β₁) * g_t
            - v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
            - m̂_t = m_t / (1-β₁^t)
            - v̂_t = v_t / (1-β₂^t)
            - θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)

        - Initialize buffers as zeros on first use
        - Use step_count for bias correction
        - Square gradients element-wise for second moment
        """

        self.step_count += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                grad_data = grad

            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data
            
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)
            
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1-self.beta1) * grad_data

            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1-self.beta2) * (grad_data ** 2)

            bias_correction_1 = 1 - self.beta1 ** self.step_count
            bias_correction_2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction_1
            v_hat = self.v_buffers[i] / bias_correction_2

            param.data = param.data - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """
        Clear gradients of all parameters.
        
        Sets the gradient of each parameter to None.
        This should be called before each backward pass.
        """
        for param in self.parameters:
            param.grad = None


class AdamW(OptimizerAbstract):
    """
    AdamW optimizer with decoupled weight decay.

    AdamW fixes a bug in Adam's weight decay implementation by decoupling
    weight decay from the gradient-based update. This leads to better
    regularization and is the preferred version for most applications.
    """

    def __init__(self, parameters: List[Tensor], learning_rate: float = DEFAULT_LEARNING_RATE_ADAM, betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2), eps: float = DEFAULT_EPS, weight_decay: float = DEFAULT_WEIGHT_DECAY_ADAMW):
        """
        1. Call parent constructor
        2. Store hyperparameters (lr, betas, eps, weight_decay)
        3. Initialize first and second moment buffers

        - lr: Learning rate (default: 0.001)
        - betas: Coefficients for computing running averages (default: (0.9, 0.999))
        - eps: Small constant for numerical stability (default: 1e-8)
        - weight_decay: L2 penalty coefficient (default: 0.01)
        """
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m_buffers = [None for _ in self.parameters]
        self.v_buffers = [None for _ in self.parameters]
    
    def step(self):
        """
        1. For each parameter with gradients:
           a. Update moments using gradients (NOT modified by weight decay)
           b. Compute bias-corrected moments
           c. Apply gradient-based update
           d. Apply weight decay directly to parameters
        
        KEY DIFFERENCE from Adam:
        - Weight decay: θ_t = θ_t - lr * weight_decay * θ_t (applied after gradient update)
        - NOT: grad = grad + weight_decay * param (Adam's incorrect approach)

        FORMULAS:
        - Same moment updates as Adam (using unmodified gradients)
        - Gradient update: θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
        - Weight decay: θ_t = θ_t * (1 - lr * weight_decay)

        HINT: Apply weight decay after gradient update for proper decoupling
        """

        self.step_count += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad

            grad = param.grad
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                grad_data = grad

            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)
            
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            param.data = param.data - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            if self.weight_decay != 0:
                param.data = param.data * (1 - self.learning_rate * self.weight_decay)

    def zero_grad(self):
        """
        Clear gradients of all parameters.
        
        Sets the gradient of each parameter to None.
        This should be called before each backward pass.
        """
        for param in self.parameters:
            param.grad = None