import numpy as np
from typing import Optional

from core.tensor import Tensor
from core.activations import ReLU
from core.layers import Linear


"""

Problem Type Decision Tree:

What are you predicting?
         │
    ┌────┼────┐
    │         │
Continuous   Categorical
 Values       Classes
    │         │
    │    ┌───┼───┐
    │    │       │
    │   2 Classes  3+ Classes
    │       │       │
 MSELoss   BCE Loss  CE Loss

Examples:
MSE: House prices, temperature, stock values
BCE: Spam detection, fraud detection, medical diagnosis

Error Sensitivity Comparison:

Small Error (0.1):     Medium Error (0.5):     Large Error (2.0):

MSE:     0.01         MSE:     0.25           MSE:     4.0
BCE:     0.11         BCE:     0.69           BCE:     ∞ (clips to large)
CE:      0.11         CE:      0.69           CE:      ∞ (clips to large)

MSE: Quadratic growth, manageable with outliers
BCE/CE: Logarithmic growth, explodes with confident wrong predictions
"""


EPSILON = 1e-7


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute log softmax with numerical stability preventing overflow

    1. Find maximum along dimension (for stability)
    2. Subtract max from input (prevents overflow)
    3. Compute log(sum(exp(shifted_input)))
    4. Return input - max - log_sum_exp

    Use np.max(x.data, axis=dim, keepdims=True) to preserve dimensions
    """

    max_vals = np.max(x.data, axis=dim, keepdims=True)
    shifted = x.data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    result = x.data - max_vals - log_sum_exp

    return Tensor(result)


class MSELoss:
    """
    Mean squared error loss for regression tasks
    It measures how far your continuous predictions are from the true values.
    """

    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        1. Compute difference: predictions - targets
        2. Square the differences: diff²
        3. Take mean across all elements

        - Use (predictions.data - targets.data) for element-wise difference
        - Square with **2 or np.power(diff, 2)
        - Use np.mean() to average over all elements

        """

        diff = predictions.data - targets.data
        squared_diff = diff ** 2
        mse = np.mean(squared_diff)
        return Tensor(mse)
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
    
    def backward(self) -> Tensor:
        pass


class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.
    It measures how wrong your probability predictions are and heavily penalizes confident mistakes.
    """

    def __init__(self):
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        1. Compute log-softmax of logits (numerically stable)
        2. Select log-probabilities for correct classes
        3. Return negative mean of selected log-probabilities

        - Use log_softmax() for numerical stability
        - targets.data.astype(int) ensures integer indices
        - Use np.arange(batch_size) for row indexing: log_probs[np.arange(batch_size), targets]
        - Return negative mean: -np.mean(selected_log_probs)

        """

        log_probs = log_softmax(logits, dim=-1)
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)
    
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)
    
    def backward(self):
        pass


class BinaryCrossEntropyLoss:
    """
    Binary Cross-Entropy is specialized for yes/no decisions. 
    It's like regular cross-entropy but optimized for the special case of exactly two classes.
    """

    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        1. Clamp predictions to avoid log(0) and log(1)
        2. Compute: -(targets * log(predictions) + (1-targets) * log(1-predictions))
        3. Return mean across all samples

        - Use np.clip(predictions.data, 1e-7, 1-1e-7) to prevent log(0)
        - Binary cross-entropy: -(targets * log(preds) + (1-targets) * log(1-preds))
        - Use np.mean() to average over all samples

        """

        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1-eps)

        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1-clamped_preds)
        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)
    
    def backward(self):
        pass
