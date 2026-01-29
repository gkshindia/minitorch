import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys
import os

from core.tensor import Tensor
from core.layers import Linear
from core.losses import MSELoss, CrossEntropyLoss
from core.optimizers import SGD, AdamW


DEFAULT_MAX_LR = 0.1
DEFAULT_MIN_LR = 0.01
DEFAULT_TOTAL_EPOCHS = 100


class CosineSchedule:
    """
    Cosine annealing learning rate schedule

    1. Store max_lr, min_lr, and total_epochs
    2. In get_lr(), compute cosine factor: (1 + cos(π * epoch / total_epochs)) / 2
    3. Interpolate: min_lr + (max_lr - min_lr) * cosine_factor

    Use np.cos() and np.pi for the cosine calculation

    """

    def __init__(self, max_lr: float = DEFAULT_MAX_LR, min_lr: float = DEFAULT_MIN_LR, total_epoch: int = DEFAULT_TOTAL_EPOCHS):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epoch
    
    def get_lr(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.min_lr
        
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor


def clip_gradeints_norm(parameters: List[Tensor], max_norm: float = 10.0) -> float:
    """
    Clip gradients by global norm

    1. Compute total norm: sqrt(sum of squared gradients across all parameters)
    2. If total_norm > max_norm, compute clip_coef = max_norm / total_norm
    3. Scale all gradients by clip_coef: grad *= clip_coef
    4. Return the original norm for monitoring

    - Use np.linalg.norm() to compute norms
    - Only clip if total_norm > max_norm
    - Modify gradients in-place for efficiency

    """

    if not parameters:
        return 0.0

    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            else:
                grad_data = param.grad.data
            total_norm += np.sum(grad_data ** 2)
        
    
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        clip_coeff = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                if isinstance(param.grad, np.ndarray):
                    param.grad *= clip_coeff
                else:
                    param.grad.data *= clip_coeff
    
    return float(total_norm)


class Trainer:
    """
    Complete training orchestrator for neural networks

    for epoch in range(num_epochs)
        for batch in data_loader

            forward pass
                1. input -> models
                2. predictions
            
            compute loss
                3. loss_fn(predictions, targets)
            
            backward pass
                4. loss.backward()
                5. gradients
            
            parameter update
                6. optimizer.step()
                7. zero gradients
        
        Learning rate update
            8. scheduler.step()
    
    1. __init__(): Store model, optimizer, loss_fn, scheduler, and grad_clip_norm
    2. train_epoch(): Loop through dataloader, forward → loss → backward → step
    3. evaluate(): Similar loop but set model.training=False, no grad updates
    4. save/load_checkpoint(): Use pickle to persist/restore all training state

    """

    def __init__(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm: Optional[float] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

        self.epoch = 0
        self.step = 0
        self.training_mode = True

        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': []
        }
    
    def train_epoch(self, dataloader, accumulation_steps: int = 1) -> float:
        self.model.training = True
        self.training_mode = True

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            scaled_loss = loss.data / accumulation_steps
            accumulated_loss += scaled_loss

            loss.backward()

            if (batch_idx +1) % accumulation_steps == 0:
                if self.grad_clip_norm is not None:
                    params = self.model.parameters()
                    clip_gradeints_norm(params, self.grad_clip_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1
        
        if accumulated_loss > 0.0:
            if self.grad_clip_norm is not None:
                params = self.model.parameters()
                clip_gradeints_norm(params, self.grad_clip_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += accumulated_loss
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.history['train_loss'].append(avg_loss)

        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            self.optimizer.learning_rate = current_lr
            self.history['learning_rates'].append(current_lr)
        
        self.epoch += 1
        return avg_loss
    
    def evaluate(self, dataloader) -> float:
        self.model.training = False
        self.training_mode = False

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for inputs, targets in dataloader:
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)

            total_loss += loss.data
            num_batches += 1

            if len(outputs.data.shape) > 1:
                predictions = np.argmax(outputs.data, axis=1)
                if len(targets.data.shape) == 1:
                    correct += np.sum(predictions == targets.data)
                else:
                    correct += np.sum(predictions == np.argmax(targets.data, axis=1))
                
                total += targets.data.shape[0]
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)
        return avg_loss, accuracy
    
    def save_checkpoint(self, filepath: str):
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state': self._get_model_state(),
            'optimizer_state': self._get_optimizer_state(),
            'history': self.history,
            'training_mode': self.training_mode
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def load_checkpoint(self, filepath: str):
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        self.training_mode = checkpoint['training_mode']
        self.model.training = self.training_mode

        if 'model_state' in checkpoint:
            self._set_model_state(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self._set_optimizer_state(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self._set_scheduler_state(checkpoint['scheduler_state'])
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {i: param.data.copy() for i, param in enumerate(self.model.parameters())}
    
    def _set_model_state(self, state: Dict[str, Any]):
        for i, param in enumerate(self.model.parameters()):
            if i in state:
                param.data = state[i].copy()

    def _get_optimizer_state(self) -> Dict[str, Any]:
        state = {}
        state['learning_rate'] = self.optimizer.learning_rate
        if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state['momentum_buffers'] = momentum_state
        return state
    
    def _set_optimizer_state(self, state: Dict[str, Any]):
        if 'learning_rate' in state:
            self.optimizer.learning_rate = state['learning_rate']
        if 'momentum_buffers' in state:
            if hasattr(self.optimizer, 'set_momentum_state') and self.optimizer.has_momentum():
                self.optimizer.set_momentum_state(state['momentum_buffers'])
    
    def _get_scheduler_state(self) -> Dict[str, Any]:
        if self.scheduler is None:
            return None
        return {
            'max_lr': getattr(self.scheduler, 'max_lr', None),
            'min_lr': getattr(self.scheduler, 'min_lr', None),
            'total_epochs': getattr(self.scheduler, 'total_epochs', None)
        }
    
    def _set_scheduler_state(self, state: Dict[str, Any]):
        if state is None or self.scheduler is None:
            return
        
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)


def demonstrate_complete_training_pipeline():
    """
    Complete end-to-end training example using all components.

    This demonstrates how Trainer, scheduler, gradient clipping,
    and checkpointing work together in a real training scenario.
    """
    print("🏗️ Building Complete Training Pipeline...")
    print("=" * 60)

    # Step 1: Create model using REAL Linear layer
    class SimpleNN:
        def __init__(self):
            self.layer1 = Linear(3, 5)
            self.layer2 = Linear(5, 2)
            self.training = True

        def forward(self, x):
            x = self.layer1.forward(x)
            # Simple ReLU-like activation (max with 0)
            x = Tensor(np.maximum(0, x.data))
            x = self.layer2.forward(x)
            return x

        def parameters(self):
            return self.layer1.parameters() + self.layer2.parameters()

    print("✓ Model created: 3 → 5 → 2 network")

    # Step 2: Create optimizer
    model = SimpleNN()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    print("✓ Optimizer: SGD with momentum")

    # Step 3: Create loss function
    loss_fn = MSELoss()
    print("✓ Loss function: MSE")

    # Step 4: Create scheduler
    scheduler = CosineSchedule(max_lr=0.1, min_lr=0.001, total_epochs=5)
    print("✓ Scheduler: Cosine annealing (0.1 → 0.001)")

    # Step 5: Create trainer with gradient clipping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=1.0
    )
    print("✓ Trainer initialized with gradient clipping")

    # Step 6: Create synthetic training data
    train_data = [
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2))),
        (Tensor(np.random.randn(4, 3)), Tensor(np.random.randn(4, 2)))
    ]
    print("✓ Training data: 3 batches of 4 samples")

    # Step 7: Train for multiple epochs
    print("\n🚀 Starting Training...")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Learning Rate':<15}")
    print("-" * 60)

    for epoch in range(3):
        loss = trainer.train_epoch(train_data)
        lr = scheduler.get_lr(epoch)
        print(f"{epoch:<8} {loss:<12.6f} {lr:<15.6f}")

    # Step 8: Save checkpoint
    checkpoint_path = "/tmp/training_example_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    # Step 9: Evaluate
    eval_loss, accuracy = trainer.evaluate(train_data)
    print(f"✓ Evaluation - Loss: {eval_loss:.6f}, Accuracy: {accuracy:.6f}")

    # Clean up
    import os
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n" + "=" * 60)
    print("✅ Complete training pipeline executed successfully!")
    print("\n💡 This pipeline demonstrates:")
    print("   • Model → Optimizer → Loss → Scheduler → Trainer integration")
    print("   • Training loop with scheduling and gradient clipping")
    print("   • Checkpointing for training persistence")
    print("   • Evaluation mode for model assessment")