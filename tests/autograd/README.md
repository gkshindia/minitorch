# Autograd Test Suite

Comprehensive test suite for the automatic differentiation (autograd) system.

## 📁 Test Structure

```
tests/autograd/
├── __init__.py                          # Package initialization
├── conftest.py                          # Shared fixtures and utilities
├── README.md                            # This file
│
├── Basic Operations
│   ├── test_add_backward.py            # Addition gradient tests
│   ├── test_mul_backward.py            # Multiplication gradient tests
│   ├── test_sub_backward.py            # Subtraction gradient tests
│   └── test_div_backward.py            # Division gradient tests
│
├── Tensor Operations
│   ├── test_matmul_backward.py         # Matrix multiplication tests
│   ├── test_transpose_backward.py      # Transpose operation tests
│   ├── test_permute_backward.py        # Permutation tests
│   └── test_reshape_backward.py        # Reshape operation tests
│
├── Advanced Operations
│   ├── test_embedding_backward.py      # Embedding lookup tests
│   ├── test_slice_backward.py          # Slicing/indexing tests
│   └── test_sum_backward.py            # Sum reduction tests
│
├── Activation Functions
│   ├── test_relu_backward.py           # ReLU activation tests
│   ├── test_sigmoid_backward.py        # Sigmoid activation tests
│   ├── test_softmax_backward.py        # Softmax activation tests
│   └── test_gelu_backward.py           # GELU activation tests
│
├── Loss Functions
│   ├── test_mse_backward.py            # MSE loss tests
│   ├── test_bce_backward.py            # Binary cross-entropy tests
│   └── test_crossentropy_backward.py   # Cross-entropy loss tests
│
└── Integration
    └── test_autograd_integration.py    # Comprehensive integration tests
```

## 🎯 Test Coverage

### Basic Operations (4 classes)
- **AddBackward**: Addition gradient computation
- **MulBackward**: Multiplication gradient computation
- **SubBackward**: Subtraction gradient computation
- **DivBackward**: Division gradient computation

### Tensor Operations (4 classes)
- **MatMulBackward**: Matrix multiplication gradients
- **TransposeBackward**: Transpose operation gradients
- **PermuteBackward**: Arbitrary axis permutation gradients
- **ReshapeBackward**: Reshape operation gradients

### Advanced Operations (3 classes)
- **EmbeddingBackward**: Embedding lookup with gradient accumulation
- **SliceBackward**: Slicing/indexing operations
- **SumBackward**: Sum reduction gradients

### Activation Functions (4 classes)
- **ReLUBackward**: ReLU activation gradients
- **SigmoidBackward**: Sigmoid activation gradients
- **SoftmaxBackward**: Softmax activation gradients
- **GELUBackward**: GELU activation gradients

### Loss Functions (3 classes)
- **MSEBackward**: Mean Squared Error loss gradients
- **BCEBackward**: Binary Cross-Entropy loss gradients
- **CrossEntropyBackward**: Cross-Entropy loss gradients

### Integration Tests
- Multi-layer neural networks
- Gradient accumulation
- Complex computation graphs
- Optimizer simulation
- Batch processing
- Mixed operations

## 🚀 Running Tests

### Run All Tests
```bash
pytest tests/autograd/
```

### Run Specific Category
```bash
# Basic operations
pytest tests/autograd/test_add_backward.py
pytest tests/autograd/test_mul_backward.py

# Activation functions
pytest tests/autograd/test_relu_backward.py
pytest tests/autograd/test_sigmoid_backward.py

# Loss functions
pytest tests/autograd/test_mse_backward.py
pytest tests/autograd/test_crossentropy_backward.py
```

### Run Integration Tests
```bash
pytest tests/autograd/test_autograd_integration.py
```

### Run Individual Test Module
```bash
python tests/autograd/test_add_backward.py
```

### Run with Verbose Output
```bash
pytest tests/autograd/ -v
```

### Run with Coverage
```bash
pytest tests/autograd/ --cov=core.autograd --cov-report=html
```

## 📊 Test Patterns

Each test file follows a consistent pattern:

1. **Simple Test**: Basic functionality verification
2. **Edge Cases**: Boundary conditions and special inputs
3. **Shapes**: Different tensor shapes (1D, 2D, 3D, batched)
4. **Chains**: Operations in computation chains
5. **Weighted**: Tests with non-uniform gradients
6. **Integration**: Real-world usage scenarios

## ✅ Test Quality

- **Comprehensive**: Tests cover all autograd classes
- **Mathematical**: Verifies gradient formulas
- **Practical**: Tests real neural network scenarios
- **Robust**: Handles edge cases and numerical stability
- **Clear**: Descriptive names and documentation

## 🔍 Key Test Scenarios

### 1. Basic Gradient Flow
Tests that gradients flow correctly through single operations.

### 2. Computation Graphs
Tests that gradients flow through complex multi-operation graphs.

### 3. Gradient Accumulation
Tests that gradients accumulate correctly across multiple backward passes.

### 4. Shape Broadcasting
Tests that gradients handle broadcasting correctly.

### 5. Numerical Stability
Tests that operations remain stable with extreme values.

### 6. Real Networks
Tests that simulate actual neural network training scenarios.

## 📝 Mathematical Verification

Each test verifies:
- **Forward pass**: Correct output computation
- **Backward pass**: Correct gradient computation
- **Gradient shape**: Matches input tensor shape
- **Gradient value**: Matches mathematical formula

## 🎓 Learning Resources

Each test file includes:
- Mathematical formulas in docstrings
- Step-by-step gradient computation explanations
- Real-world usage examples
- Links to relevant concepts

## 🐛 Debugging Tips

If a test fails:
1. Check the mathematical formula in the docstring
2. Verify input shapes match expectations
3. Check for numerical overflow/underflow
4. Ensure autograd is enabled
5. Verify requires_grad is set correctly

## 📈 Performance

Tests are designed to:
- Run quickly (< 1 second per test file)
- Use small tensors for speed
- Focus on correctness over performance
- Be deterministic (use fixed seeds where needed)

## 🔧 Maintenance

When adding new autograd operations:
1. Create new test file following naming convention
2. Include all test pattern types
3. Add mathematical documentation
4. Update this README
5. Add to integration tests

## 📚 References

- Automatic Differentiation: https://en.wikipedia.org/wiki/Automatic_differentiation
- Backpropagation: https://en.wikipedia.org/wiki/Backpropagation
- PyTorch Autograd: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
