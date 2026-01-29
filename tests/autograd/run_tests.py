#!/usr/bin/env python3
"""
Quick Test Runner for Autograd Module

This script provides a convenient way to run all autograd tests
and see a summary of results.

Usage:
    python tests/autograd/run_tests.py              # Run all tests
    python tests/autograd/run_tests.py --quick      # Run integration test only
    python tests/autograd/run_tests.py --category basic  # Run specific category
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_integration_test():
    """Run the comprehensive integration test."""
    print("\n" + "="*80)
    print("🚀 Running Autograd Integration Test")
    print("="*80)
    
    from tests.autograd.test_autograd_integration import test_module
    test_module()


def run_all_tests():
    """Run all individual test modules."""
    print("\n" + "="*80)
    print("🧪 Running All Autograd Tests")
    print("="*80)
    
    test_modules = [
        # Basic Operations
        ('test_add_backward', 'Addition'),
        ('test_mul_backward', 'Multiplication'),
        ('test_sub_backward', 'Subtraction'),
        ('test_div_backward', 'Division'),
        
        # Tensor Operations
        ('test_matmul_backward', 'Matrix Multiplication'),
        ('test_transpose_backward', 'Transpose'),
        ('test_permute_backward', 'Permute'),
        ('test_reshape_backward', 'Reshape'),
        
        # Advanced Operations
        ('test_embedding_backward', 'Embedding'),
        ('test_slice_backward', 'Slice'),
        ('test_sum_backward', 'Sum'),
        
        # Activation Functions
        ('test_relu_backward', 'ReLU'),
        ('test_sigmoid_backward', 'Sigmoid'),
        ('test_softmax_backward', 'Softmax'),
        ('test_gelu_backward', 'GELU'),
        
        # Loss Functions
        ('test_mse_backward', 'MSE Loss'),
        ('test_bce_backward', 'BCE Loss'),
        ('test_crossentropy_backward', 'Cross-Entropy Loss'),
    ]
    
    passed = []
    failed = []
    
    for module_name, display_name in test_modules:
        try:
            print(f"\n{'─'*80}")
            print(f"Testing: {display_name}")
            print(f"{'─'*80}")
            
            module = __import__(f'tests.autograd.{module_name}', fromlist=['test_module'])
            module.test_module()
            passed.append(display_name)
            print(f"✅ {display_name} - PASSED")
            
        except Exception as e:
            failed.append((display_name, str(e)))
            print(f"❌ {display_name} - FAILED")
            print(f"   Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {len(passed)}/{len(test_modules)}")
    print(f"❌ Failed: {len(failed)}/{len(test_modules)}")
    
    if passed:
        print("\n✅ Passed Tests:")
        for test in passed:
            print(f"   • {test}")
    
    if failed:
        print("\n❌ Failed Tests:")
        for test, error in failed:
            print(f"   • {test}")
            print(f"     Error: {error[:100]}...")
    
    print("\n" + "="*80)
    
    return len(failed) == 0


def run_category(category):
    """Run tests for a specific category."""
    categories = {
        'basic': ['add', 'mul', 'sub', 'div'],
        'tensor': ['matmul', 'transpose', 'permute', 'reshape'],
        'advanced': ['embedding', 'slice', 'sum'],
        'activation': ['relu', 'sigmoid', 'softmax', 'gelu'],
        'loss': ['mse', 'bce', 'crossentropy'],
    }
    
    if category not in categories:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(categories.keys())}")
        return False
    
    print(f"\n🧪 Running {category.upper()} tests...")
    
    for test_name in categories[category]:
        module_name = f'test_{test_name}_backward'
        try:
            module = __import__(f'tests.autograd.{module_name}', fromlist=['test_module'])
            module.test_module()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            return False
    
    return True


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print(__doc__)
        return
    
    if '--quick' in args:
        run_integration_test()
    elif '--category' in args:
        idx = args.index('--category')
        if idx + 1 < len(args):
            category = args[idx + 1]
            success = run_category(category)
            sys.exit(0 if success else 1)
        else:
            print("❌ Please specify a category after --category")
            sys.exit(1)
    else:
        # Run all tests
        success = run_all_tests()
        print("\n")
        run_integration_test()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
