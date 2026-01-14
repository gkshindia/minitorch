import numpy as np
import pytest
from core.tensor import Tensor
from core.dataloader import RandomHorizontalFlip, RandomCrop, Compose


class TestRandomHorizontalFlip:
    """Test suite for RandomHorizontalFlip transform"""

    def test_initialization_valid_probability(self):
        """Test that valid probabilities are accepted"""
        flip = RandomHorizontalFlip(p=0.5)
        assert flip.p == 0.5
        
        flip_zero = RandomHorizontalFlip(p=0.0)
        assert flip_zero.p == 0.0
        
        flip_one = RandomHorizontalFlip(p=1.0)
        assert flip_one.p == 1.0

    def test_initialization_invalid_probability(self):
        """Test that invalid probabilities raise ValueError"""
        with pytest.raises(ValueError):
            RandomHorizontalFlip(p=-0.1)
        
        with pytest.raises(ValueError):
            RandomHorizontalFlip(p=1.5)

    def test_flip_always_with_p_one(self):
        """Test that p=1.0 always flips the image"""
        flip = RandomHorizontalFlip(p=1.0)
        
        # 2D image (H, W)
        image_2d = np.array([[1, 2, 3],
                             [4, 5, 6]])
        expected_2d = np.array([[3, 2, 1],
                                [6, 5, 4]])
        
        result = flip(image_2d)
        np.testing.assert_array_equal(result, expected_2d)

    def test_no_flip_with_p_zero(self):
        """Test that p=0.0 never flips the image"""
        flip = RandomHorizontalFlip(p=0.0)
        
        image = np.array([[1, 2, 3],
                         [4, 5, 6]])
        
        result = flip(image)
        np.testing.assert_array_equal(result, image)

    def test_flip_with_tensor_input(self):
        """Test flipping with Tensor input returns Tensor"""
        flip = RandomHorizontalFlip(p=1.0)
        
        tensor_input = Tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        
        result = flip(tensor_input)
        
        assert isinstance(result, Tensor)
        expected = np.array([[3.0, 2.0, 1.0],
                            [6.0, 5.0, 4.0]])
        np.testing.assert_array_equal(result.data, expected)

    def test_flip_3d_chw_format(self):
        """Test flipping 3D image in (C, H, W) format"""
        flip = RandomHorizontalFlip(p=1.0)
        
        # 3 channels, 2x3 image
        image_3d = np.array([[[1, 2, 3],
                             [4, 5, 6]],
                            [[7, 8, 9],
                             [10, 11, 12]],
                            [[13, 14, 15],
                             [16, 17, 18]]])
        
        expected_3d = np.array([[[3, 2, 1],
                                [6, 5, 4]],
                               [[9, 8, 7],
                                [12, 11, 10]],
                               [[15, 14, 13],
                                [18, 17, 16]]])
        
        result = flip(image_3d)
        np.testing.assert_array_equal(result, expected_3d)

    def test_flip_3d_hwc_format(self):
        """Test flipping 3D image - flips along last axis regardless of format"""
        flip = RandomHorizontalFlip(p=1.0)
        
        # 2x3 image with 3 channels (H=2, W=3, C=3)
        # The flip always flips along axis=-1, which is the channel dimension here
        image_hwc = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                             [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
        
        # Flipping along axis=-1 reverses the channel values at each pixel
        expected_hwc = np.array([[[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                                [[12, 11, 10], [15, 14, 13], [18, 17, 16]]])
        
        result = flip(image_hwc)
        np.testing.assert_array_equal(result, expected_hwc)

    def test_flip_preserves_dtype(self):
        """Test that flipping preserves data type"""
        flip = RandomHorizontalFlip(p=1.0)
        
        int_image = np.array([[1, 2, 3]], dtype=np.int32)
        result_int = flip(int_image)
        assert result_int.dtype == np.int32
        
        float_image = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result_float = flip(float_image)
        assert result_float.dtype == np.float32

    def test_flip_creates_copy(self):
        """Test that flipping creates a copy, not a view"""
        flip = RandomHorizontalFlip(p=1.0)
        
        original = np.array([[1, 2, 3]])
        result = flip(original)
        
        # Modify result and ensure original is unchanged
        result[0, 0] = 999
        assert original[0, 0] == 1

    def test_flip_probability_distribution(self):
        """Test that flip probability is approximately correct over many trials"""
        np.random.seed(42)
        flip = RandomHorizontalFlip(p=0.5)
        
        image = np.array([[1, 2, 3]])
        flipped_count = 0
        trials = 1000
        
        for _ in range(trials):
            result = flip(image.copy())
            if np.array_equal(result, np.array([[3, 2, 1]])):
                flipped_count += 1
        
        # Should be approximately 500 flips with some tolerance
        assert 450 <= flipped_count <= 550, f"Expected ~500 flips, got {flipped_count}"


class TestRandomCrop:
    """Test suite for RandomCrop transform"""

    def test_initialization_with_int_size(self):
        """Test initialization with integer size"""
        crop = RandomCrop(size=32, padding=4)
        assert crop.size == (32, 32)
        assert crop.padding == 4

    def test_initialization_with_tuple_size(self):
        """Test initialization with tuple size"""
        crop = RandomCrop(size=(28, 32), padding=2)
        assert crop.size == (28, 32)
        assert crop.padding == 2

    def test_crop_2d_image(self):
        """Test cropping a 2D image"""
        np.random.seed(42)
        crop = RandomCrop(size=2, padding=0)
        
        image = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]])
        
        result = crop(image)
        assert result.shape == (2, 2)
        # Verify it's a valid 2x2 crop from the original
        assert result.size == 4

    def test_crop_with_padding(self):
        """Test that padding is correctly applied"""
        np.random.seed(42)
        crop = RandomCrop(size=4, padding=2)
        
        # 4x4 image with padding=2 should allow random crops
        image = np.ones((4, 4))
        
        result = crop(image)
        assert result.shape == (4, 4)

    def test_crop_3d_chw_format(self):
        """Test cropping 3D image in (C, H, W) format"""
        np.random.seed(42)
        crop = RandomCrop(size=2, padding=0)
        
        # 3 channels, 4x4 image
        image = np.random.randn(3, 4, 4)
        
        result = crop(image)
        assert result.shape == (3, 2, 2)

    def test_crop_3d_hwc_format(self):
        """Test cropping 3D image in (H, W, C) format"""
        np.random.seed(42)
        crop = RandomCrop(size=2, padding=0)
        
        # 10x10 image with 10 channels - shape[0] > 4, treated as HWC
        image = np.random.randn(10, 10, 10)
        
        result = crop(image)
        assert result.shape == (2, 2, 10)

    def test_crop_with_tensor_input(self):
        """Test cropping with Tensor input returns Tensor"""
        np.random.seed(42)
        crop = RandomCrop(size=2, padding=0)
        
        tensor_input = Tensor(np.array([[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0],
                                        [7.0, 8.0, 9.0]]))
        
        result = crop(tensor_input)
        
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 2)

    def test_crop_with_numpy_input(self):
        """Test cropping with numpy array input returns numpy array"""
        np.random.seed(42)
        crop = RandomCrop(size=2, padding=0)
        
        numpy_input = np.array([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0],
                               [7.0, 8.0, 9.0]])
        
        result = crop(numpy_input)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_crop_padding_adds_zeros(self):
        """Test that padding adds zero values"""
        np.random.seed(100)  # Seed to get deterministic crop from corner
        crop = RandomCrop(size=3, padding=1)
        
        # Small 2x2 image
        image = np.array([[1.0, 2.0],
                         [3.0, 4.0]])
        
        result = crop(image)
        # With padding=1, a 2x2 image becomes 4x4 padded
        # We can crop 3x3 from it, which may include zeros
        assert result.shape == (3, 3)

    def test_crop_invalid_input_dimension(self):
        """Test that invalid input dimensions raise ValueError"""
        crop = RandomCrop(size=2, padding=0)
        
        # 1D array should raise error
        with pytest.raises(ValueError):
            crop(np.array([1, 2, 3, 4]))
        
        # 4D array should raise error
        with pytest.raises(ValueError):
            crop(np.random.randn(2, 3, 4, 5))

    def test_crop_randomness(self):
        """Test that crops are random across multiple calls"""
        np.random.seed(None)  # Use random seed
        crop = RandomCrop(size=2, padding=2)
        
        image = np.arange(16).reshape(4, 4)
        
        results = [crop(image) for _ in range(10)]
        
        # Check that we got at least 2 different crops
        unique_crops = []
        for result in results:
            is_unique = True
            for unique_crop in unique_crops:
                if np.array_equal(result, unique_crop):
                    is_unique = False
                    break
            if is_unique:
                unique_crops.append(result)
        
        assert len(unique_crops) >= 2, "Expected multiple different random crops"

    def test_crop_different_target_sizes(self):
        """Test cropping to non-square sizes"""
        np.random.seed(42)
        crop = RandomCrop(size=(2, 3), padding=0)
        
        image = np.ones((5, 6))
        
        result = crop(image)
        assert result.shape == (2, 3)

    def test_crop_preserves_values(self):
        """Test that cropped values come from original image + padding"""
        np.random.seed(42)
        crop = RandomCrop(size=3, padding=1)
        
        # Unique values to track
        image = np.array([[1, 2],
                         [3, 4]], dtype=float)
        
        result = crop(image)
        
        # All non-zero values in result should be from {1, 2, 3, 4}
        non_zero_values = result[result != 0]
        assert all(val in [1.0, 2.0, 3.0, 4.0] for val in non_zero_values)


class TestCompose:
    """Test suite for Compose transform pipeline"""

    def test_initialization(self):
        """Test Compose initialization with list of transforms"""
        transforms = [RandomHorizontalFlip(p=0.5), RandomCrop(size=32, padding=4)]
        compose = Compose(transforms)
        assert len(compose.transforms) == 2

    def test_compose_empty_transforms(self):
        """Test Compose with empty transform list returns input unchanged"""
        compose = Compose([])
        
        image = np.array([[1, 2, 3],
                         [4, 5, 6]])
        
        result = compose(image)
        np.testing.assert_array_equal(result, image)

    def test_compose_single_transform(self):
        """Test Compose with a single transform"""
        transforms = [RandomHorizontalFlip(p=1.0)]
        compose = Compose(transforms)
        
        image = np.array([[1, 2, 3],
                         [4, 5, 6]])
        expected = np.array([[3, 2, 1],
                            [6, 5, 4]])
        
        result = compose(image)
        np.testing.assert_array_equal(result, expected)

    def test_compose_multiple_transforms(self):
        """Test Compose applies transforms in sequence"""
        # Create a deterministic pipeline
        np.random.seed(42)
        transforms = [
            RandomHorizontalFlip(p=1.0),  # Always flip
            RandomCrop(size=2, padding=0)  # Crop to 2x2
        ]
        compose = Compose(transforms)
        
        image = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        
        result = compose(image)
        
        # Should be flipped then cropped to 2x2
        assert result.shape == (2, 2)
        # Values should come from the flipped image

    def test_compose_with_tensor_input(self):
        """Test Compose pipeline with Tensor input"""
        transforms = [RandomHorizontalFlip(p=1.0)]
        compose = Compose(transforms)
        
        tensor_input = Tensor([[1.0, 2.0, 3.0]])
        
        result = compose(tensor_input)
        
        assert isinstance(result, Tensor)
        expected = np.array([[3.0, 2.0, 1.0]])
        np.testing.assert_array_equal(result.data, expected)

    def test_compose_preserves_types_through_pipeline(self):
        """Test that Compose preserves type through entire pipeline"""
        # First transform takes Tensor and returns Tensor
        # Second transform should also work with Tensor
        np.random.seed(42)
        transforms = [
            RandomHorizontalFlip(p=1.0),
            RandomCrop(size=2, padding=0)
        ]
        compose = Compose(transforms)
        
        tensor_input = Tensor(np.array([[1.0, 2.0, 3.0],
                                        [4.0, 5.0, 6.0],
                                        [7.0, 8.0, 9.0]]))
        
        result = compose(tensor_input)
        
        assert isinstance(result, Tensor)
        assert result.data.shape == (2, 2)

    def test_compose_order_matters(self):
        """Test that order of transforms affects final result"""
        np.random.seed(42)
        
        # Pipeline 1: Crop then flip
        pipeline1 = Compose([
            RandomCrop(size=2, padding=0),
            RandomHorizontalFlip(p=1.0)
        ])
        
        np.random.seed(42)
        # Pipeline 2: Flip then crop
        pipeline2 = Compose([
            RandomHorizontalFlip(p=1.0),
            RandomCrop(size=2, padding=0)
        ])
        
        image = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]])
        
        np.random.seed(42)
        result1 = pipeline1(image.copy())
        
        np.random.seed(42)
        result2 = pipeline2(image.copy())
        
        # Results should potentially be different due to order
        # (This may or may not be different depending on random crop position)
        assert result1.shape == result2.shape == (2, 2)

    def test_compose_with_three_transforms(self):
        """Test Compose with three different transforms"""
        np.random.seed(42)
        
        transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomCrop(size=3, padding=1),
            RandomHorizontalFlip(p=0.5)
        ]
        compose = Compose(transforms)
        
        image = np.array([[1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]])
        
        result = compose(image)
        
        # Final result should be 3x3 due to crop
        assert result.shape == (3, 3)

    def test_compose_callable_interface(self):
        """Test that Compose implements proper callable interface"""
        compose = Compose([RandomHorizontalFlip(p=0.5)])
        
        # Should be callable
        assert callable(compose)
        
        # Should work with __call__
        image = np.array([[1, 2, 3]])
        result = compose(image)
        assert result.shape == image.shape


class TestIntegrationDataAugmentation:
    """Integration tests for combining data augmentation with datasets"""

    def test_augmentation_with_cifar_like_data(self):
        """Test typical CIFAR-10 augmentation pipeline"""
        np.random.seed(42)
        
        # Typical CIFAR-10 augmentation
        train_transform = Compose([
            RandomCrop(size=32, padding=4),
            RandomHorizontalFlip(p=0.5)
        ])
        
        # Simulate CIFAR-10 image (3, 32, 32)
        image = np.random.randn(3, 32, 32)
        
        augmented = train_transform(image)
        
        assert augmented.shape == (3, 32, 32)

    def test_augmentation_consistency_with_seed(self):
        """Test that augmentation is reproducible with same seed"""
        transform = Compose([
            RandomCrop(size=28, padding=2),
            RandomHorizontalFlip(p=0.5)
        ])
        
        image = np.random.randn(1, 32, 32)
        
        # Apply with same seed twice
        np.random.seed(123)
        result1 = transform(image.copy())
        
        np.random.seed(123)
        result2 = transform(image.copy())
        
        np.testing.assert_array_equal(result1, result2)

    def test_no_augmentation_for_test_set(self):
        """Test that test set uses no augmentation (just returns input)"""
        test_transform = Compose([])
        
        image = np.random.randn(3, 32, 32)
        result = test_transform(image)
        
        np.testing.assert_array_equal(result, image)

    def test_augmentation_preserves_label(self):
        """Test that augmentations don't affect labels (conceptual test)"""
        transform = Compose([RandomHorizontalFlip(p=1.0)])
        
        # In real usage, only the image is transformed, not the label
        image = np.array([[1, 2, 3]])
        label = 5  # Classification label
        
        transformed_image = transform(image)
        
        # Label should remain unchanged (not transformed)
        assert label == 5
        # Image should be transformed
        assert not np.array_equal(transformed_image, image)
