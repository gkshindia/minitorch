import numpy as np
import pytest
from core.tensor import Tensor
from core.layers import Dropout


class TestDropoutInitialization:
    def test_creation_basic(self):
        d = Dropout(0.5)
        assert d.p == 0.5

    @pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_boundary_valid_probs(self, p):
        d = Dropout(p)
        assert d.p == p

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            Dropout(-1e-3)
        with pytest.raises(ValueError):
            Dropout(1.0 + 1e-3)


class TestDropoutForwardBasics:
    def test_inference_passthrough(self):
        x = Tensor([1, 2, 3, 4])
        d = Dropout(0.5)
        y = d.forward(x, training=False)
        assert np.array_equal(x.data, y.data)

    def test_zero_probability_training(self):
        x = Tensor([1, 2, 3, 4])
        d = Dropout(0.0)
        y = d.forward(x, training=True)
        assert np.array_equal(x.data, y.data)

    def test_full_probability_training(self):
        x = Tensor([1, 2, 3, 4])
        d = Dropout(1.0)
        y = d.forward(x, training=True)
        assert np.allclose(y.data, 0)

    def test_returns_tensor(self):
        x = Tensor(np.ones((10,)))
        d = Dropout(0.5)
        y = d.forward(x, training=True)
        assert isinstance(y, Tensor)

    def test_no_parameters(self):
        d = Dropout(0.5)
        assert d.parameters() == []


class TestDropoutShapesAndDtypes:
    def test_shape_preservation_1d(self):
        x = Tensor(np.arange(10, dtype=np.float32))
        d = Dropout(0.3)
        y = d.forward(x, training=True)
        assert y.shape == x.shape
        assert y.dtype == np.float32

    def test_shape_preservation_2d_batch(self):
        x = Tensor(np.random.randn(32, 64))
        d = Dropout(0.4)
        y = d.forward(x, training=True)
        assert y.shape == (32, 64)

    def test_shape_preservation_3d(self):
        x = Tensor(np.random.randn(8, 16, 4))
        d = Dropout(0.5)
        y = d.forward(x, training=True)
        assert y.shape == (8, 16, 4)


class TestDropoutScalingAndExpectation:
    def test_scaling_values(self):
        np.random.seed(7)
        p = 0.6
        keep = 1.0 - p
        x = Tensor(np.ones((1000,), dtype=np.float32))
        d = Dropout(p)
        y = d.forward(x, training=True)
        survivors = y.data[y.data != 0]
        expected_survivor_value = 1.0 / keep
        assert np.allclose(survivors, expected_survivor_value)

    def test_expectation_matches_input_mean(self):
        np.random.seed(123)
        p = 0.3
        x = Tensor(np.random.randn(5000).astype(np.float32))
        d = Dropout(p)
        y = d.forward(x, training=True)
        # With inverted dropout scaling, E[y] ≈ x element-wise; compare means
        assert np.isclose(y.data.mean(), x.data.mean(), atol=1e-2)

    def test_statistics_survivor_count_bounds(self):
        np.random.seed(42)
        p = 0.5
        keep = 1.0 - p
        x = Tensor(np.ones((2000,), dtype=np.float32))
        d = Dropout(p)
        y = d.forward(x, training=True)
        count = np.count_nonzero(y.data)
        n = x.size
        expected = n * keep
        std = np.sqrt(n * keep * (1.0 - keep))
        lo, hi = expected - 3 * std, expected + 3 * std
        assert lo < count < hi


class TestDropoutDeterminism:
    def test_reproducible_with_seed(self):
        p = 0.4
        x = Tensor(np.ones((100,), dtype=np.float32))
        d = Dropout(p)
        np.random.seed(99)
        y1 = d.forward(x, training=True)
        np.random.seed(99)
        y2 = d.forward(x, training=True)
        assert np.array_equal(y1.data, y2.data)


class TestDropoutNonMutation:
    def test_input_not_mutated(self):
        x = Tensor(np.random.randn(64).astype(np.float32))
        x_copy = x.data.copy()
        d = Dropout(0.5)
        _ = d.forward(x, training=True)
        assert np.allclose(x.data, x_copy)


class TestDropoutRepr:
    def test_repr_contains_p(self):
        d = Dropout(0.25)
        s = repr(d)
        assert "Dropout" in s and "0.25" in s
