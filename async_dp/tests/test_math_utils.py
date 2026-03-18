import numpy as np
from src.utils.math_utils import linear_interpolate, apply_ema_filter
def test_linear_interpolation():
    assert np.allclose(linear_interpolate(np.array([0.]), np.array([10.]), 0.5), np.array([5.]))
def test_ema_filter():
    assert apply_ema_filter(np.array([10.]), None)[0] == 10.