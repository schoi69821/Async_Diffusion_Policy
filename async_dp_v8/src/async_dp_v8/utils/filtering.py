"""Signal filtering utilities."""
import numpy as np


def exponential_moving_average(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply EMA to 1D or 2D data along axis 0."""
    result = data.copy()
    for i in range(1, len(result)):
        result[i] = alpha * result[i] + (1 - alpha) * result[i - 1]
    return result


def low_pass_filter(data: np.ndarray, cutoff_ratio: float = 0.1) -> np.ndarray:
    """Simple moving average low-pass filter."""
    window = max(1, int(1.0 / cutoff_ratio))
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    if data.ndim == 1:
        return np.convolve(data, kernel, mode="same")
    else:
        result = np.zeros_like(data)
        for j in range(data.shape[1]):
            result[:, j] = np.convolve(data[:, j], kernel, mode="same")
        return result
