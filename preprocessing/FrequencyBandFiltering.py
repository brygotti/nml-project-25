import numpy as np

def frequency_bands_features(x: np.ndarray) -> np.ndarray:
    """Implement feature extraction of the following paper:
    https://link.springer.com/article/10.1007/s40747-021-00627-z
    """
    x = np.fft.fft(x, axis=0)

    win_len = x.shape[0]
    delta = x[int(0.5 * win_len // 250) : 4 * win_len // 250].T
    theta = x[int(4 * win_len // 250) : 8 * win_len // 250].T
    alpha = x[int(8 * win_len // 250) : 12 * win_len // 250].T
    beta = x[int(12 * win_len // 250) : 30 * win_len // 250].T
    gamma = x[int(30 * win_len // 250) :].T

    return np.concatenate([
        power(delta),
        power(theta),
        power(alpha),
        power(beta),
        power(gamma),
        mean_amplitude(delta),
        mean_amplitude(theta),
        mean_amplitude(alpha),
        mean_amplitude(beta),
        mean_amplitude(gamma),
    ], axis=1).astype(np.float64).flatten()

def power(x: np.ndarray) -> np.ndarray:
    """Compute the power of the signal"""
    return np.mean(np.abs(x)**2, axis=-1, keepdims=True)

def mean_amplitude(x: np.ndarray) -> np.ndarray:
    """Compute the mean amplitude of the signal"""
    return np.mean(np.abs(x), axis=-1, keepdims=True)