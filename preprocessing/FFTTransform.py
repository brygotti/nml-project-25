from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import signal


bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)


def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

def fft_node_features(segment: np.ndarray) -> np.ndarray:
    """Transform EEG segment [T, 19] into node features [19, D]"""
    ffted = fft_filtering(segment)  # shape [F, 19]
    return ffted.mean(axis=0)  # [19] → 1 feature per node


def fft_multi_band(segment: np.ndarray) -> np.ndarray:
    """Transform EEG segment [T, 19] into node features [19, F]"""
    ffted = fft_filtering(segment)
    return ffted.T  # [19, F]
