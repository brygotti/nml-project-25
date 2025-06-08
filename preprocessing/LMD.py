import numpy as np
from PyLMD import LMD

lmd = LMD(
    max_smooth_iteration=1,
    max_envelope_iteration=20,
    envelope_epsilon=0.1,
    convergence_epsilon=0.1,
    max_num_pf=3
)

def lmd_features(x: np.ndarray) -> np.ndarray:
    """
    Implement feature extraction of the following paper:
    https://www.sciencedirect.com/science/article/pii/S0010482520302614#bib18 (Scalp EEG classification using deep Bi-LSTM network for seizure detection)
    As discussed in the documentation of the model, this preprocessing took too long to run due to the bad implementation of the LMD algorithm in PyLMD.
    """
    x = x.T
    channels = []
    for i in range(x.shape[0]):
        # Apply LMD to each channel
        if not x[i].any():
            # Array contains only zeros, lmd will fail, replace with zeros PFs
            PFs = np.zeros((3, x[i].shape[0]))
        else:
            PFs, _ = lmd.lmd(x[i])
            # Ensure PFs has the shape (3, n_samples)
            if PFs.shape[0] < 3:
                PFs = np.pad(PFs, ((0, 3 - PFs.shape[0]), (0, 0)))
        channels.append(PFs)
    channels = np.stack(channels)
    return extract_features(channels)


def extract_features(channels: np.ndarray) -> np.ndarray:
    """
    Get the features from the PFs
    Args:
        PFs: The PFs
    Returns:
        The features
    """
    channels = channels.astype(np.float128)
    epsilon = 1e-08
    return np.concatenate([
        np.max(channels, axis=-1),
        np.min(channels, axis=-1),
        np.median(channels, axis=-1),
        np.mean(channels, axis=-1),
        np.std(channels, axis=-1),
        np.var(channels, axis=-1),
        # Mean absolute deviation
        np.mean(np.abs(channels - np.mean(channels, axis=-1, keepdims=True)), axis=-1),
        # Root mean square
        np.sqrt(np.mean(channels**2, axis=-1)),
        # Skewness
        np.mean((channels - np.mean(channels, axis=-1, keepdims=True))**3, axis=-1) / (epsilon + (np.std(channels, axis=-1)**3)),
        # Kurtosis
        np.mean((channels - np.mean(channels, axis=-1, keepdims=True))**4, axis=-1) / (epsilon + (np.std(channels, axis=-1)**4)),
    ], axis=1).astype(np.float64)