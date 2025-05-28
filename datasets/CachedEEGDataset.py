from seiz_eeg.dataset import EEGDataset
import numpy as np

class CachedEEGDataset(EEGDataset):
    def __init__(
        self,
        cached_array_path: str,
        *args,
        **kwargs
    ) -> None:
        """
        Custom dataset that uses precomputed arrays for EEG signals.
        Note: signal_transform will be ignored in this dataset.
        """
        with open(cached_array_path, 'rb') as f:
            self.cached_array = np.load(f)
        
        super().__init__(*args, **kwargs)

        if len(self.cached_array) != len(self):
            raise ValueError(
                f"Length of cached array ({len(self.cached_array)}) "
                f"does not match length of this dataset ({len(self)})."
            )

    def __getitem__(self, idx):
        _, y = super().__getitem__(idx)
        return self.cached_array[idx], y
