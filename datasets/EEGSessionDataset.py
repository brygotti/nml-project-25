from torch.utils.data import Dataset
from seiz_eeg.dataset import EEGDataset
import numpy as np
import torch

class EEGSessionDataset(Dataset):
    def __init__(self, eeg_dataset: EEGDataset):
        """
        Custom dataset that groups EEG clips by session.
        Args:
            eeg_dataset (EEGDataset): The EEG dataset containing clips.
        """
        self.eeg_dataset = eeg_dataset
        self.clips_df = eeg_dataset.clips_df
        self.session_to_indices = self._group_indices_by_session()

    def _group_indices_by_session(self):
        session_map = {}
        for i, idx in enumerate(self.clips_df.index):
            patient, session, _ = idx
            session_id = f"{patient}_{session}"
            session_map.setdefault(session_id, []).append(i)
        return list(session_map.values())

    def __len__(self):
        return len(self.session_to_indices)

    def __getitem__(self, idx):
        clip_indices = self.session_to_indices[idx]
        session_clips = np.stack([self.eeg_dataset[i] for i in clip_indices])
        return torch.from_numpy(session_clips)