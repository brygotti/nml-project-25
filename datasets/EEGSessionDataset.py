import pandas as pd
from torch.utils.data import Dataset
from seiz_eeg.dataset import EEGDataset
import numpy as np
import torch

def group_indices_by_session(clips_df: pd.DataFrame) -> list[list[int]]:
    """
    Groups indices of EEG clips by session based on the DataFrame index.
    Args:
        clips_df (pd.DataFrame): DataFrame containing EEG clips with indices formatted as 'patient_session_clip' or similar.
    Returns:
        list[list[int]]: A list of lists, where each inner list contains indices of clips belonging to the same session.
    """
    session_map = {}
    for i, idx in enumerate(clips_df.index):
        if isinstance(idx, str):
            parts = idx.split("_")
            patient = parts[0]
            session = f"{parts[1]}_{parts[2]}"
        else:
            patient, session, _ = idx
        session_id = f"{patient}_{session}"
        session_map.setdefault(session_id, []).append(i)
    return list(session_map.values())

class EEGSessionDataset(Dataset):
    def __init__(self, eeg_dataset: EEGDataset):
        """
        Custom dataset that groups EEG clips by session.
        Args:
            eeg_dataset (EEGDataset): The EEG dataset containing clips.
        """
        self.eeg_dataset = eeg_dataset
        self.clips_df = eeg_dataset.clips_df
        self.session_to_indices = group_indices_by_session(self.clips_df)

    def __len__(self):
        return len(self.session_to_indices)

    def __getitem__(self, idx):
        clip_indices = self.session_to_indices[idx]
        dataset_items = [self.eeg_dataset[i] for i in clip_indices]
        session_clips = torch.from_numpy(np.stack([item[0] for item in dataset_items]))
        session_labels = [item[1] for item in dataset_items]
        # Labels might be transformed to a different type (for example if return_id is True or if label_transform is used)
        # Check if the first label is an int and convert to tensor if so
        is_int_label = isinstance(session_labels[0], (int, np.integer))
        if is_int_label:
            session_labels = torch.from_numpy(np.array(session_labels))
        return session_clips, session_labels