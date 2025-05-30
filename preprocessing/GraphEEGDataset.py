from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd
from pathlib import Path

class GraphEEGDataset(Dataset):
    def __init__(self, clips_df, edge_index, transform_fn, segment_len=64, base_path=None):
        self.clips = clips_df.reset_index(drop=True)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long) if isinstance(edge_index, np.ndarray) else edge_index
        self.segment_len = segment_len
        self.transform_fn = transform_fn
        self.base_path = Path(base_path or ".")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        row = self.clips.iloc[idx]
        signal_path = self.base_path / row["signals_path"]

        if not signal_path.exists():
            raise FileNotFoundError(f"Missing EEG file: {signal_path}")

        df = pd.read_parquet(signal_path)
        eeg = df.to_numpy()  # shape: [256, 19]

        label = row["label"]
        graphs = []
        T = eeg.shape[0] // self.segment_len

        for t in range(T):
            segment = eeg[t * self.segment_len : (t + 1) * self.segment_len]  # [segment_len, 19]
            features = self.transform_fn(segment)  # expected [19, D] or similar

            # Ensure features is a proper torch.Tensor
            if isinstance(features, pd.DataFrame):
                features = features.to_numpy()
            if isinstance(features, np.ndarray):
                features = torch.tensor(features.copy(), dtype=torch.float32)
            elif not isinstance(features, torch.Tensor):
                raise TypeError("Feature matrix must be a torch.Tensor or convertible to one.")

            x = features.contiguous()
            y = torch.tensor([label], dtype=torch.float32)

            edge_index_tensor = torch.tensor(self.edge_index, dtype=torch.long).contiguous() if isinstance(self.edge_index, np.ndarray) else self.edge_index.clone()
            graph = Data(x=x, edge_index=edge_index_tensor, y=y)
            graphs.append(graph)

        return graphs, label