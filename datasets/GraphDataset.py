from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from seiz_eeg.dataset import EEGDataset
from tqdm import tqdm

class GraphDataset(InMemoryDataset):
    def __init__(self, root, dataset: EEGDataset, generate_graphs, mode='train'):
        self.dataset = dataset
        self.generate_graphs = generate_graphs
        self.mode = mode
        super().__init__(Path("tmp") / root, transform=None, pre_transform=None, pre_filter=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.mode}_graphs.pt"]

    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        distances = pd.read_csv(Path(self.dataset.signals_root.parts[0]) / "distances_3d.csv")
        nodes_order = pd.read_parquet(self.dataset.signals_root / self.dataset.clips_df.iloc[0]["signals_path"]).columns.to_list()
        for i in tqdm(range(len(self.dataset)), desc="Generating graphs"):
            x, y = self.dataset[i]
            graph: Data = self.generate_graphs(distances, nodes_order, x)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            graph.x = x
            graph.y = y
            graphs.append(graph)

        self.save(graphs, self.processed_paths[0])