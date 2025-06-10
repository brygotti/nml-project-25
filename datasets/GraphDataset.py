from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from seiz_eeg.dataset import EEGDataset
from tqdm import tqdm


standard_10_20_coords = {
    'FP1': [-2, 8, 6], 'FP2': [2, 8, 6], 'F7': [-6, 6, 5], 'F3': [-3, 6, 6], 'FZ': [0, 6, 6], 'F4': [3, 6, 6], 'F8': [6, 6, 5],
    'C3': [-3, 3, 5], 'CZ': [0, 3, 5], 'C4': [3, 3, 5],
    'P3': [-3, 0, 5], 'PZ': [0, 0, 5], 'P4': [3, 0, 5],
    'O1': [-2, -3, 4], 'O2': [2, -3, 4],
    'T3': [6, 0, 4], 'T4': [-6, 0, 4], 'T5': [6, -3, 4], 'T6': [-6, -3, 4]
}

class GraphDataset(InMemoryDataset):
    def __init__(self, root, dataset: EEGDataset, generate_graphs, mode='train', model_name=None):
        """
        Custom dataset for generating graphs from EEG data.
        """
        self.dataset = dataset
        self.generate_graphs = generate_graphs
        self.mode = mode
        self.model_name = model_name
        super().__init__(Path("tmp") / root, transform=None, pre_transform=None, pre_filter=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.mode}_graphs.pt"]

    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        distances = pd.read_csv(Path(self.dataset.signals_root.parts[0]) / "distances_3d.csv")
        electrodes = sorted(set(distances['from']) | set(distances['to']))

        positions_df = pd.DataFrame.from_dict(standard_10_20_coords, orient='index', columns=['x', 'y', 'z'])
        positions_df.index.name = 'name'
        positions_df = positions_df.loc[electrodes]  # align order

        train_path = Path(self.dataset.signals_root.parts[0]) / "train" if self.mode == 'train' else "test"
        
        if self.model_name == "GCNN":
            nodes_order = pd.read_parquet(self.dataset.signals_root / self.dataset.clips_df.iloc[0]["signals_path"]).columns.to_list()
            for i in tqdm(range(len(self.dataset)), desc="Generating graphs"):
                x, y = self.dataset[i]
                graph: Data = self.generate_graphs(distances, nodes_order, x)
                graph.y = y
                graphs.append(graph)

        
        if self.model_name == "EEGAT" or self.model_name == "GraphSage":
            for idx in tqdm(range(len(self.dataset.clips_df)), desc="Generating graphs"):
                row = self.dataset.clips_df.iloc[idx]

                try:
                    graph = self.generate_graphs(
                        row=row,
                        electrodes=electrodes,
                        positions_df=positions_df,
                        base_path=train_path,
                        test=(self.mode == 'test')
                    )
                    if graph is not None:
                        graphs.append(graph)
                except Exception as e:
                    print(f"[Skip idx {idx}] ❌ Error during graph generation: {e}")
                    continue

        if self.model_name == "GraphSage":
            for idx in tqdm(range(len(self.dataset.clips_df)), desc="Generating graphs"):
                row = self.dataset.clips_df.iloc[idx]

                try:
                    graph = self.generate_graphs(
                        row=row,
                        electrodes=electrodes,
                        positions_df=positions_df,
                        base_path=train_path,
                        test=(self.mode == 'test')
                    )
                    if graph is not None:
                        graphs.append(graph)
                except Exception as e:
                    print(f"[Skip idx {idx}] ❌ Error during graph generation: {e}")
                    continue

        self.save(graphs, self.processed_paths[0])
