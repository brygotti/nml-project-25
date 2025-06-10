import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from utils import bandpass_filter
import os

def generate_distances_graph(
        distances: pd.DataFrame,
        nodes_order: list[str],
        x: np.ndarray,
        epsilon: float = 0.9,
    ) -> Data:
    """
    Create an epsilon-neighborhood graph from a distance matrix.
    
    Parameters:
        distances (pd.DataFrame): A DataFrame containing the distance matrix with columns 'from', 'to', and 'distance'.
        nodes_order (list[str]): A list of node identifiers in the order they should appear in the graph.
        x (np.ndarray): The node features, which will be assigned to the graph.
        epsilon (float): The threshold distance to consider for edges.
        
    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    edges = []
    # Add edges based on the epsilon threshold
    for index, row in distances.iterrows():
        node1 = nodes_order.index(row['from'])
        node2 = nodes_order.index(row['to'])
        distance = row['distance']
        if distance < epsilon and node1 != node2:
            # No self-loops
            edges.append([node1, node2])
    
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.from_numpy(x)
    return Data(x=x, edge_index=edge_index.t().contiguous())


def build_knn_edge_index_from_positions(positions_df, k=4):
    pos = positions_df[['x', 'y', 'z']].values
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(pos)
    _, indices = nbrs.kneighbors(pos)

    edge_list = []
    for i in range(len(pos)):
        for j in indices[i][1:]:  # skip self
            edge_list.append([i, j])
            edge_list.append([j, i])

    return torch.tensor(edge_list, dtype=torch.long).T

def build_graph_dataset(df, electrodes, positions_df, base_path="../data/train/", test=False):
    """
    Build a graph dataset from the given DataFrame and electrode positions.
    Parameters:
        df (pd.DataFrame): DataFrame containing the segments with columns:
            - 'signals_path': Path to the signals file.
            - 'start_time': Start time of the segment in seconds.
            - 'end_time': End time of the segment in seconds.
            - 'sampling_rate': Sampling rate of the signals.
            - 'label': Label for the segment (optional, set to 0 for test).
        electrodes (list): List of electrode names to use as nodes.
        positions_df (pd.DataFrame): DataFrame containing electrode positions with columns:
            - 'name': Electrode name.
            - 'x', 'y', 'z': Coordinates of the electrode.
        base_path (str): Base path where signal files are stored.
        test (bool): If True, labels will be set to 0 for all segments.
    """

    dataset = []
    n_nodes = len(electrodes)

    # Fully connected undirected graph
    # edge_index = torch.combinations(torch.arange(n_nodes), r=2).T
    # edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # make bidirectional
    edge_index = build_knn_edge_index_from_positions(positions_df, k=4)

    loaded_files = {}
    invalid_rows = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if test:
            row["label"] = 0  # dummy label for test


        path = os.path.join(str(base_path), row["signals_path"])
        if path not in loaded_files:
            try:
                loaded_files[path] = pd.read_parquet(path)
            except Exception as e:
                print(f"❌ Error reading file {path}: {e}")
                continue

        full_signals = loaded_files[path]
        start = int(row["start_time"] * row["sampling_rate"])
        end = int(row["end_time"] * row["sampling_rate"])
        segment = full_signals.iloc[start:end]

        if segment.shape[0] != 3000:
            invalid_rows += 1
            continue

        try:
            segment_np = segment[electrodes].values
            segment_np = bandpass_filter(segment_np, fs=row["sampling_rate"])
            segment = pd.DataFrame(segment_np, columns=electrodes)
            segment = (segment - segment.mean()) / (segment.std() + 1e-6)

            x = torch.tensor(segment.values.T, dtype=torch.float32)  # shape [n_nodes, time]

            if not torch.isfinite(x).all():
                raise ValueError("NaN or Inf in input")

        except Exception as e:
            print(f"[Skip row {idx}] ❌ Normalization failed: {e}")
            continue

        y = torch.tensor([int(row["label"])], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y)
        dataset.append(data)

    print(f"✅ Valid segments: {len(dataset)} | 🚫 Invalid segments: {invalid_rows}")
    return dataset



def generate_single_graph(row, electrodes, positions_df, base_path, test=False):
    edge_index = build_knn_edge_index_from_positions(positions_df, k=4)

    path = os.path.join(str(base_path), row["signals_path"])
    try:
        signals = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ Error reading file {path}: {e}")
        return None

    start = int(row["start_time"] * row["sampling_rate"])
    end = int(row["end_time"] * row["sampling_rate"])
    segment = signals.iloc[start:end]

    if segment.shape[0] != 3000:
        return None

    try:
        segment_np = segment[electrodes].values
        segment_np = bandpass_filter(segment_np, fs=row["sampling_rate"])
        segment = pd.DataFrame(segment_np, columns=electrodes)
        segment = (segment - segment.mean()) / (segment.std() + 1e-6)
        x = torch.tensor(segment.values.T, dtype=torch.float32)
        if not torch.isfinite(x).all():
            raise ValueError("Non-finite in input")
    except Exception as e:
        print(f"❌ Normalization failed: {e}")
        return None

    y = torch.tensor([int(row["label"] if not test else 0)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=None, y=y)

