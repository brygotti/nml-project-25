import pandas as pd
from torch_geometric.data import Data
import torch

def generate_distances_graph(distances: pd.DataFrame, nodes_order: list[str], epsilon: float = 0.9) -> Data:
    """
    Create an epsilon-neighborhood graph from a distance matrix.
    
    Parameters:
        distances (pd.DataFrame): A DataFrame containing the distance matrix with columns 'from', 'to', and 'distance'.
        nodes_order (list[str]): A list of node identifiers in the order they should appear in the graph.
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
    return Data(edge_index=edge_index.t().contiguous())