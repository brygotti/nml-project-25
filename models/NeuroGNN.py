import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class NeuroGNN(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=32, out_channels=1, dropout=0.3):
        super(NeuroGNN, self).__init__()

        # Graph Attention Layers
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, concat=True)
        self.gat3 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=1, concat=True)

        # Optional GRU for temporal modeling
        self.use_gru = True
        if self.use_gru:
            self.gru = nn.GRU(input_size=hidden_channels, hidden_size=64, batch_first=True)
            self.classifier = nn.Linear(64, out_channels)
        else:
            self.classifier = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, batched_graph_sequence):
        # Input: list of [Batch] objects for each time step
        temporal_embeddings = []

        for batch in batched_graph_sequence:
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

            x = F.elu(self.gat1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.gat2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.gat3(x, edge_index))

            pooled = global_mean_pool(x, batch_idx)  # shape: [batch_size, hidden_channels]
            temporal_embeddings.append(pooled)

        x_seq = torch.stack(temporal_embeddings, dim=1)  # shape: [batch_size, T, hidden_channels]

        if self.use_gru:
            out, _ = self.gru(x_seq)
            x = out[:, -1, :]  # Use final time step
        else:
            x = x_seq.mean(dim=1)

        return self.classifier(x)