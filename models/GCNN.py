import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNN(torch.nn.Module):
    def __init__(self, conv_channels=(18, 64, 32), fc_dims=(32, 16), dropout=0.3):
        """
        Implements the GCNN used in the following paper:
        https://www.sciencedirect.com/science/article/pii/S0169260722003327#bib0026 (A graph convolutional neural network for the automated detection of seizures in the neonatal EEG)
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.conv1 = GCNConv(conv_channels[0], conv_channels[1])
        self.conv2 = GCNConv(conv_channels[1], conv_channels[2])
        self.fc1 = torch.nn.Linear(conv_channels[2], fc_dims[0])
        self.fc2 = torch.nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = torch.nn.Linear(fc_dims[1], 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # Mean that handles batch properly
        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x