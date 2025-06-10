import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphSage(nn.Module):
    """
    GraphSage model for EEG classification.
    Args:
        in_channels (int): Number of input features.
        hidden_channels (int): Number of hidden features.
        num_layers (int): Number of layers.
        num_classes (int): Number of output classes.
    Returns:
        torch.Tensor: The output logits.
    """
    def __init__(self, in_channels=3000, hidden_channels=16, num_layers=2, num_classes=1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if num_layers > 1: 
            for _ in range(num_layers-1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        

        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        elif len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        x = global_mean_pool(x, batch)
        logits = self.lin(x)
        return logits.squeeze()     