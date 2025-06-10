import torch
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

import torch.nn as nn
import torch.nn.functional as F

class EEGGAT(nn.Module):
    """
    EEGGAT: A neural network model for EEG signal processing using temporal convolutions and Graph Attention Networks (GAT).
    This model combines temporal convolutional layers for feature extraction from EEG signals with multiple GAT layers to capture spatial dependencies between EEG channels represented as a graph.
    Args:
        in_channels (int): Number of input channels (features) per node.
        gat_out_channels (int, optional): Number of output channels for each GAT head. Default is 32.
        heads (int, optional): Number of attention heads in each GAT layer. Default is 4.
        proj_len (int, optional): Length to which temporal features are projected via adaptive pooling. Default is 128.
    Layers:
        - Three temporal convolutional layers with batch normalization and PReLU activation.
        - Depthwise convolution for channel-wise feature extraction.
        - Adaptive average pooling to project temporal features to a fixed length.
        - Three GATConv layers (from torch_geometric) with batch normalization and PReLU activation.
        - Residual connections between GAT layers.
        - Global mean pooling to aggregate node features into a graph-level representation.
        - Fully connected classifier with dropout and ReLU activation.
    Forward Args:
        data (torch_geometric.data.Data): Input graph data object containing:
            - x: Node feature matrix of shape [num_nodes, in_channels].
            - edge_index: Graph connectivity in COO format.
            - batch: Batch vector assigning each node to a specific graph.
    Returns:
        torch.Tensor: Output logits of shape [batch_size], representing the predicted value for each graph in the batch.
    """

    def __init__(self, in_channels, gat_out_channels=32, heads=4, proj_len=128):
        super(EEGGAT, self).__init__()
        self.proj_len = proj_len

        # Temporal convolutional layers for feature extraction
        self.temporal_conv1 = nn.Conv2d(1, 30, kernel_size=(1, 15), padding=(0, 7))
        self.temporal_conv2 = nn.Conv2d(30, 60, kernel_size=(1, 7), padding=(0, 3))
        self.temporal_conv3 = nn.Conv2d(60, 90, kernel_size=(1, 3), padding=(0, 1))

        # Batch normalization for temporal conv layers
        self.bn1 = nn.BatchNorm2d(30)
        self.bn2 = nn.BatchNorm2d(60)
        self.bn3 = nn.BatchNorm2d(90)

        # PReLU activations for temporal conv layers
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        # Depthwise convolution for channel-wise feature extraction
        self.depthwise = nn.Conv2d(90, 90, kernel_size=(1, 1), groups=90)
        self.depth_bn = nn.BatchNorm2d(90)
        self.prelu_depth = nn.PReLU()

        # Adaptive average pooling to project temporal features
        self.temporal_pool = nn.AdaptiveAvgPool1d(proj_len)

        # GATConv layers for graph attention
        self.gat1 = GATConv(proj_len, gat_out_channels, heads=heads, concat=True, dropout=0.3)
        self.gat2 = GATConv(gat_out_channels * heads, gat_out_channels, heads=heads, concat=True, dropout=0.3)
        self.gat3 = GATConv(gat_out_channels * heads, gat_out_channels, heads=heads, concat=True, dropout=0.3)

        # Batch normalization for GAT layers
        self.bn_g1 = BatchNorm(gat_out_channels * heads)
        self.bn_g2 = BatchNorm(gat_out_channels * heads)
        self.bn_g3 = BatchNorm(gat_out_channels * heads)

        # PReLU activations for GAT layers
        self.prelu_g1 = nn.PReLU()
        self.prelu_g2 = nn.PReLU()
        self.prelu_g3 = nn.PReLU()

        # Final classifier: two linear layers with dropout and ReLU
        self.classifier = nn.Sequential(
            nn.Linear(gat_out_channels * heads, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        # Unpack data object
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Temporal feature extraction with 2D convolutions, batch norm, and PReLU
        x_raw = x.unsqueeze(1).unsqueeze(1)  # [num_nodes, 1, 1, in_channels]
        x_raw = self.prelu1(self.bn1(self.temporal_conv1(x_raw)))
        x_raw = self.prelu2(self.bn2(self.temporal_conv2(x_raw)))
        x_raw = self.prelu3(self.bn3(self.temporal_conv3(x_raw)))

        # Depthwise convolution for channel-wise feature extraction
        x_raw = self.prelu_depth(self.depth_bn(self.depthwise(x_raw)))

        # Remove singleton dimension and project temporal features to fixed length
        x_raw = x_raw.squeeze(2)  # [num_nodes, channels, time]
        x_raw = self.temporal_pool(x_raw)  # [num_nodes, channels, proj_len]
        x_raw = x_raw.mean(dim=1)  # Aggregate over channels: [num_nodes, proj_len]

        # Graph Attention Network (GAT) layers with batch norm, PReLU, and residual connections
        x = x_raw
        x = self.prelu_g1(self.bn_g1(self.gat1(x, edge_index)))
        x = self.prelu_g2(self.bn_g2(self.gat2(x, edge_index))) + x  # Residual connection
        x = self.prelu_g3(self.bn_g3(self.gat3(x, edge_index))) + x  # Residual connection

        # Global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)

        # Final classifier
        return self.classifier(x).squeeze(-1)