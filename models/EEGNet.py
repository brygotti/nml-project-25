import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet model for EEG signal classification.
    This model consists of a series of convolutional layers followed by
    optional GRU layers for temporal feature extraction.
    Args:
        input_dim (int): Number of EEG channels.
        num_samples (int): Number of time samples in the input.
        hidden_dim (int): Hidden dimension for GRU layers.
        num_layers (int): Number of GRU layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self,
                 input_dim=21, 
                 num_samples=256, 
                 hidden_dim=None,   
                 num_layers=None,   
                 num_classes=2, 
                 dropout=0.5):
        super(EEGNet, self).__init__()

        # Block 1: Temporal Convolution (shorter kernel)
        # Goal   : Detect temoral patterns like spikes, 
        #          sharp edges in each EEG channel independently
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(16),
        )

        # Block 2: Depthwise Convolution
        # Goal   : Mix the information between each EEG channel
        #          and learn the spatial patterns
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (input_dim, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),  # downsample less
            nn.Dropout(dropout)
        )

        # Block 3: Separable Convolution
        # Goal   : Extract large temporal patterns, now that the channel are mixed
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 64, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # Optional GRU after convolution
        # Goal   : Capture the long temporal dynamics  on the extracted features
        self.use_gru = True  # ← change to False if you don't want GRU
        self.gru_hidden = 64

        if self.use_gru:
            self.gru = nn.GRU(input_size=64, hidden_size=self.gru_hidden, batch_first=True)
            self.classify = nn.Linear(self.gru_hidden, num_classes)
        else:
            self.classify = None  # lazy init below

        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.depthwiseConv(x)
        print("After depthwiseConv:", x.shape)
        x = self.separableConv(x)
        print("After separableConv:", x.shape)

        if self.use_gru:
            # Adaptively pool to [B, C, 1, T] → ex: T = 32
            x = F.adaptive_avg_pool2d(x, (1, 32))  # [B, C, 1, 32]
            x = x.squeeze(2)         # [B, C, 32]
            x = x.permute(0, 2, 1)   # [B, 32, C]
            out, _ = self.gru(x)     # GRU over time
            x = out[:, -1, :]        # last timestep
            return self.classify(x)

        else:
            x = x.view(x.size(0), -1)
            if self.classify is None:
                in_features = x.shape[1]
                self.classify = nn.Linear(in_features, self.num_classes).to(x.device)
            return self.classify(x)