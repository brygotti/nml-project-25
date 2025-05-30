from typing import OrderedDict
import numpy as np
import torch.nn as nn

class TemporalLSTMCombinedChannels(nn.Module):
    def __init__(self, input_dim=190, hidden_dims=(128, 64), fc2_dim=16, dropout=0.25):
        """
        Implement LSTM of the following paper:
        https://link.springer.com/article/10.1007/s40747-021-00627-z (Two-layer LSTM network-based prediction of epileptic seizures using EEG spectral features)
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0])
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1])
        self.fc1 = nn.Linear(hidden_dims[1], hidden_dims[1])
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dims[1], fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, 1)


    def forward(self, x):
        """
        x shape: [seq_len, input_dim]
        Returns:
            out shape: [seq_len, 1]
        """
        out, (h_n, c_n) = self.lstm1(x)  # out shape: [seq_len, hidden_dims[0]]
        out, (h_n, c_n) = self.lstm2(out) # out shape: [seq_len, hidden_dims[1]]
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out) # out shape: [seq_len, 1]
        return out
    
