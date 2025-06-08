import numpy as np
import torch
import torch.nn as nn

class TemporalBiLSTM(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=1, dropout=0.35):
        """
        Implement Bi-LSTM of the following paper:
        https://www.sciencedirect.com/science/article/pii/S0010482520302614#bib18 (Scalp EEG classification using deep Bi-LSTM network for seizure detection)
        The preprocessing of this model could not be implemented due to the lack of a good python library for the required algorithm.
        The model was tested with other preprocessing methods, but resulted in worse performance than the SimpleLSTM model, so it was
        not included in the final report.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=False,
            bidirectional=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(2*hidden_dim, 1)  # Output for binary classification

    def forward(self, x):
        """
        x shape: [seq_len, n_channels, input_dim]
        """
        out, (h_n, c_n) = self.lstm(x)  # out shape: [seq_len, n_channels, 2*hidden_dim]
        logits = self.fc(out).squeeze(-1)  # logits shape: [seq_len, n_channels]
        return logits
    
    def custom_loss(self, criterion, logits, labels):
        """
        logits shape: [seq_len, n_channels]
        labels shape: [seq_len, 1]
        """
        labels = labels.squeeze(-1)
        labels = labels.repeat(logits.shape[1])  # Repeat labels for each channel
        logits = logits.permute(1, 0).flatten()  # Reshape logits to [n_channels * seq_len]
        loss = criterion(logits, labels) # Compute loss for each channel at once

        return loss
    
    def custom_predict(self, logits):
        """
        logits shape: [seq_len, n_channels]
        """
        # From the paper: each channel is classified independently, and the sample is classified as positive if at least 9 channels are classified as positive.
        preds = torch.sum(logits > 0, dim=-1) >= 9
        return preds
