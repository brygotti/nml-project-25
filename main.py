import torch
import torch.nn as nn
from preprocessing.LMD import lmd_features
from preprocessing.FrequencyBandFiltering import frequency_bands_features
from preprocessing.FFTTransform import fft_filtering, time_filtering
from training_pipeline import *
from utils import *
from preprocessing.Graphs import generate_distances_graph, generate_single_graph

## Edit this config to change the training parameters
CONFIG = {
    "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": 512, # Can be set to "session" to use each recording session as a batch
    "num_epochs": 100, # Ignored if early stopping is enabled
    "optimizer": None, # Defaults to Adam with learning rate config["lr"]
    "lr": 1e-3,
    "model": "EEGNet",
    "criterion_fn": lambda weight_pos_class: nn.BCEWithLogitsLoss(), # weight_pos_class can be used in the loss function to reweight the loss for the positive class
    "signal_transform": fft_filtering,
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 20,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.05      # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    }
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_pipeline(CONFIG, device)

    create_submission(CONFIG, model, device)


   


