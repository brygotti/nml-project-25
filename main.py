import torch
import torch.nn as nn
from preprocessing.LMD import lmd_features
from preprocessing.FrequencyBandFiltering import frequency_bands_features
from preprocessing.FFTTransform import fft_filtering
from training_pipeline import *
from utils import *
from preprocessing.Graphs import generate_distances_graph

data_path = "data"
DATA_ROOT = Path(data_path)
clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")

n_negatives = clips_tr['label'].value_counts()[0]
n_positives = clips_tr['label'].value_counts()[1]
pos_weight = n_negatives / n_positives

print(f"Number of negatives: {n_negatives}, Number of positives: {n_positives}")
print(f"Positive weight for loss function: {pos_weight}")

## Add you config file here 
CONFIG = {
    "k_folds": 3, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": 512,
    "num_epochs": 100,
    "optimizer": None, # defaults to Adam with learning rate config["lr"]
    "lr": 1e-4,
    "model": "GCNN",
    "model_params": {
        "conv_channels": (10, 64, 32),
    },
    "criterion_fn": lambda: nn.BCEWithLogitsLoss(), #pos_weight=torch.tensor(pos_weight)),
    # "cached_preprocessing": {
    #     "train": "tmp/lmd_train_arrays.npy",
    #     "test": "tmp/lmd_test_arrays.npy"
    # },
    "signal_transform": lambda x: frequency_bands_features(x, flatten=False),
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 10,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.1       # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    },
    "graph_cache_name": "distances_frequency_bands_features",
    "generate_graph": lambda distances, nodes_order, x: generate_distances_graph(distances, nodes_order), # Set this if you want to work with graphs
    # the generate_graph function should return a Data object with edge_index defined. The signal_transform output should also have 2 dimensions: (n_nodes, n_features)
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_pipeline(CONFIG, device)

    create_submission(CONFIG, model, device)


   


