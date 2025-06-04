import torch
import torch.nn as nn
from preprocessing.LMD import lmd_features
from preprocessing.FrequencyBandFiltering import frequency_bands_features
from preprocessing.FFTTransform import fft_filtering
from training_pipeline import *
from utils import *

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
    # "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": 512,
    "num_epochs": 100,
    "optimizer": None, # defaults to Adam with learning rate config["lr"]
    "lr": 1e-4,
    "model": "SimpleLSTM",
    "criterion_fn": lambda: nn.BCEWithLogitsLoss(), #pos_weight=torch.tensor(pos_weight)),
    # "cached_preprocessing": {
    #     "train": "tmp/lmd_train_arrays.npy",
    #     "test": "tmp/lmd_test_arrays.npy"
    # },
    "signal_transform": fft_filtering,
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 50,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.1       # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    },
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = train_pipeline(CONFIG, device)

    create_submission(CONFIG, model, device)


   


