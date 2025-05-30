import torch.nn as nn
from preprocessing.FFTTransform import fft_filtering
from training_pipeline import *
from utils import *


## Add you config file here 
CONFIG_EEGNet = {
    "data_path": "/home/ogut/data",
    "batch_size": 512,
    "num_epochs": 100,
    "lr": 1e-3,
    "model": "EEGNet",
    "model_params": {'input_dim': 19, 'num_samples': 256, 'dropout': 0.3},
    "criterion_fn": nn.BCEWithLogitsLoss,
    "signal_transform": fft_filtering,
    "k_folds": 5
}

CONFIG_NEUROGNN = {
    "data_path": "/home/ogut/data",
    "batch_size": 32,  # ⚠️ tu dois réduire (graphe + séquence = + gourmand)
    "num_epochs": 100,
    "lr": 1e-3,
    "model": "NeuroGNN",
    "model_params": {
        "in_channels": 8,  # ← nb de features par électrode (ex: log-FFT), adapte selon ton transform
        "hidden_channels": 32,
        "dropout": 0.3
    },
    "criterion_fn": nn.BCEWithLogitsLoss,
    "signal_transform": fft_filtering,  # OK si tu as modifié pour retourner [19, 8] par segment
    "k_folds": 5,
    "segment_len": 64  # ← longueur d’un segment dans la fenêtre EEG (ex: 256 // 64 = 4 graphes)
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, final_metrics = train_pipeline(CONFIG_NEUROGNN, device)

    create_submission(CONFIG_NEUROGNN, model, device)

