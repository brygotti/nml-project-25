import torch.nn as nn
from preprocessing.FFTTransform import fft_filtering
from training_pipeline import *
from utils import *


## Add you config file here 
CONFIG = {
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, final_metrics = train_pipeline(CONFIG, device)

    create_submission(CONFIG, model, device)

