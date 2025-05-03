import torch
import random
import numpy as np
import os
import pandas as pd 
from pathlib import Path
from seiz_eeg.dataset import EEGDataset

from torch.utils.data import DataLoader


def seed_everything(seed: int):
    # Python random module
    random.seed(seed)
    # Numpy random module
    np.random.seed(seed)
    # Torch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(config, mode='train'):
    clips = pd.read_parquet(Path(config["data_path"]) / f"{mode}/segments.parquet")
    if mode=='train':
        dataset = EEGDataset(
            clips,
            signals_root=Path(config["data_path"]) / f"{mode}",
            signal_transform=config["signal_transform"],
            prefetch=True,
        )
    elif mode=='test':
        dataset = EEGDataset(
            clips,
            signals_root=Path(config["data_path"]) / f"{mode}",
            signal_transform=config["signal_transform"],
            prefetch=True,
            return_id=True
        ) 

    return dataset

def create_submission(config, model, device, submission_name_csv='submission'):
    dataset_te=get_dataset(config, 'test')

    # Create DataLoader for the test dataset
    loader_te = DataLoader(dataset_te, batch_size=512, shuffle=False)
    # Generate the submission file for Kaggle

    # Set the model to evaluation mode
    model.eval()

    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch

            # Move the input data to the device (GPU or CPU)
            x_batch = x_batch.float().to(device)

            # Perform the forward pass to get the model's output logits
            logits = model(x_batch)

            # Convert logits to predictions.
            # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
            predictions = (logits > 0).int().cpu().numpy()

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv(f"{submission_name_csv}.csv", index=False)
    print(f"Kaggle submission file generated: {submission_name_csv}.csv")