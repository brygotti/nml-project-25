import torch
import random
import numpy as np
import os
import pandas as pd 
from pathlib import Path
from seiz_eeg.dataset import EEGDataset
from datasets.EEGSessionDataset import EEGSessionDataset

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
    dataset = EEGDataset(
        clips,
        signals_root=Path(config["data_path"]) / f"{mode}",
        signal_transform=config["signal_transform"],
        prefetch=True,
        return_id=(mode == 'test')
    )
    if config["batch_size"] == 'session':
        # If batch size is 'session', we need to group clips by session
        # and create a custom dataset that handles this
        dataset = EEGSessionDataset(dataset)

    return dataset

def get_loader(config, dataset, mode='train'):
    return DataLoader(
        dataset,
        # Disable automatic batching if batch_size is 'session'
        batch_size=(None if config["batch_size"] == 'session' else config["batch_size"]),
        shuffle=(mode == 'train')
    )

def create_submission(config, model, device, submission_name_csv='submission'):
    dataset_te = get_dataset(config, mode='test')

    # Create DataLoader for the test dataset
    loader_te = get_loader(config, dataset_te, mode='test')
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
            if hasattr(model, 'custom_predict') and callable(model.custom_predict):
                # Allow for custom prediction logic
                predictions = model.custom_predict(logits)
            else:
                # Assuming sigmoid activation for binary classification
                predictions = (logits > 0)

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.int().cpu().numpy().flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    if all_ids[0].count("_") > 3:
        # If the ID contains more than 3 underscores, it means the dataset index was in the Kaggle format
        # And EEGDataset joined all characters with a "_". We need to remove those "_" in between the characters.
        all_ids = [x_id[::2] for x_id in all_ids]
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv(f"{submission_name_csv}.csv", index=False)
    print(f"Kaggle submission file generated: {submission_name_csv}.csv")