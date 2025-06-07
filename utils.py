import torch
import random
import numpy as np
import os
import pandas as pd 
from pathlib import Path
from seiz_eeg.dataset import EEGDataset
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets.EEGSessionDataset import EEGSessionDataset, group_indices_by_session
from datasets.CachedEEGDataset import CachedEEGDataset
from datasets.GraphDataset import GraphDataset

from torch.utils.data import DataLoader, Subset, random_split
from torch import optim

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PygDataLoader


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
    if config.get("cached_preprocessing") is not None:
        dataset = CachedEEGDataset(
            config["cached_preprocessing"][mode],
            clips,
            signals_root=Path(config["data_path"]) / f"{mode}",
            signal_transform=config.get("signal_transform"),
            prefetch=True,
            return_id=(mode == 'test')
        )
    else:
        dataset = EEGDataset(
            clips,
            signals_root=Path(config["data_path"]) / f"{mode}",
            signal_transform=config.get("signal_transform"),
            prefetch=True,
            return_id=(mode == 'test')
        )
    if config["batch_size"] == 'session':
        # If batch size is 'session', we need to group clips by session
        # and create a custom dataset that handles this
        dataset = EEGSessionDataset(dataset)

    if config.get("generate_graph", None) is not None:
        if not isinstance(dataset, EEGDataset):
            # TODO: we can add support for EEGSessionDataset in the future
            raise ValueError("Graph generation is only supported for EEGDataset, not other datasets like EEGSessionDataset. Please do not use 'session' batch size if you want to generate a graph.")
        
        dataset = GraphDataset(
            root=config["graph_cache_name"],
            dataset=dataset,
            generate_graphs=config["generate_graph"],
            mode=mode
        )

    return dataset

def flatten(lst: list[list[int]]) -> list:
    return sum(lst, [])

def split_dataset_by_session(dataset: EEGDataset | GraphDataset | EEGSessionDataset, lengths):
    if isinstance(dataset, EEGSessionDataset):
        # Normal split is ok since EEGSessionDataset already groups by session
        return random_split(dataset, lengths)
    
    print("Splitting dataset by session to avoid data leakage.")
    if isinstance(dataset, GraphDataset):
        eeg_dataset = dataset.dataset
    else:
        eeg_dataset = dataset
    session_to_indices = group_indices_by_session(eeg_dataset.clips_df)
    split_indices = [flatten(list(s)) for s in random_split(session_to_indices, lengths)]
    return [Subset(dataset, indices) for indices in split_indices]

def k_fold_by_session(kf: KFold, dataset: EEGDataset | GraphDataset | EEGSessionDataset):
    if isinstance(dataset, EEGSessionDataset):
        # Normal split is ok since EEGSessionDataset already groups by session
        return kf.split(dataset)

    print("Splitting dataset by session to avoid data leakage.")
    if isinstance(dataset, GraphDataset):
        eeg_dataset = dataset.dataset
    else:
        eeg_dataset = dataset
    session_to_indices = group_indices_by_session(eeg_dataset.clips_df)
    split_indices = [(flatten([session_to_indices[i] for i in train]), flatten([session_to_indices[i] for i in test])) for train, test in kf.split(session_to_indices)]
    
    return split_indices

def prepare_batch(batch, device):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x_batch, y_batch = batch
        x_batch = x_batch.float().to(device)
        if isinstance(y_batch, torch.Tensor):
            y_batch = y_batch.float().unsqueeze(1).to(device)    
    elif isinstance(batch, Data):
        batch = batch.to(device)
        batch.x = batch.x.float()
        is_list_of_ints = isinstance(batch.y, list) and isinstance(batch.y[0], (int, np.integer))
        if is_list_of_ints:
            batch.y = torch.tensor(batch.y).float().unsqueeze(1).to(device)
        batch_x = batch
        batch_y = batch.y
    else:
        print(batch)
        raise ValueError(f"Unsupported batch type: {type(batch)}. Expected tuple or Data object.")

    return batch_x, batch_y
    

def get_loader(config, dataset, mode='train'):
    if isinstance(dataset[0], Data):
        # If dataset is made of PyG Data objects, use PyG DataLoader
        return PygDataLoader(
            dataset,
            batch_size=(None if config["batch_size"] == 'session' else config["batch_size"]),
            shuffle=(mode == 'train')
        )
    return DataLoader(
        dataset,
        # Disable automatic batching if batch_size is 'session'
        batch_size=(None if config["batch_size"] == 'session' else config["batch_size"]),
        shuffle=(mode == 'train')
    )

def get_criterion(config):
    return config["criterion_fn"]()

def get_optimizer(model, config):
    if config.get("optimizer") is not None:
        optimizer = config["optimizer"](model.parameters())
    else:
        optimizer = optim.Adam(model.parameters() , lr=config["lr"])

    return optimizer

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
            x_batch, x_ids = prepare_batch(batch, device)

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