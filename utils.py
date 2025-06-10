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

from scipy.signal import butter, filtfilt


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility across various libraries.
    Args:
        seed (int): The seed value to use for random number generation.
    """
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
    """
    Get the EEG dataset based on the configuration and mode.
    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        mode (str): The mode of the dataset, either 'train', 'val', or 'test'.
    Returns:
        The dataset object for the specified mode.
    """
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
            mode=mode,
            model_name=config["model"]
        )

    return dataset

def flatten(lst: list[list[int]]) -> list:
    return sum(lst, [])

def split_dataset_by_session(dataset: EEGDataset | GraphDataset | EEGSessionDataset, lengths):
    """
    Splits the dataset by session to avoid data leakage.
    Args:
        dataset (EEGDataset | GraphDataset | EEGSessionDataset): The dataset to split.
        lengths (list[int]): The lengths for each split.
    Returns:
        list[Subset]: A list of subsets of the dataset, each corresponding to a split.
    """
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
    """
    Splits the dataset into K folds by session to avoid data leakage.
    Args:
        kf (KFold): The KFold object to use for splitting.
        dataset (EEGDataset | GraphDataset | EEGSessionDataset): The dataset to split.
    Returns:
        list[tuple[list[int], list[int]]]: A list of tuples, each containing train and test indices for each fold.
    """
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
    """
    Prepares the batch for training or evaluation by moving it to the specified device and converting types.
    Args:
        batch (tuple | list | Data): The batch to prepare, can be a tuple/list of x_batch, y_batch or a PyG Data object.
        device (torch.device): The device to move the batch to.
    Returns:
        tuple: A tuple containing the prepared x_batch and y_batch.
    """
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
        x_batch = batch
        y_batch = batch.y
    else:
        print(batch)
        raise ValueError(f"Unsupported batch type: {type(batch)}. Expected tuple, list or Data object.")

    return x_batch, y_batch
    

def get_loader(config, dataset, mode='train'):
    """
    Returns a DataLoader for the given dataset and mode.
    Args:
        config (dict): Configuration dictionary containing batch size and other parameters.
        dataset (EEGDataset | GraphDataset | EEGSessionDataset): The dataset to load.
        mode (str): The mode of the dataset, either 'train', 'val', or 'test'.
    Returns:
        DataLoader: A DataLoader for the dataset.
    """
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
    """
    Returns the loss function based on the configuration.
    Args:
        config (dict): Configuration dictionary containing the criterion function.
    Returns:
        callable: The loss function to use.
    """
    clips_tr = pd.read_parquet(Path(config["data_path"]) / "train/segments.parquet")

    n_negatives = clips_tr['label'].value_counts()[0]
    n_positives = clips_tr['label'].value_counts()[1]
    pos_weight = n_negatives / n_positives

    print(f"Number of negatives: {n_negatives}, Number of positives: {n_positives}")
    print(f"Positive weight for loss function: {pos_weight}")
    return config["criterion_fn"](pos_weight)

def get_optimizer(model, config):
    """
    Returns the optimizer for the model based on the configuration.
    Args:
        model (torch.nn.Module): The model to optimize.
        config (dict): Configuration dictionary containing optimizer parameters.
    Returns:
        torch.optim.Optimizer: The optimizer for the model.
    """
    if config.get("optimizer") is not None:
        optimizer = config["optimizer"](model.parameters())
    else:
        optimizer = optim.Adam(model.parameters() , lr=config["lr"])

    return optimizer

def create_submission(config, model, device, submission_name_csv='submission'):
    """
    Generates a Kaggle submission file based on the model's predictions on the test dataset.
    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        model (torch.nn.Module): The trained model to use for predictions.
        device (torch.device): The device to run the model on.
        submission_name_csv (str): The name of the submission CSV file to generate.
    """
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

def bandpass_filter(signal, low=0.5, high=70, fs=1000):
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, signal, axis=0)