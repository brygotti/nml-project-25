import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import numpy as np
from torchinfo import summary

from models.ModelWrapper import get_model
from datasets.EEGSessionDataset import EEGSessionDataset
from utils import *

# ===============================
# Training Function
# ===============================

def train_one_epoch(model, criterion, optimizer, loader, device):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        criterion (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        loader (DataLoader): The data loader for training data.
        device (torch.device): The device to run the training on.
    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for batch in loader:
        x_batch, y_batch = prepare_batch(batch, device)

        logits = model(x_batch)
        if hasattr(model, 'custom_loss') and callable(model.custom_loss):
            # Allow for custom loss logic
            loss = model.custom_loss(criterion, logits, y_batch)
        else:
            loss = criterion(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    return avg_loss

class EarlyStopping:
    def __init__(self, patience, delta_tolerance, greater_is_better):
        """
        Early stopping mechanism to stop training when a monitored metric stops improving.
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            delta_tolerance (float): Deltas in metric higher than that will be considered as "no improvement" and trigger early stopping
            greater_is_better (bool): Whether a higher value of the metric is better.
        """
        self.patience = patience
        self.delta_tolerance = delta_tolerance
        self.greater_is_better = greater_is_better
        self.counter = 0
        self.min_metric = float('inf')

    def __call__(self, metric):
        if self.greater_is_better:
            metric = -metric
        if metric < self.min_metric:
            self.min_metric = metric
            self.counter = 0
        elif metric > (self.min_metric + self.delta_tolerance):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ===============================
# Evaluation Function
# ===============================

def evaluate_model(model, loader, device):
    """
    Evaluate the model on the validation or test set.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): The data loader for validation or test data.
        device (torch.device): The device to run the evaluation on.
    Returns:
        dict: A dictionary containing evaluation metrics (accuracy, precision, recall, f1_score).
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch = prepare_batch(batch, device)

            logits = model(x_batch)

            if hasattr(model, 'custom_predict') and callable(model.custom_predict):
                # Allow for custom prediction logic
                preds = model.custom_predict(logits)
            else:
                # Assuming sigmoid activation for binary classification
                preds = (logits > 0)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.int().cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average="macro", zero_division=0)
    }


# ===============================
# K-Fold Training Pipeline
# ===============================

def train_pipeline(config, device):
    """
    Main training pipeline that handles early stopping, K-fold cross-validation, and final training on the full dataset.
    Args:
        config (dict): Configuration dictionary containing model, dataset, training parameters, etc.
        device (torch.device): The device to run the training on.
    Returns:
        torch.nn.Module: The trained model.
    """
    seed_everything(42)
    
    dataset = get_dataset(config)

    if config.get("early_stopping", None) is not None:
        print(f"\n===== Early Stopping =====")
        print(f"Estimating number of epochs with early stopping on {config['early_stopping']['validation_size']} of data for a max of {config['early_stopping']['max_epochs']} epochs.")
        early_stopping = EarlyStopping(patience=config["early_stopping"]["patience"], 
                                       delta_tolerance=config["early_stopping"]["delta_tolerance"],
                                       greater_is_better=config["early_stopping"]["greater_is_better"])
        model = get_model(config["model"], config.get("model_params", {}), device)
        criterion = get_criterion(config)
        optimizer = get_optimizer(model, config)
        train_subset, val_subset = split_dataset_by_session(dataset, [1 - config["early_stopping"]["validation_size"], config["early_stopping"]["validation_size"]])
        train_loader = get_loader(config, train_subset, mode='train')
        val_loader = get_loader(config, val_subset, mode='val')

        train_losses = []
        all_metrics = []
        for epoch in tqdm(range(config["early_stopping"]["max_epochs"]), desc="Epochs"):
            avg_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
            train_losses.append(avg_loss)
            metrics = evaluate_model(model, val_loader, device)
            all_metrics.append(metrics)
            metric_name = config["early_stopping"]["metric"]
            if early_stopping(metrics[metric_name]):
                # Set to previous epoch
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break
        # TODO: plot train_losses ?
        early_stopping_metrics = [m[metric_name] for m in all_metrics]
        best_idx = np.argmax(early_stopping_metrics) if config["early_stopping"]["greater_is_better"] else np.argmin(early_stopping_metrics)
        best_metrics = all_metrics[best_idx]
        print("\n=== Best Metrics During Early Stopping ===")
        print(best_metrics)
        num_epochs = best_idx + 1
    else:
        print("Skipping early stopping epochs estimation.")
        num_epochs = config["num_epochs"]

    if config.get("k_folds", None) is not None:
        kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=42)
        all_metrics = []
        for fold, (train_idx, val_idx) in enumerate(k_fold_by_session(kf, dataset)):
            model = get_model(config["model"], config.get("model_params", {}), device)
            criterion = get_criterion(config)
            optimizer = get_optimizer(model, config)

            print(f"\n===== Fold {fold + 1} =====")
            # Train
            train_subset = Subset(dataset, train_idx)
            train_loader = get_loader(config, train_subset, mode='train')
            train_losses = []
            for epoch in tqdm(range(num_epochs), desc="Epochs"):
                avg_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
                train_losses.append(avg_loss)
            # TODO: plot train_losses ?

            # Evaluate
            val_subset = Subset(dataset, val_idx)
            val_loader = get_loader(config, val_subset, mode='val')
            metrics = evaluate_model(model, val_loader, device)
            print(f"Fold {fold + 1} Metrics:", metrics)
            all_metrics.append(metrics)

        # Aggregate metrics across folds
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        print("\n=== Average Metrics Across Folds ===")
        print(avg_metrics)
    else:
        print("Skipping K-fold cross validation. Training directly on the full dataset.")

    # Retrain on the full dataset
    print(f"\n===== Full Training =====")
    print(f"Training on the full dataset for {num_epochs} epochs.")
    train_loader = get_loader(config, dataset, mode='train')
    model = get_model(config["model"], config.get("model_params", {}), device)
    criterion = get_criterion(config)
    optimizer = get_optimizer(model, config)

    train_losses = []
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        avg_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        train_losses.append(avg_loss)
    # TODO: plot train_losses ?

    return model 



