import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
import numpy as np

from models.ModelWrapper import get_model
from utils import *

# ===============================
# Training Function
# ===============================

def train_one_fold(model, loader, device, config):
    criterion = config["criterion_fn"]()
    optimizer = optim.Adam(model.parameters() , lr=config["lr"])
    train_losses = []

    model.train()
    for epoch in tqdm(range(config["num_epochs"]), desc="Epochs"):
        running_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)

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
        train_losses.append(avg_loss)

    return model, train_losses


# ===============================
# Evaluation Function
# ===============================

def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)

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
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }


# ===============================
# K-Fold Training Pipeline
# ===============================

def train_pipeline(config, device):
    seed_everything(42)
    
    dataset = get_dataset(config)
    kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=42)

    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n===== Fold {fold + 1} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = get_loader(config, train_subset, mode='train')
        val_loader = get_loader(config, val_subset, mode='val')

        # Initialize model
        model= get_model(config["model"], config.get("model_params", {}), device)

        # Train
        model, _ = train_one_fold(model, train_loader, device, config)

        # Evaluate
        metrics = evaluate_model(model, val_loader, device)
        print(f"Fold {fold + 1} Metrics:", metrics)
        all_metrics.append(metrics)

    # Aggregate metrics across folds
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\n=== Average Metrics Across Folds ===")
    print(avg_metrics)

    return model, avg_metrics 



