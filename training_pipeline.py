import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold

from models.ModelWrapper import get_model
from utils import *

# ===============================
# Training Function
# ===============================

def train_one_fold(model, loader, device, config):

    pos_count = sum([y for _, y in loader.dataset])
    neg_count = len(loader.dataset) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    # add pos_weight to the criterion to handle class imbalance
    if config["model"] == "EEGNet" or config["model"] == "NeuroGNN":
        # For EEGNet, we use BCEWithLogitsLoss
        criterion = config["criterion_fn"](pos_weight=pos_weight)
    else:
        # For other models, we can use the provided criterion function
        # Ensure the criterion function is compatible with the model
        criterion = config["criterion_fn"](t)


    optimizer = optim.Adam(model.parameters() , lr=config["lr"])
    train_losses = []

    model.train()
    for epoch in tqdm(range(config["num_epochs"]), desc="Epochs"):
        running_loss = 0.0
        for x_batch, y_batch in loader:
            y_batch = y_batch.float().unsqueeze(1).to(device)

            if config["model"] == "NeuroGNN":
                graph_seq = [g.to(device) for g in x_batch]
                logits = model(graph_seq)
            else:
                x_batch = x_batch.float().to(device)

                # Special case for EEGNet
                if config["model"] == "EEGNet":
                    if x_batch.dim() == 3:
                        x_batch = x_batch.unsqueeze(1)
                    elif x_batch.dim() == 4 and x_batch.shape[1] != 1:
                        x_batch = x_batch.permute(0, 3, 1, 2)

                logits = model(x_batch)

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
    y_true, y_probs = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_batch = y_batch.float().unsqueeze(1).to(device)

            if model.__class__.__name__ == "NeuroGNN":
                graph_seq = [g.to(device) for g in x_batch]
                logits = model(graph_seq)
            else:
                x_batch = x_batch.float().to(device)

                if model.__class__.__name__ == "EEGNet":
                    if x_batch.dim() == 3:
                        x_batch = x_batch.unsqueeze(1)
                    elif x_batch.dim() == 4 and x_batch.shape[1] != 1:
                        x_batch = x_batch.permute(0, 3, 1, 2)

                logits = model(x_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
    y_true = np.array(y_true).flatten()
    y_probs = np.array(y_probs).flatten()
    if len(y_probs) == 0 or len(y_true) == 0:
        print("No predictions made. Returning default metrics.")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'threshold': 0.5
        }

    best_th, best_f1 = 0.5, 0
    for th in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs > th).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    y_pred_final = (y_probs > best_th).astype(int)


    return {
        'accuracy': accuracy_score(y_true, y_pred_final),
        'precision': precision_score(y_true, y_pred_final, zero_division=0),
        'recall': recall_score(y_true, y_pred_final, zero_division=0),
        'f1_score': best_f1,
        'threshold': best_th
    }


# ===============================
# K-Fold Training Pipeline
# ===============================

def train_pipeline(config, device):
    seed_everything(42)
    
    dataset = get_dataset(config)
    
    if config["model"] == "NeuroGNN":
        y_all = [y for _, y in dataset]
        kf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=42)
    elif config["model"] == "EEGNet":
        y_all = [y for _, y in dataset]
        kf = StratifiedKFold(n_splits=config["k_folds"], shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=config["k_folds"], shuffle=True, random_state=42)

    all_metrics = []

    enum = enumerate(kf.split(dataset, y_all) if config["model"] in ["EEGNet", "NeuroGNN"] else kf.split(dataset))

    for fold, (train_idx, val_idx) in enum:
        print(f"\n===== Fold {fold + 1} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # GNN requires custom collate function
        if config["model"] == "NeuroGNN":
            train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, collate_fn=graph_sequence_collate)
            val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, collate_fn=graph_sequence_collate)
        else:
            train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)

        # Initialize model
        model = get_model(config["model"], config["model_params"], device)

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



