import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score
from GraphLearner import *
from GGNN import *
from utils import to_cuda
from Loss import *

from sklearn.preprocessing import MinMaxScaler


def minmax_normalize_adjacency(adj):
    B, C, _ = adj.shape

    adj_flat = adj.view(B, -1)  # [B, C*C]
    adj_min = adj_flat.min(dim=1, keepdim=True)[0]  # [B, 1]
    adj_max = adj_flat.max(dim=1, keepdim=True)[0]  # [B, 1]

    adj_norm_flat = (adj_flat - adj_min) / (adj_max - adj_min + 1e-8)  # avoid div-by-zero
    adj_norm = adj_norm_flat.view(B, C, C)  # back to [B, C, C]

    return adj_norm

def update_adjacency(A_1_norm, A_m, L_0): 
    lambda_ = 0.3
    eta = 0.4
    
    A_m_norm = minmax_normalize_adjacency(A_m)
    A_m_upd = lambda_ * L_0 + (1 - eta)* (eta * A_m_norm + (1-eta)* A_1_norm)
    return A_m_upd

class IGGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, heads=4, max_iters=3, delta=1e-3, device=None):
        super().__init__()
        self.device = device
        self.graph_learner= GraphLearner(input_dim, heads, hidden_dim)
        self.ggnn = GGNN(input_dim, hidden_dim)
        self.max_iters = max_iters
        self.delta = delta

    def forward(self, X, A_0):
        H_m = X
        A_m = A_0
        A_1 = None
        L_0 = compute_normalized_laplacian(A_0)

        losses = []
        stop_cond = False
        for m in range(self.max_iters):

            #1) update A_m with graph_learner
            A_m_old=A_m
            A_m = self.graph_learner(H_m, A_m)
            
            if m ==0:
                A_1 = minmax_normalize_adjacency(A_m)

            else: 
                A_m = update_adjacency(A_1, A_m, L_0)
            
          

            #2) update H_m with ggnn 
            H_m, logits= self.ggnn(H_m, A_m)
            
            """
            loss_fn = nn.BCEWithLogitsLoss()
            pred_loss = loss_fn(logits.squeeze(1), self.y_batch.float())
            """
            
            focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
            pred_loss = focal_loss(logits.squeeze(1), self.y_batch.float())


            L_G, stop_cond = graph_regularization_loss(A_m_old, A_m, X, A_1, self.delta, self.device)
            loss = pred_loss + L_G
            losses.append(loss)

            

            if stop_cond:
                break

        return logits, sum(losses) / len(losses)

    def fit(self, dataloader, epochs=5, lr=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        for epoch in tqdm(range(epochs), desc="Epochs"):
            total_loss = 0
            for X_batch, y_batch, A_init in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
                self.y_batch = y_batch  # Set for use in forward()
                logits, loss = self.forward(X_batch.to(self.device), A_init.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")


    def evaluate(self, dataloader):
        self.eval()
        correct = total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch, A_init in dataloader:
                self.y_batch = y_batch
                logits, _ = self.forward(X_batch, A_init)
                probs = torch.sigmoid(logits)          
                preds = (probs >= 0.5).long()  

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        acc = correct / total
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Evaluation Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    
