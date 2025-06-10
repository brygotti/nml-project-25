
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from IGGCNProcessing import *
from IGCNTrainer import *
from torch.utils.data import Dataset

class EEGPreprocessedDataset(Dataset):
    def __init__(self, features, adjacencies, labels=None):
        """
        features: [num_samples, num_nodes, feature_dim]
        adjacencies: [num_samples, num_nodes, num_nodes]
        labels: [num_samples]
        """
        self.features = features
        self.adjacencies = adjacencies
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.adjacencies[idx]

def batch_normalize_adj(mx, mask=None):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    # mx: shape: [batch_size, N, N]

    # strategy 1)
    # rowsum = mx.sum(1)
    # r_inv_sqrt = torch.pow(rowsum, -0.5)
    # r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0. # I got this error: copy_if failed to synchronize: device-side assert triggered

    # strategy 2)
    rowsum = torch.clamp(mx.sum(1), min=1e-12)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    if mask is not None:
        r_inv_sqrt = r_inv_sqrt * mask

    r_mat_inv_sqrt = []
    for i in range(r_inv_sqrt.size(0)):
        r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))

    r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def create_submission_igcn(config, model, device, submission_name_csv='submission'):

        clips_test = pd.read_parquet(Path(config["data_path"]) / f"test/segments.parquet")
        adj = compute_adjacency_matrix(config, clips_test, mode='test')
        features, ids= compute_feature_matrix(config, clips_test, mode='test')

        adj = torch.tensor(adj, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
        
        dataset_test = EEGPreprocessedDataset(features, adj, ids)

        loader_te = DataLoader(dataset_test, batch_size=64, shuffle=False)

         # Set the model to evaluation mode
        model.eval()

        # Lists to store sample IDs and predictions
        all_predictions = []
        all_ids = []

        # Disable gradient computation for inference
        with torch.no_grad():
            for x_batch, x_ids, A_init in loader_te:
                # Assume each batch returns a tuple (x_batch, sample_id)
                # If your dataset does not provide IDs, you can generate them based on the batch index.

                # Move the input data to the device (GPU or CPU)
                x_batch = x_batch.float().to(device)

                # Perform the forward pass to get the model's output logits
                logits = model(x_batch.to(device), A_init.to(device))

                
                probs = torch.sigmoid(logits)          
                predictions = (probs >= 0.5).long()  

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