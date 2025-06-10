from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from seiz_eeg.dataset import EEGDataset, to_arrays
from sklearn.preprocessing import MinMaxScaler
from mne_connectivity import spectral_connectivity_epochs
from sklearn.model_selection import train_test_split

import os
import torch


# ===============================
# Compute Feature Matrix
# ===============================
def compute_log_amp(x, sr=250):
    fft = np.fft.fft(x, axis=0)
    
    idx_pos = int(np.floor(fft.shape[0] / 2))
    fft =fft[:idx_pos, :]
    amp = np.abs(fft)
    amp[amp == 0.0] = 1e-8  # avoid log of 0
    log_amp = np.log(amp)
    
    return  log_amp

def compute_feature_matrix(config, clips, mode='train'):

    dataset = EEGDataset(
        clips,
        signals_root=Path(config["data_path"]) / f"{mode}",
        signal_transform=compute_log_amp,
        prefetch=True, 
        return_id=(mode == 'test') 
    )

    feature_matrix = to_arrays(dataset, pbar=False)[0]
    feature_matrix = np.transpose(feature_matrix, (0,2,1))
    scaler = MinMaxScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix
                                             .reshape(-1, feature_matrix.shape[-1])).reshape(feature_matrix.shape)
    
    labels = to_arrays(dataset, pbar=False)[1]
    return feature_matrix_scaled, labels
    


# ===============================
# Compute Adjacency Matrix 
# ===============================

def compute_spectral_connectivity(x, sr=250):
    log_amp = compute_log_amp(x, sr=250)
    
    # shape (1, channels, freqs)
    log_amp = np.transpose(log_amp, (1, 0))
    transf_fft = np.expand_dims(log_amp, axis=0)

    n_channels = transf_fft.shape[1]

    # index pairs (i,j) for i in 0..18, j in 0..18
    indices = ([], [])
    for i in range(n_channels):
        for j in range(n_channels):
            indices[0].append(i)
            indices[1].append(j)

    spec_conn = spectral_connectivity_epochs(
        data=transf_fft,
        method='coh',
        indices=indices,
        sfreq=sr,
        fmin=1.0,
        fmax=40.0,
        faverage=True,
        verbose=False
    )

    spec_conn_values = np.squeeze(spec_conn.get_data())  # shape: (n_pairs,)
    spec_conn_matrix = spec_conn_values.reshape(n_channels, n_channels)

    return spec_conn_matrix



def compute_adjacency_matrix(config, clips, mode='train'): 
    dataset = EEGDataset(
        clips,
        signals_root=Path(config["data_path"]) / f"{mode}",
        signal_transform=compute_spectral_connectivity,
        prefetch=True,  
        return_id=(mode == 'test')
    )
    adjacency_matrix = to_arrays(dataset, pbar=False)[0]

    scaler = MinMaxScaler()
    adjacency_matrix_scaled = scaler.fit_transform(adjacency_matrix.reshape(-1, adjacency_matrix.shape[-1])).reshape(adjacency_matrix.shape)
    return adjacency_matrix_scaled


# ===============================
# Preprocessing 
# ===============================

def split_test_val_dataset(config, mode='train'): 
    clips =pd.read_parquet(Path(config["data_path"]) / f"{mode}/segments.parquet")
    clips = clips.reset_index()
    session_ids = clips['session'].unique()

    train_sessions, val_sessions = train_test_split(
            session_ids, test_size=0.2, random_state=42
        )

    train_df = clips[clips['session'].isin(train_sessions)]
    val_df = clips[clips['session'].isin(val_sessions)]

    clips_tr = train_df.set_index(["patient", "session", "segment"])
    clips_val =val_df.set_index(["patient", "session", "segment"])

    return clips_tr, clips_val




def preprocessing_test_val_iggcn(config, mode='train'):
    save_dir = ".preprocessed/"
    os.makedirs(save_dir, exist_ok=True)

    clips_tr, clips_val = split_test_val_dataset(config, mode)

    print("Computing training features...")
    feature_tr, labels_tr = compute_feature_matrix(config, clips_tr)  
    print("Computing training adjacency...")
    adj_tr = compute_adjacency_matrix(config, clips_tr)

    print("Computing val features...")
    feature_val, labels_val = compute_feature_matrix(config, clips_val)
    print("Computing val adjacency...")
    adj_val = compute_adjacency_matrix(config, clips_val)
    
    # convert into tensor
    feature_tr = torch.tensor(feature_tr, dtype=torch.float32)
    adj_tr = torch.tensor(adj_tr, dtype=torch.float32)
    labels_tr = torch.tensor(labels_tr, dtype=torch.long)

    feature_val = torch.tensor(feature_val, dtype=torch.float32)
    adj_val = torch.tensor(adj_val, dtype=torch.float32)
    labels_val = torch.tensor(labels_val, dtype=torch.long)

    # Save tensors
    torch.save(feature_tr, os.path.join(save_dir, "feature_train.pt"))
    torch.save(adj_tr, os.path.join(save_dir, "adjacency_train.pt"))
    torch.save(labels_tr, os.path.join(save_dir, "labels_train.pt"))

    torch.save(feature_val, os.path.join(save_dir, "feature_val.pt"))
    torch.save(adj_val, os.path.join(save_dir, "adjacency_val.pt"))
    torch.save(labels_val, os.path.join(save_dir, "labels_val.pt"))

    print(f"Saved processed data to: {save_dir}")
    

    return feature_tr, adj_tr, feature_val, adj_val, labels_tr, labels_val
    













