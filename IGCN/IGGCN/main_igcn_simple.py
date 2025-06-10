import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from IGCNTrainer import *
from IGGCNProcessing import *


from utils import create_submission_igcn, EEGPreprocessedDataset
    


config = {
    "data_path": "/home/ogut/data"
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #train_feature, train_adjacency, val_feature, val_adjacency, train_labels, val_labels = preprocessing_test_val_iggcn(config, mode='train')  
    
    train_feature = torch.load(".preprocessed/feature_train.pt")
    train_adjacency = torch.load(".preprocessed/adjacency_train.pt")
    train_labels = torch.load(".preprocessed/labels_train.pt")


    val_feature = torch.load(".preprocessed/feature_val.pt")
    val_adjacency = torch.load(".preprocessed/adjacency_val.pt")
    val_labels = torch.load(".preprocessed/labels_val.pt")


    train_dataset = EEGPreprocessedDataset(train_feature, train_adjacency, train_labels)
    val_dataset = EEGPreprocessedDataset(val_feature, val_adjacency, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
 

    input_dim = 1500
    hidden_dim = 19
    output_dim = 1
    model = IGGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, heads=4, max_iters=2, delta=1e-3)
    print("Training...")
    model.fit(train_loader, epochs=5, lr=1e-3)
    torch.save(model.state_dict(), "model_iggcn_20_4.pth")


    print("Evaluating...")
    model.evaluate(val_loader)
    #torch.save(model.state_dict(), "model_iggcn_20_4.pth")

    create_submission_igcn(config, model, device)

    
