# Pipeline for the project of EE-452 - Graph-based EEG Analysis

## Requirements
To run the pipeline, please install the conda environment from the `env.yml` file:

```bash
conda env create -f env.yml
conda activate nml
```

## Running the pipeline
You can edit the `main.py`, it contains a configuration object that allows you to set the parameters for the pipeline. You can choose the model, preprocessing, caching, hyperparameters, and more.
Make sure to adjust the paths to your data and results directories as needed.
To run the pipeline, execute the following command:

```bash
python main.py
```

This will start the training and evaluation process based on the configuration provided in `main.py`. It will then generate the submission file for the Kaggle competition.

## Examples of configurations

### Baseline LSTM
```python
CONFIG = {
    "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": 512, # Can be set to "session" to use each recording session as a batch
    "num_epochs": 100, # Ignored if early stopping is enabled
    "optimizer": None, # Defaults to Adam with learning rate config["lr"]
    "lr": 1e-3,
    "model": "SimpleLSTM",
    "criterion_fn": lambda weight_pos_class: nn.BCEWithLogitsLoss(), # weight_pos_class can be used in the loss function to reweight the loss for the positive class
    "signal_transform": fft_filtering,
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 20,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.05      # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    }
}
```

### Temporal LSTM
```python
CONFIG = {
    "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": "session", # Can be set to "session" to use each recording session as a batch
    "num_epochs": 100, # Ignored if early stopping is enabled
    "optimizer": None, # Defaults to Adam with learning rate config["lr"]
    "lr": 1e-4,
    "model": "TemporalLSTMCombinedChannels",
    "criterion_fn": lambda weight_pos_class: nn.BCEWithLogitsLoss(), # weight_pos_class can be used in the loss function to reweight the loss for the positive class
    "signal_transform": frequency_bands_features,
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 20,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.05      # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    }
}
```

### Conformer
```python
CONFIG = {
    "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": 256, # Can be set to "session" to use each recording session as a batch
    "num_epochs": 100, # Ignored if early stopping is enabled
    "optimizer": None, # Defaults to Adam with learning rate config["lr"]
    "lr": 2e-4,
    "model": "Conformer",
    "model_params": {'input_dim': 19, 'emb_size': 10, 'depth': 1},
    "signal_transform": time_filtering,
    "criterion_fn": lambda weight_pos_class: nn.BCEWithLogitsLoss(), # weight_pos_class can be used in the loss function to reweight the loss for the positive class
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 10,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.05      # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    }
}
```

### Graph convolutional network
```python
CONFIG = {
    "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "data",
    "batch_size": 512, # Can be set to "session" to use each recording session as a batch
    "num_epochs": 100, # Ignored if early stopping is enabled
    "optimizer": None, # Defaults to Adam with learning rate config["lr"]
    "lr": 1e-4,
    "model": "GCNN",
    "criterion_fn": lambda weight_pos_class: nn.BCEWithLogitsLoss(), # weight_pos_class can be used in the loss function to reweight the loss for the positive class
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 10,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.05      # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    },
    "graph_cache_name": "distances_fft_filtering",
    "generate_graph": lambda distances, nodes_order, x: generate_distances_graph(distances, nodes_order, fft_filtering(x, transpose=True)), # Set this if you want to work with graphs
    # The generate_graph function should return a Data object with edge_index and x defined.
}
```

### Graph SAGE
```python

CONFIG = {
    "k_folds": 5, # Number of folds for K-fold cross validation
    "data_path": "/home/ogut/data",
    "batch_size": 256, # Can be set to "session" to use each recording session as a batch
    "num_epochs": 100, # Ignored if early stopping is enabled
    "optimizer": None, # Defaults to Adam with learning rate config["lr"]
    "lr": 2e-3, # Increased learning rate for faster convergence
    "model": "GraphSage",
    "early_stopping": {
        "metric": "f1_score",        # Metric to monitor for early stopping
        "greater_is_better": True,   # Whether a higher value of the metric is better
        "validation_size": 0.2,      # Size of validation set for early stopping
        "max_epochs": 10000,         # Maximum epochs to train before early stopping
        "patience": 10,              # Number of epochs with no improvement after which training will be stopped
        "delta_tolerance": 0.05      # Deltas in F1 score higher than that will be considered as "no improvement" and trigger early stopping
    },
    "criterion_fn": lambda weight_pos_class: nn.BCEWithLogitsLoss(), # weight_pos_class can be used in the loss function to reweight the loss for the positive class
    "graph_cache_name": "sage_features",
    "generate_graph": lambda row, electrodes, positions_df, base_path, test: generate_single_graph(row, electrodes, positions_df, base_path, test),
}
```

### Iterative Gated Graph Convolutional Network
Due to the high complexity of this model, it could not be adapted to the pipeline. A custom training script for this model, as well as the model and preprocessing code can be found in the `IGCN` directory.