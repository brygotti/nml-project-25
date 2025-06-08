# Pipeline for the project EE-452 - Graph-based EEG Analysis

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