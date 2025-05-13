
from models.SimpleLSTM import SimpleLSTM
from models.EEGNet import EEGNet


# Add your model here
def get_model(model_name='SimpleLSTM', model_params=None, device=None):
    if model_name == 'SimpleLSTM':
        return SimpleLSTM(**model_params).to(device)
    elif model_name == 'EEGNet':
        return EEGNet(
            input_dim=model_params["input_dim"],
            num_samples=model_params["num_samples"],
            dropout=model_params.get("dropout", 0.5),
            num_classes=1
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

